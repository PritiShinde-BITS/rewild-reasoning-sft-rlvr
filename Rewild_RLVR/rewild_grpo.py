# %%
from importlib.metadata import version

used_libraries = [
    "reasoning_from_scratch",
    "torch",
    "tokenizers"  # Used by reasoning_from_scratch 
]

for lib in used_libraries:
    print(f"{lib} version: {version(lib)}")

# %%
import torch
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer

device = get_device()
model, tokenizer = load_model_and_tokenizer(
    which_model="base", device=device, use_compile=False
)
print("Loaded Qwen3-0.6B on", device)


# %% [markdown]
# &nbsp;
# ## Loading ReWild train/test splits
# 

# %% [markdown]
# - ReWild contains standalone and hybrid questions. This notebook uses the official train/test files from the public dataset.
# - **Scope:** GRPO optimizes **final-answer correctness** without tool execution. Standalone `gsm8k`, `folio`, and `reclor` plus optional hybrid questions are supported. `proscript` requires a graph/F1 checker and is excluded; full trajectory/tactic/step evaluation requires the original ReWild runner.
# 

# %%
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import random

REWILD_REPO = "yuan-yang/ReWild"
# These have an unambiguous answer verifier for the simple GRPO loop.
# proscript uses a graph F1 verifier, which is not implemented here.
STANDALONE_DATASETS = ("gsm8k", "folio", "reclor")
INCLUDE_HYBRID = True  # Set False to start with standalone questions only.

def load_rewild(split="train", include_hybrid=INCLUDE_HYBRID, local_dir="rewild_data"):
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    filenames = [f"standalone_{split}.json"]
    if include_hybrid:
        filenames.append(f"hybrid_{split}.json")

    examples = []
    skipped = 0
    for filename in filenames:
        local_path = hf_hub_download(
            repo_id=REWILD_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=local_dir,
        )
        with Path(local_path).open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list in {filename}")

        is_hybrid = filename.startswith("hybrid_")
        for row in rows:
            dataset = "hybrid" if is_hybrid else str(row.get("dataset", "")).lower()
            if not is_hybrid and dataset not in STANDALONE_DATASETS:
                skipped += 1
                continue
            problem, answer = row.get("input"), row.get("label")
            if not isinstance(problem, str) or not problem.strip() or answer is None:
                skipped += 1
                continue
            if isinstance(answer, (dict, list)):
                skipped += 1
                continue

            # GRPO needs only a problem and verified target answer.
            # ReWild's original trajectory remains in the downloaded JSON file.
            examples.append({
                "problem": problem.strip(),
                "answer": str(answer).strip(),
                "dataset": dataset,
                "split": split,
            })

    random.Random(42).shuffle(examples)
    if not examples:
        raise ValueError(f"No usable ReWild examples for split={split}")
    print(f"ReWild {split}: {len(examples)} loaded; {skipped} excluded")
    return examples


# %%
rewild_train = load_rewild("train")
rewild_test = load_rewild("test")

from collections import Counter
print("Train categories:", Counter(e["dataset"] for e in rewild_train))
print("Test categories:", Counter(e["dataset"] for e in rewild_test))

# Avoid unintentional train/test leakage if source splits share a question.
train_problems = {e["problem"] for e in rewild_train}
overlapping = sum(e["problem"] in train_problems for e in rewild_test)
if overlapping:
    print(f"WARNING: {overlapping} test rows also occur in training. Excluding them.")
    rewild_test = [e for e in rewild_test if e["problem"] not in train_problems]
assert rewild_test, "No held-out test examples remain"


# %%
from pprint import pprint
pprint(rewild_train[0])


# %%
def render_rewild_prompt(example):
    dataset = example["dataset"]

    if dataset == "gsm8k":
        instruction = (
            "Solve the math problem carefully. "
            "Check the arithmetic. "
            "Give the final numerical answer."
        )

    elif dataset == "folio":
        instruction = (
            "Determine whether the hypothesis follows from "
            "the premises. Use only the given premises. "
            "Answer with exactly one of: "
            "Agree, Contradict, Uncertain."
        )

    elif dataset == "reclor":
        instruction = (
            "Read the argument and evaluate all options. "
            "Choose the single best answer. "
            "Return its option number (1, 2, 3, or 4)."
        )

    elif dataset == "hybrid":
        instruction = (
            "Solve each relevant subproblem carefully. "
            "Combine the results and compare all options. "
            "Choose the single correct option number."
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return (
        "You are a careful reasoning assistant.\n"
        f"{instruction}\n"
        "Keep your reasoning concise.\n"
        "Finish with the final answer in the format "
        "\\boxed{ANSWER}.\n\n"
        f"Question:\n{example['problem']}\n\n"
        "Answer:"
    )

# %% [markdown]
# &nbsp;
# ## Sampling rollouts

# %%
from reasoning_from_scratch.qwen3 import KVCache
from reasoning_from_scratch.ch04 import top_p_filter


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
        )

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for step in range(max_new_tokens):
        if temperature == 0:  # Greedy, repeatable predictions for evaluation
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            if temperature != 1.0:
                logits = logits / temperature

            probas = torch.softmax(logits, dim=-1)
            probas = top_p_filter(probas, top_p)
            next_token = torch.multinomial(
                probas.cpu(), num_samples=1
            ).to(device)

        token_id = next_token.item()
        generated.append(token_id)

        if (
            tokenizer.eos_token_id is not None
            and token_id == tokenizer.eos_token_id
        ):
            break
        if step + 1 < max_new_tokens:  # Skips the forward pass after the last token is generated
            logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)

# %% [markdown]
# &nbsp;
# ## Calculating rewards

# %%
import re
from reasoning_from_scratch.ch03 import (
    extract_final_candidate, grade_answer
)


def extract_rewild_answer(answer_text, dataset):
    """Get a conservative final answer and track strict boxed compliance.

    Plain-text fallbacks are intentionally anchored to the FIRST nonempty line.
    We do not search arbitrary reasoning text for a matching label or number.
    """
    cleaned = re.sub(r"<\|[^>]+?\|>", "", answer_text or "").strip()
    boxed = extract_final_candidate(cleaned, fallback=None)
    if boxed:
        return str(boxed).strip(), True

    first = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first = re.sub(r"^[\s>*_`]+", "", first)

    if dataset == "folio":
        match = re.match(r"^(Agree|Contradict|Uncertain)(?=$|[\s.,!?*_`])", first, re.I)
        return (match.group(1) if match else ""), False

    if dataset in ("reclor", "hybrid"):
        match = re.match(r"^([1-9]\d{0,2})\s*[.)](?=\s|$)", first)
        if match:
            return match.group(1), False
        # Accept a single option number without the trailing period.
        match = re.fullmatch(r"([1-9]\d{0,2})[*_`]*", first)
        return (match.group(1) if match else ""), False

    if dataset == "gsm8k":
        # Numeric first-line answers only: no arbitrary number extraction from a derivation.
        match = re.fullmatch(
            r"(?:(?:[Ff]inal answer|[Tt]he answer is)\s*[:=]?\s*)?"
            r"([-+]?\$?\d[\d,]*(?:\.\d+)?)"
            r"(?:\s+[A-Za-z%]+)?\s*[.!]?",
            first,
        )
        return (match.group(1) if match else ""), False

    raise ValueError(f"Unknown verifier for dataset={dataset}")



# %%
def reward_rlvr(answer_text, ground_truth, dataset="gsm8k"):
    extracted, is_boxed = extract_rewild_answer(answer_text, dataset)
    
    # Base format reward
    format_reward = 0.1 if is_boxed else 0.0
    
    if not extracted:
        return format_reward

    ground_truth = str(ground_truth).strip()
    correct = False
    
    if dataset == "gsm8k":
        ground_truth = ground_truth.split("####")[-1].strip()
        correct = bool(grade_answer(extracted, ground_truth))
    elif dataset == "folio":
        valid = {"agree", "contradict", "uncertain"}
        correct = (extracted.casefold() in valid and extracted.casefold() == ground_truth.casefold())
    elif dataset in ("reclor", "hybrid"):
        if extracted.isdecimal() and ground_truth.isdecimal():
            answer = int(extracted)
            if dataset == "reclor" and 1 <= answer <= 4:
                correct = (answer == int(ground_truth))
            elif dataset == "hybrid":
                correct = (answer == int(ground_truth))

    # Scale correctness significantly higher than formatting
    correctness_reward = 1.0 if correct else 0.0
    return correctness_reward + format_reward


# %%
def sequence_logprob(model, token_ids, prompt_len):
    # Compute log-probabilities for generated tokens only.
    # This avoids allocating an extra full-vocabulary log_softmax for prompt tokens.
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    completion_logits = logits[prompt_len - 1:-1].float()
    completion_ids = token_ids[prompt_len:]
    return -torch.nn.functional.cross_entropy(
        completion_logits, completion_ids, reduction="sum"
    )


# %% [markdown]
# &nbsp;
# ## Putting everything together in a GRPO step

# %%
def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    backward_in_chunks=False,  # True during training: lower MPS peak memory
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    saved_rollouts = []
    prompt = render_rewild_prompt(example)

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        # Stage 1: generate rollouts
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Stage 2: compute rewards
        reward = reward_rlvr(text, example["answer"], example["dataset"])

   
        gen_len = token_ids.numel() - prompt_len
        if gen_len > 100:  # Adjust threshold based on dataset needs
            reward -= 0.002 * (gen_len - 100)  # Subtle penalty per excess token


        # Stage 4: score each rollout later to keep only one gradient graph.
        if backward_in_chunks:
            saved_rollouts.append((token_ids, prompt_len))
        else:
            logp = sequence_logprob(model, token_ids, prompt_len)
            roll_logps.append(logp)

        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    # Stage 2: collect all rewards
    rewards = torch.tensor(roll_rewards, device=device)

    # Stage 3: compute advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # Stage 4 and 5: compute logprobs and policy-gradient loss.
    if backward_in_chunks:
        # Backpropagate one rollout at a time, accumulating gradients.
        # This matches the original mean GRPO gradient without retaining
        # num_rollouts copies of the model's activation graph.
        loss_parts = []
        for (ids, p_len), advantage in zip(saved_rollouts, advantages):
            logp = sequence_logprob(model, ids, p_len)
            part_loss = -(advantage.detach() * logp) / num_rollouts
            part_loss.backward()
            loss_parts.append(part_loss.detach())
            del logp, part_loss
        pg_loss = torch.stack(loss_parts).sum()
    else:
        # Original educational path: keeps a differentiable loss tensor.
        logps = torch.stack(roll_logps)
        pg_loss = -(advantages.detach() * logps).mean()

    loss = pg_loss  # In the next chapter we add a KL term here
    if was_training:
        model.train()

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }

# %% [markdown]
# - The stages in the code comments map to the stages in the GRPO figure
# - Note that following stage 1, we have stages 2 and 4 (instead of 3 and 4) in the code comments, since this results in a simpler code implementation (so that we don't have to implement multiple for-loops)

# %% [markdown]
# &nbsp;
# ## Implementing the GRPO training loop

# %%
import time

def train_rlvr_grpo(
    model,
    tokenizer,
    train_data,
    device,
    steps=None,
    num_rollouts=2,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=".",
    csv_log_path=None,

):
    if steps is None:
        steps = len(train_data)

    # Stage 1: initialize optimizer
    # (the model was already initialized outside the function)
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=0.0,
    foreach=False)
    model.train()
    current_step = 0
    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)

    try:
        # Stage 2: Iterate over training steps
        for step in range(steps):

            # Stage 3: Reset loss gradient
            # (it's best practice to do this at the beginning of each step)
            optimizer.zero_grad(set_to_none=True)

            current_step = step + 1
            example = train_data[step % len(train_data)]

            # Stage 4: calculate GRPO loss
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                backward_in_chunks=True,  # Lower memory for MPS
            )

            # Stage 5: Gradients were accumulated per rollout above.
            # No second backward() call is needed.

            # Only update when GRPO has a reward contrast
            if max(stats["rewards"]) != min(stats["rewards"]):
            
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
            
                optimizer.step()
            
            else:
                print(
                    f"Step {current_step}: "
                    "No reward contrast, skipping update."
                )

            # Stage 7: Collect rewards, response lengths, and losses
            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(
                sample["gen_len"] for sample in stats["samples"]
            )
            avg_response_len = (
                step_tokens / len(stats["samples"]) 
                if stats["samples"] else 0.0
            )
            append_csv_metrics(
                csv_log_path, current_step, steps, stats["loss"],
                reward_avg, avg_response_len,
            )

            # Print step metrics
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )

            # Sample outputs (every 10 steps) to check if model
            # generates coherent text
            if current_step % 10 == 0:
                print(f"[Step {current_step}] sample outputs")
                for i, sample in enumerate(stats["samples"][:3]):
                    text = sample["text"].replace("\n", "\\n")
                    print(
                        f"  {i+1}) reward={sample['reward']:.3f} "
                        f"len={sample['gen_len']}: {text}"
                    )
                print()

            # Stage 8: Save model checkpoint
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")

    # Save a model checkpoint if we interrupt the training early
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model

    return model


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def append_csv_metrics(
    csv_log_path,
    step_idx,
    total_steps,
    loss,
    reward_avg,
    avg_response_len,
):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )

# %% [markdown]
# ### Baseline evaluation before GRPO training

# %%
import json
from collections import defaultdict

@torch.no_grad()
def evaluate_rewild(model, tokenizer, test_data, device, output_path,
                    max_new_tokens=192):
    model.eval()
    rows = []
    for idx, example in enumerate(test_data, 1):
        token_ids, prompt_len, response = sample_response(
            model, tokenizer, render_rewild_prompt(example), device,
            max_new_tokens=max_new_tokens, temperature=0, top_p=1.0,
        )
        pred, boxed = extract_rewild_answer(response, example["dataset"])
        correct = reward_rlvr(response, example["answer"], example["dataset"])
        rows.append({
            "dataset": example["dataset"],
            "problem": example["problem"],
            "ground_truth": example["answer"],
            "prediction": pred,
            "response": response,
            "correct": int(correct),
            "format_valid": bool(boxed),  # same field, now explicitly boxed compliance
            "answer_parse_valid": bool(pred),  # separate answer extraction measure
            "generated_tokens": int(token_ids.numel() - prompt_len),
        })
        if idx % 10 == 0 or idx == len(test_data):
            print(f"Evaluated {idx}/{len(test_data)}")

    totals = defaultdict(lambda: {
        "n": 0, "correct": 0, "valid_format": 0, "parseable": 0
    })
    for row in rows:
        for category in ("all", row["dataset"]):
            totals[category]["n"] += 1
            totals[category]["correct"] += row["correct"]
            totals[category]["valid_format"] += int(row["format_valid"])
            totals[category]["parseable"] += int(row["answer_parse_valid"])
    summary = {
        category: {
            **count,
            "accuracy": round(count["correct"] / count["n"], 4),
            "format_rate": round(count["valid_format"] / count["n"], 4),
            "parse_rate": round(count["parseable"] / count["n"], 4),
        }
        for category, count in totals.items()
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions: {output_path}")
    return summary

EVAL_N = 25      # Increase for final research results
TRAIN_STEPS = 50 # Increase after the notebook works end to end
MAX_NEW_TOKENS = 192
eval_examples = rewild_test[:EVAL_N]  # Fixed, held-out subset

before = evaluate_rewild(
    model, tokenizer, eval_examples, device,
    output_path="rewild_results/before.json",
    max_new_tokens=MAX_NEW_TOKENS,
)


# %%
hybrid_train = [
    x for x in rewild_train
    if x["dataset"] == "hybrid"
]

print("Available hybrid examples:", len(hybrid_train))

# Keep the dataset's fixed shuffled order
hybrid_train_small = hybrid_train[:30]

print("Selected for training:", len(hybrid_train_small))

# %%
import os

os.makedirs("rewild_results", exist_ok=True)

eval_examples = rewild_test[:25]

before = evaluate_rewild(
    model=model,
    tokenizer=tokenizer,
    test_data=eval_examples,
    device=device,
    output_path="rewild_results/before_hybrid_v2.json",
    max_new_tokens=128,
)

# %%
train_rlvr_grpo(
    model=model,
    tokenizer=tokenizer,
    train_data=hybrid_train_small,
    device=device,
    steps=5,
    num_rollouts=7,
    max_new_tokens=256,
    temperature=0.9,
    top_p=0.95,
    lr=5e-6,
    checkpoint_every=10,
    checkpoint_dir="rewild_checkpoints_hybrid",
    csv_log_path="rewild_results/train_hybrid_v2.csv",
)

# %% [markdown]
# - Note that this code does not support batched training
# - This is a deliberate choice to keep the code simpler and more readable, and because sampling multiple (potentially long) rollouts can already be very resource-intensive
# - However, if you have access to multiple GPUs, you can use the optional version of this code with batch and multi-GPU support that can be found in the supplementary materials at [../02_rlvr_grpo_scripts_intro](../02_rlvr_grpo_scripts_intro), which trains the model faster

# %% [markdown]
# &nbsp;
# ## Evaluating the fine-tuned model (held-out ReWild data)
# 

# %% [markdown]
# - Compare `before` and `after` on the exact same held-out samples; both use greedy decoding.
# - The CSV tracks training rewards; the JSON files save predictions and per-category final-answer accuracy. A positive change does not by itself establish improvement on the full ReWild program-execution benchmark.
# 

# %%
after = evaluate_rewild(
    model=model,
    tokenizer=tokenizer,
    test_data=eval_examples,
    device=device,
    output_path="rewild_results/after_hybrid_v2.json",
    max_new_tokens=128,
)

# %%
print("\nBEFORE vs AFTER GRPO")
print("-" * 60)

for dataset in ["all", "folio", "gsm8k", "hybrid", "reclor"]:

    if dataset not in before or dataset not in after:
        continue

    before_acc = before[dataset]["accuracy"] * 100
    after_acc = after[dataset]["accuracy"] * 100

    print(
        f"{dataset.upper():10s}: "
        f"{before_acc:5.1f}% -> {after_acc:5.1f}% "
        f"| Change: {after_acc - before_acc:+.1f} pp"
    )

# %% [markdown]
# - No code in this section


