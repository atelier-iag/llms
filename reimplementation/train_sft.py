"""Fine-tune the existing model on masked responses and measure general retention."""

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from evaluation.language_model import evaluate
from evaluation.sft import evaluate_responses
from reimplementation.checkpoint import load_checkpoint, save_checkpoint
from reimplementation.corpus_manifest import validate_manifest
from reimplementation.data import TokenFile
from reimplementation.model import CausalLanguageModel
from reimplementation.precision import validate_precision
from reimplementation.prepare_corpus import file_hash
from reimplementation.prepare_sft import balanced_select
from reimplementation.sft_data import IGNORE_INDEX, batches, collate, encode_example, load_sft
from reimplementation.sft_loss import evaluate_sft, sft_step
from reimplementation.tokenizer import DOLMA_TOKENIZER, load_tokenizer
from reimplementation.train_baseline import learning_rate


ROOT = Path(__file__).resolve().parents[1]


def select_evaluation_examples(rows, count, seed):
    # A complete diagnostic split needs no category-dependent subsampling.
    return list(rows) if count == len(rows) else balanced_select(rows, count, seed)


def run(config, data_dir, *, general_data_dir, probes_path, device, output_root,
        init_from=None, resume=None):
    if (init_from is None) == (resume is None):
        raise ValueError("provide exactly one of init_from or resume")
    validate_precision(config["precision"], device)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    tokenizer = load_tokenizer(DOLMA_TOKENIZER)
    if tokenizer.get_vocab_size() != config["model"]["vocab_size"]:
        raise ValueError("tokenizer vocabulary differs from configuration")
    splits, manifest = load_sft(data_dir, tokenizer=DOLMA_TOKENIZER,
                                vocab_size=config["model"]["vocab_size"], max_tokens=config["sequence_length"])
    train, val = splits["train"], splits["val"]
    for name in ("train", "val"):
        if len(splits[name]) != config["corpus"][f"{name}_examples"]:
            raise ValueError("unexpected SFT example budget")
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos != manifest["contents"]["eos_token_id"]:
        raise ValueError("SFT EOS differs from tokenizer")
    general_manifest = validate_manifest(
        general_data_dir, tokenizer=DOLMA_TOKENIZER, vocab_size=config["model"]["vocab_size"],
        required=config["general_evaluation"].get("require_manifest", True))
    general = TokenFile(Path(general_data_dir) / "val.npy", config["sequence_length"], config["model"]["vocab_size"])
    if len(general.tokens) != config["general_evaluation"]["validation_tokens"]:
        raise ValueError("unexpected general dev token count")
    offsets = general.evaluation_starts(config["evaluation_windows"])
    probes = []
    for probe in json.loads(Path(probes_path).read_text(encoding="utf-8")):
        row = encode_example(tokenizer, probe["instruction"], probe["context"], probe["response"], eos)
        if len(row["ids"]) > config["sequence_length"]:
            raise ValueError("diagnostic probe exceeds the context budget")
        probes.append({**row, "source_id": probe["id"], "category": probe["category"]})
    if not probes:
        raise ValueError("diagnostic probes must be nonempty")
    dev_generation = select_evaluation_examples(val, config["generation_examples"], config["seed"])
    train_sample = select_evaluation_examples(train, config["train_evaluation_examples"], config["seed"])
    identity = {"sft_manifest_sha256": manifest["sha256"],
                "general_manifest_sha256": general_manifest["sha256"] if general_manifest else None,
                "general_validation_sha256": file_hash(general.path), "probes_sha256": file_hash(probes_path)}
    batch_size = config["batch_size"]
    if batch_size < 1 or config["epochs"] < 1:
        raise ValueError("SFT batch size and epochs must be positive")
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = steps_per_epoch * config["epochs"]
    if not 0 <= config["warmup_updates"] < total_steps:
        raise ValueError("invalid SFT batch/epoch/warmup configuration")
    if (config["evaluate_every"] < 1 or config["log_every"] < 1 or config["max_new_tokens"] < 1
            or config["probe_max_new_tokens"] < 1
            or not 0 <= config["min_lr_ratio"] <= 1 or config["max_grad_norm"] <= 0):
        raise ValueError("invalid SFT controls")
    recovery = None
    start_step = seen = seen_examples = 0
    if resume is not None:
        recovery = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
        if recovery.get("kind") != "sft-recovery-v1" or recovery["config"] != config:
            raise ValueError("incompatible SFT recovery configuration")
        if recovery["data_identity"] != identity:
            raise ValueError("SFT resume data identity differs")
        if bool(recovery["cuda_rng_states"]) != (device == "cuda") or (
            device == "cuda" and len(recovery["cuda_rng_states"]) != torch.cuda.device_count()
        ):
            raise ValueError("SFT resume requires the same CPU/CUDA configuration")
        start_step = recovery["step"]
        if type(start_step) is not int or not 0 < start_step <= total_steps:
            raise ValueError("invalid SFT recovery step")
        # Validate saved coverage against the actual deterministic epoch order.
        for epoch in range(config["epochs"]):
            for index, (_, labels) in enumerate(batches(train, batch_size, eos, np.random.default_rng(config["seed"] + epoch))):
                if epoch * steps_per_epoch + index >= start_step:
                    break
                seen += int((labels != IGNORE_INDEX).sum())
                seen_examples += labels.shape[0]
        if seen != recovery["response_tokens"] or seen_examples != recovery["seen_examples"]:
            raise ValueError("SFT recovery coverage differs from data order")
        model = CausalLanguageModel(**config["model"]).to(device)
        model.load_state_dict(recovery.pop("model_state"), strict=True)
        initialization = recovery["initialization"]
    else:
        model, metadata = load_checkpoint(Path(init_from), device=device)
        if (metadata["model_config"] != config["model"] or metadata["tokenizer"] != DOLMA_TOKENIZER
                or metadata["context_length"] != config["sequence_length"]):
            raise ValueError("SFT initialization architecture/context/tokenizer differs")
        initialization = {"checkpoint": str(Path(init_from).resolve()), "sha256": file_hash(init_from),
                          "source_step": metadata["step"], "optimizer": "fresh AdamW"}
    # A completed recovery needs evaluation only, so keep Adam's state off the GPU.
    optimizer = (torch.optim.AdamW(model.parameters(), **config["optimizer"])
                 if start_step < total_steps else None)
    if recovery is not None:
        saved_optimizer = recovery.pop("optimizer_state")
        if optimizer is not None:
            optimizer.load_state_dict(saved_optimizer)
        del saved_optimizer
        torch.set_rng_state(recovery["torch_rng_state"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(recovery["cuda_rng_states"])
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="sft-", dir=output_root))

    def write_json(name, value):
        (run_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def event(record):
        line = json.dumps(record, allow_nan=False)
        with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def measure():
        return {"train_response": evaluate_sft(model, batches(train_sample, batch_size, eos)),
                "dev_response": evaluate_sft(model, batches(val, batch_size, eos)),
                "general_validation_sample": evaluate(model, (general.batch(offsets[i:i + batch_size])
                                                              for i in range(0, len(offsets), batch_size)))}

    def generations():
        dev, details = evaluate_responses(model, dev_generation, tokenizer, context_length=config["sequence_length"],
                                         max_new_tokens=config["max_new_tokens"], eos_id=eos)
        probe_metrics, probe_details = evaluate_responses(model, probes, tokenizer,
            context_length=config["sequence_length"], max_new_tokens=config["probe_max_new_tokens"], eos_id=eos)
        return {"dev_generation": dev, "dev_generation_details": details,
                "probes": probe_metrics, "probe_outputs": probe_details}

    write_json("config.json", config)
    start = time.perf_counter()
    event({"event": "start", "run_dir": str(run_dir), "total_updates": total_steps,
           "examples_per_epoch": len(train), "epochs": config["epochs"]})
    if recovery is None:
        event({"event": "initial_evaluation_start"})
        measured = measure()
        initial = {**measured, "general_validation_full": evaluate(model, general.epoch_batches(batch_size)),
                   **generations()}
        history = []
        evaluations = [{"step": 0, "response_tokens": 0, **measured}]
        window_nll = window_tokens = 0
        event({"event": "initial", **measured, "general_validation_full": initial["general_validation_full"],
               "dev_exact_matches": initial["dev_generation"]["exact_matches"],
               "probe_exact_matches": initial["probes"]["exact_matches"]})
    else:
        state = recovery["run_state"]
        initial, history, evaluations = state["initial"], state["training_history"], state["evaluation_history"]
        window_nll, window_tokens = state["window_nll"], state["window_tokens"]
        del recovery
        event({"event": "resume", "step": start_step, "response_tokens": seen})
    write_json("initial-generations.json", {"dev": initial["dev_generation_details"], "probes": initial["probe_outputs"]})

    def save_recovery(step):
        state = {"kind": "sft-recovery-v1", "config": config, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "step": step,
                 "response_tokens": seen, "seen_examples": seen_examples,
                 "data_identity": identity, "initialization": initialization,
                 "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_states": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                 "run_state": {"initial": initial, "training_history": history, "evaluation_history": evaluations,
                               "window_nll": window_nll, "window_tokens": window_tokens}}
        temporary = run_dir / "recovery.tmp"
        torch.save(state, temporary)
        temporary.replace(run_dir / "recovery.pt")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_seconds = 0
    synchronize()
    segment = time.perf_counter()
    for epoch in range(config["epochs"]):
        for index, batch in enumerate(batches(train, batch_size, eos, np.random.default_rng(config["seed"] + epoch))):
            step = epoch * steps_per_epoch + index + 1
            if step <= start_step:
                continue
            lr = learning_rate(step, total_steps, config["optimizer"]["lr"], config["warmup_updates"], config["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            inputs, labels = (tensor.to(device) for tensor in batch)
            count = int((labels != IGNORE_INDEX).sum())
            loss = sft_step(model, optimizer, (inputs, labels), precision=config["precision"], max_grad_norm=config["max_grad_norm"])
            seen += count
            seen_examples += inputs.shape[0]
            window_tokens += count
            window_nll += loss * count
            if step % config["log_every"] == 0 or step == total_steps:
                record = {"step": step, "response_tokens": seen, "seen_examples": seen_examples,
                          "mean_training_loss": window_nll / window_tokens, "window_response_tokens": window_tokens,
                          "lr": lr, "elapsed_seconds": time.perf_counter() - start}
                history.append(record)
                event({"event": "train", **record})
                window_nll = window_tokens = 0
            if step % config["evaluate_every"] == 0 or step == total_steps:
                synchronize()
                train_seconds += time.perf_counter() - segment
                record = {"step": step, "response_tokens": seen, **measure()}
                evaluations.append(record)
                event({"event": "evaluation", **record})
                save_recovery(step)
                synchronize()
                segment = time.perf_counter()
    expected = config["epochs"] * manifest["contents"]["files"]["train"]["response_tokens"]
    if seen != expected or seen_examples != config["epochs"] * len(train):
        raise RuntimeError("SFT did not cover the planned examples and response targets")
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    # Save the inference weights before the long evaluation and release training memory.
    checkpoint = run_dir / "model.pt"
    save_checkpoint(checkpoint, model, config["model"], step=total_steps,
                    context_length=config["sequence_length"], tokenizer=DOLMA_TOKENIZER)
    del optimizer
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    event({"event": "final_evaluation_start", "checkpoint": str(checkpoint)})
    final = {"dev_response": evaluate_sft(model, batches(val, batch_size, eos)),
             "general_validation_full": evaluate(model, general.epoch_batches(batch_size))}
    probe_inputs = collate(val[:1], eos)[0].to(device)
    model.eval()
    with torch.no_grad():
        expected_logits = model(probe_inputs).cpu()
    del model
    model, _ = load_checkpoint(checkpoint, device=device)
    with torch.no_grad():
        actual_logits = model(probe_inputs).cpu()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    final.update(generations())
    evaluation_peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    write_json("final-generations.json", {"dev": final["dev_generation_details"], "probes": final["probe_outputs"]})
    _, checked = load_sft(data_dir, tokenizer=DOLMA_TOKENIZER, vocab_size=config["model"]["vocab_size"], max_tokens=config["sequence_length"])
    if (checked["sha256"] != manifest["sha256"] or file_hash(general.path) != identity["general_validation_sha256"]
            or file_hash(probes_path) != identity["probes_sha256"]
            or (general_manifest and file_hash(general_manifest["path"]) != general_manifest["sha256"])):
        raise RuntimeError("SFT data or probes changed during training")
    if init_from is not None and file_hash(init_from) != initialization["sha256"]:
        raise RuntimeError("SFT initial checkpoint changed during training")
    session = {"start_step": start_step, "updates": total_steps - start_step,
               "training_seconds": train_seconds, "wall_seconds": time.perf_counter() - start,
               "peak_cuda_allocated_bytes": peak,
               "final_evaluation_peak_cuda_allocated_bytes": evaluation_peak}
    # Source prompts/reference answers from Dolly stay in the local generation files.
    # Aggregate scores and self-authored diagnostic probes are safe to publish together.
    result = {"experiment": "supervised-fine-tuning", "config": config, "initialization": initialization,
              "data_identity": identity, "sft_manifest": manifest, "general_manifest": general_manifest,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "hardware": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "torch_version": str(torch.__version__), "training_precision": config["precision"], "evaluation_precision": "fp32",
              "updates": total_steps, "response_tokens": seen, "seen_examples": seen_examples,
              "training_history": history, "evaluation_history": evaluations,
              "general_evaluation_offsets": offsets.tolist(),
              "initial": {k: v for k, v in initial.items() if k != "dev_generation_details"},
              "final": {k: v for k, v in final.items() if k != "dev_generation_details"},
              "checkpoint": str(checkpoint), "checkpoint_reload_max_logit_error": (actual_logits - expected_logits).abs().max().item(),
              "data_unchanged": True, "resume_from": str(resume) if resume else None, "session": session,
              **{key: session[key] if resume is None else None for key in ("training_seconds", "wall_seconds", "peak_cuda_allocated_bytes")}}
    write_json("metrics.json", result)
    event({"event": "complete", "metrics": str(run_dir / "metrics.json"),
           "dev_response": final["dev_response"], "general_validation_full": final["general_validation_full"],
           "dev_exact_matches": final["dev_generation"]["exact_matches"], "probe_exact_matches": final["probes"]["exact_matches"]})
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-from", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--general-data-dir", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=ROOT / "experiments/sft_probes.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    elif args.resume is not None:
        config = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)["config"]
    else:
        config = json.loads((ROOT / "experiments/sft_dolly_config.json").read_text(encoding="utf-8"))
    run(config, args.data_dir, general_data_dir=args.general_data_dir, probes_path=args.probes,
        device=args.device, output_root=ROOT / "runs", init_from=args.init_from, resume=args.resume)


if __name__ == "__main__":
    main()
