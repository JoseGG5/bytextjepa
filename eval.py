import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.utils import load_cfg, mean_pooling, pad_tokens, validate_cfg
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import BaselineTokenizer
from src.aug.augmentations import Augmentations
from src.model.model import ByteModernBertEncoder
from src.model.cnn_byte_model import CnnByteModernBertEncoder
from src.mlm.model import ByteModernBertMlm

"""Simple script to evaluate trained or untrained encoders on view-level retrieval,
similarity, and embedding geometry."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JEPA embeddings with simple retrieval metrics.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=str, help="Path to a saved training checkpoint.")
    source_group.add_argument(
        "--baseline-untrained",
        action="store_true",
        help="Evaluate a randomly initialized encoder with no checkpoint loaded.",
    )
    parser.add_argument("--cfg", type=str, default="cfg.yml", help="Path to the config file.")
    parser.add_argument(
        "--objective",
        type=str,
        choices=["lejepa", "mlm"],
        default=None,
        help="Checkpoint objective. Defaults to cfg.yml when omitted.",
    )
    parser.add_argument("--num-samples", type=int, default=500, help="Number of records to evaluate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used during evaluation.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for deterministic subset selection.")
    parser.add_argument(
        "--non-overlap-positive-pairs",
        action="store_true",
        help="Use two non-overlapping spans from the same record instead of the dataset's default random views.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_cfg(
    cfg_path: str,
    checkpoint_path: str | None,
    baseline_untrained: bool,
    objective: str | None,
) -> tuple[dict, torch.device, nn.Module, str, str]:
    cfg = load_cfg(cfg_path)
    cfg["dataset"]["dev"] = False
    validate_cfg(cfg)
    resolved_objective = objective or cfg["exp"].get("objective", "lejepa")

    requested_device = cfg["exp"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    if resolved_objective == "lejepa":
        if cfg["model"].get("input_mode", "byte") == "cnn_byte":
            model = CnnByteModernBertEncoder(cfg=cfg).to(device)
        else:
            model = ByteModernBertEncoder(cfg=cfg).to(device)
    elif resolved_objective == "mlm":
        model = ByteModernBertMlm(cfg=cfg).to(device)
    else:
        raise ValueError(f"Objective {resolved_objective} not implemented in eval.py")

    if baseline_untrained:
        checkpoint_label = "baseline_untrained"
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_label = checkpoint_path
    model.eval()
    return cfg, device, model, checkpoint_label, resolved_objective


def get_encoder_backbone(model: nn.Module, objective: str) -> nn.Module:
    if objective == "lejepa":
        return model.encoder
    if objective == "mlm":
        return model.encoder
    raise ValueError(f"Objective {objective} not implemented in eval.py")


def build_loader(cfg: dict, num_samples: int, batch_size: int, seed: int) -> DataLoader:
    tokenizer = BaselineTokenizer(cfg=cfg)
    augmenter = Augmentations(cfg=cfg)
    dataset = TextDataset(cfg=cfg, tokenizer=tokenizer, augmenter=augmenter)

    generator = torch.Generator().manual_seed(seed)
    subset_size = min(num_samples, len(dataset))
    subset_indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    subset = Subset(dataset, subset_indices)

    return DataLoader(dataset=subset, batch_size=batch_size, shuffle=False)


def build_hard_eval_loader(cfg: dict, num_samples: int, batch_size: int, seed: int) -> DataLoader:
    tokenizer = BaselineTokenizer(cfg=cfg)
    augmenter = Augmentations(cfg=cfg)
    dataset = TextDataset(cfg=cfg, tokenizer=tokenizer, augmenter=augmenter)

    generator = torch.Generator().manual_seed(seed)
    subset_size = min(num_samples, len(dataset))
    subset_indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()

    records = [dataset.data[idx]["text"] for idx in subset_indices]

    return DataLoader(
        dataset=records,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )


def sample_non_overlapping_pair(
    tokens: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = cfg["aug"]["global_max_length"]
    pad_token_id = cfg["model"]["pad_token_id"]
    min_gap = max(1, max_length // 4)
    total_length = tokens.size(0)

    if total_length <= 2:
        first = tokens[:1]
        second = tokens[1:2] if total_length > 1 else tokens[:1]
    else:
        max_span_length = min(max_length, max(1, total_length // 2))
        min_span_length = min(max_span_length, max(1, total_length // 4))

        for _ in range(32):
            span_length_a = random.randint(min_span_length, max_span_length)
            span_length_b = random.randint(min_span_length, max_span_length)
            start_a = random.randint(0, total_length - span_length_a)
            start_b = random.randint(0, total_length - span_length_b)
            end_a = start_a + span_length_a
            end_b = start_b + span_length_b

            non_overlapping = end_a <= start_b or end_b <= start_a
            separated = abs(start_a - start_b) >= min_gap or abs(end_a - end_b) >= min_gap
            if non_overlapping and separated:
                first = tokens[start_a:end_a]
                second = tokens[start_b:end_b]
                break
        else:
            midpoint = total_length // 2
            first = tokens[:midpoint]
            second = tokens[midpoint:]
            if first.numel() == 0:
                first = tokens[:1]
            if second.numel() == 0:
                second = tokens[-1:]

    first, first_mask = pad_tokens(
        x=first,
        output_attn_mask=True,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    second, second_mask = pad_tokens(
        x=second,
        output_attn_mask=True,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    return first, second, first_mask, second_mask


@torch.no_grad()
def extract_global_view_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings_view_a = []
    embeddings_view_b = []
    encoder = get_encoder_backbone(model, objective)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        global_ids = batch["global_crops"]
        global_masks = batch["global_masks"]

        input_a = global_ids[:, 0, :]
        input_b = global_ids[:, 1, :]
        mask_a = global_masks[:, 0, :]
        mask_b = global_masks[:, 1, :]

        hidden_a = encoder(input_ids=input_a, attention_mask=mask_a).last_hidden_state
        hidden_b = encoder(input_ids=input_b, attention_mask=mask_b).last_hidden_state

        embeddings_view_a.append(mean_pooling(hidden_a, mask_a).cpu())
        embeddings_view_b.append(mean_pooling(hidden_b, mask_b).cpu())

    view_a = torch.cat(embeddings_view_a, dim=0)
    view_b = torch.cat(embeddings_view_b, dim=0)
    return view_a, view_b


@torch.no_grad()
def extract_non_overlapping_embeddings(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: BaselineTokenizer,
    cfg: dict,
    device: torch.device,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings_view_a = []
    embeddings_view_b = []
    encoder = get_encoder_backbone(model, objective)

    for texts in loader:
        view_a_ids = []
        view_b_ids = []
        view_a_masks = []
        view_b_masks = []

        for text in texts:
            tokenized = tokenizer.tokenize(text)
            first, second, first_mask, second_mask = sample_non_overlapping_pair(
                tokenized["input_ids"],
                cfg,
            )
            view_a_ids.append(first)
            view_b_ids.append(second)
            view_a_masks.append(first_mask)
            view_b_masks.append(second_mask)

        input_a = torch.stack(view_a_ids).to(device)
        input_b = torch.stack(view_b_ids).to(device)
        mask_a = torch.stack(view_a_masks).to(device)
        mask_b = torch.stack(view_b_masks).to(device)

        hidden_a = encoder(input_ids=input_a, attention_mask=mask_a).last_hidden_state
        hidden_b = encoder(input_ids=input_b, attention_mask=mask_b).last_hidden_state

        embeddings_view_a.append(mean_pooling(hidden_a, mask_a).cpu())
        embeddings_view_b.append(mean_pooling(hidden_b, mask_b).cpu())

    view_a = torch.cat(embeddings_view_a, dim=0)
    view_b = torch.cat(embeddings_view_b, dim=0)
    return view_a, view_b


def retrieval_at_k(view_a: torch.Tensor, view_b: torch.Tensor, k: int) -> float:
    sim = F.normalize(view_a, dim=1) @ F.normalize(view_b, dim=1).T
    topk = sim.topk(k=k, dim=1).indices
    targets = torch.arange(sim.size(0)).unsqueeze(1)
    correct = (topk == targets).any(dim=1).float()
    return correct.mean().item()


def similarity_summary(view_a: torch.Tensor, view_b: torch.Tensor) -> dict[str, float]:
    view_a = F.normalize(view_a, dim=1)
    view_b = F.normalize(view_b, dim=1)
    positive = (view_a * view_b).sum(dim=1)

    neg_perm = torch.roll(torch.arange(view_b.size(0)), shifts=1)
    negative = (view_a * view_b[neg_perm]).sum(dim=1)

    return {
        "positive_mean_cosine": positive.mean().item(),
        "positive_std_cosine": positive.std(unbiased=False).item(),
        "negative_mean_cosine": negative.mean().item(),
        "negative_std_cosine": negative.std(unbiased=False).item(),
    }


def geometry_summary(embeddings: torch.Tensor) -> dict[str, float]:
    norms = embeddings.norm(dim=1)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    feature_std = centered.std(dim=0, unbiased=False)
    covariance = centered.T @ centered / max(1, centered.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if total.item() == 0:
        effective_rank = 0.0
    else:
        probs = eigenvalues / total
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        effective_rank = torch.exp(entropy).item()

    return {
        "embedding_norm_mean": norms.mean().item(),
        "embedding_norm_std": norms.std(unbiased=False).item(),
        "feature_std_mean": feature_std.mean().item(),
        "effective_rank": effective_rank,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg, device, model, checkpoint_label, objective = load_model_and_cfg(
        args.cfg,
        args.checkpoint,
        args.baseline_untrained,
        args.objective,
    )
    if args.non_overlap_positive_pairs:
        tokenizer = BaselineTokenizer(cfg=cfg)
        loader = build_hard_eval_loader(cfg, args.num_samples, args.batch_size, args.seed)
        view_a, view_b = extract_non_overlapping_embeddings(
            model,
            loader,
            tokenizer,
            cfg,
            device,
            objective,
        )
    else:
        loader = build_loader(cfg, args.num_samples, args.batch_size, args.seed)
        view_a, view_b = extract_global_view_embeddings(model, loader, device, objective)

    metrics = {
        "checkpoint": checkpoint_label,
        "objective": objective,
        "eval_mode": "non_overlap_positive_pairs" if args.non_overlap_positive_pairs else "default_views",
        "num_samples": int(view_a.size(0)),
        "retrieval_top1": retrieval_at_k(view_a, view_b, k=1),
        "retrieval_top5": retrieval_at_k(view_a, view_b, k=min(5, view_b.size(0))),
    }
    metrics.update(similarity_summary(view_a, view_b))
    metrics.update(geometry_summary(torch.cat([view_a, view_b], dim=0)))

    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
