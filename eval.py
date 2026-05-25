import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.utils import load_cfg, validate_cfg
from src.data.dataset import TextDataset
from src.data.byte_tokenizer import BaselineTokenizer
from src.aug.augmentations import Augmentations
from src.model.model import ByteModernBertEncoder

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
    parser.add_argument("--num-samples", type=int, default=500, help="Number of records to evaluate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used during evaluation.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for deterministic subset selection.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_cfg(
    cfg_path: str,
    checkpoint_path: str | None,
    baseline_untrained: bool,
) -> tuple[dict, torch.device, ByteModernBertEncoder, str]:
    cfg = load_cfg(cfg_path)
    cfg["dataset"]["dev"] = False
    validate_cfg(cfg)

    requested_device = cfg["exp"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    model = ByteModernBertEncoder(cfg=cfg).to(device)
    if baseline_untrained:
        checkpoint_label = "baseline_untrained"
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_label = checkpoint_path
    model.eval()
    return cfg, device, model, checkpoint_label


def build_loader(cfg: dict, num_samples: int, batch_size: int, seed: int) -> DataLoader:
    tokenizer = BaselineTokenizer(cfg=cfg)
    augmenter = Augmentations(cfg=cfg)
    dataset = TextDataset(cfg=cfg, tokenizer=tokenizer, augmenter=augmenter)

    generator = torch.Generator().manual_seed(seed)
    subset_size = min(num_samples, len(dataset))
    subset_indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    subset = Subset(dataset, subset_indices)

    return DataLoader(dataset=subset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def extract_global_view_embeddings(
    model: ByteModernBertEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings_view_a = []
    embeddings_view_b = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        z = model(
            global_input_ids=batch["global_crops"],
            global_attn_mask=batch["global_masks"],
            local_input_ids=batch["local_crops"],
            local_attn_mask=batch["local_masks"],
        )
        embeddings_view_a.append(z[:, 0, :].cpu())
        embeddings_view_b.append(z[:, 1, :].cpu())

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

    cfg, device, model, checkpoint_label = load_model_and_cfg(
        args.cfg,
        args.checkpoint,
        args.baseline_untrained,
    )
    loader = build_loader(cfg, args.num_samples, args.batch_size, args.seed)
    view_a, view_b = extract_global_view_embeddings(model, loader, device)

    metrics = {
        "checkpoint": checkpoint_label,
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
