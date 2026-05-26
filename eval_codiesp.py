import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from umap import UMAP

from src.utils import load_cfg, mean_pooling, pad_tokens, validate_cfg
from src.data.byte_tokenizer import BaselineTokenizer
from src.model.model import ByteModernBertEncoder
from src.model.cnn_byte_model import CnnByteModernBertEncoder
from src.mlm.model import ByteModernBertMlm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen JEPA/MLM encoders on CodiEsp diagnosis coding.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=str, help="Path to a saved training checkpoint.")
    source_group.add_argument(
        "--baseline-untrained",
        action="store_true",
        help="Evaluate a randomly initialized encoder with no checkpoint loaded.",
    )
    parser.add_argument("--objective", type=str, choices=["lejepa", "mlm"], required=True)
    parser.add_argument("--cfg", type=str, default="cfg.yml", help="Path to the config file.")
    parser.add_argument("--dataset-name", type=str, default="bigbio/codiesp", help="Hugging Face dataset id.")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="codiesp_D_source",
        help="Hugging Face dataset configuration to load.",
    )
    parser.add_argument("--top-k-codes", type=int, default=50, help="Number of most frequent diagnosis codes to keep.")
    parser.add_argument("--chunk-batch-size", type=int, default=16, help="Number of chunks encoded at once.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional maximum chunks per document.")
    parser.add_argument(
        "--retrieval-jaccard-threshold",
        type=float,
        default=0.5,
        help="Minimum label Jaccard overlap required to count a retrieved document as relevant.",
    )
    parser.add_argument("--umap-output", type=str, default=None, help="Optional path to save a small UMAP HTML scatter.")
    parser.add_argument("--seed", type=int, default=13, help="Seed for reproducibility.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_cfg(
    cfg_path: str,
    checkpoint_path: str | None,
    baseline_untrained: bool,
    objective: str,
) -> tuple[dict, torch.device, torch.nn.Module, str]:
    cfg = load_cfg(cfg_path)
    cfg["dataset"]["dev"] = False
    validate_cfg(cfg)

    requested_device = cfg["exp"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    if objective == "lejepa":
        if cfg["model"].get("input_mode", "byte") == "cnn_byte":
            model = CnnByteModernBertEncoder(cfg=cfg).to(device)
        else:
            model = ByteModernBertEncoder(cfg=cfg).to(device)
    else:
        model = ByteModernBertMlm(cfg=cfg).to(device)

    if baseline_untrained:
        checkpoint_label = "baseline_untrained"
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_label = checkpoint_path

    model.eval()
    return cfg, device, model, checkpoint_label


def resolve_chunk_length(cfg: dict, objective: str) -> int:
    if objective == "mlm":
        return int(cfg["mlm"]["max_length"])
    return int(cfg["aug"]["global_max_length"])


def build_codiesp_data_files(dataset_name: str, dataset_config: str) -> dict[str, str]:
    base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/refs%2Fconvert%2Fparquet/{dataset_config}"
    return {
        "train": f"{base_url}/train/0000.parquet",
        "validation": f"{base_url}/validation/0000.parquet",
        "test": f"{base_url}/test/0000.parquet",
    }


def load_codiesp_splits(dataset_name: str, dataset_config: str) -> dict[str, list[dict[str, object]]]:
    dataset = load_dataset("parquet", data_files=build_codiesp_data_files(dataset_name, dataset_config))
    splits = {}
    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        records = []
        for example in split:
            text = example["text"]
            labels = example["labels"]
            if text is None or not str(text).strip():
                continue
            records.append(
                {
                    "document_id": str(example["document_id"]),
                    "text": text,
                    "labels": [str(label) for label in labels],
                }
            )
        splits[split_name] = records
    return splits


def select_top_codes(train_records: list[dict[str, object]], top_k_codes: int) -> list[str]:
    counts = Counter()
    for record in train_records:
        counts.update(record["labels"])
    return [code for code, _ in counts.most_common(top_k_codes)]


def filter_labels(records: list[dict[str, object]], allowed_codes: set[str]) -> list[dict[str, object]]:
    filtered = []
    for record in records:
        labels = [label for label in record["labels"] if label in allowed_codes]
        if not labels:
            continue
        filtered.append(
            {
                "document_id": record["document_id"],
                "text": record["text"],
                "labels": labels,
            }
        )
    return filtered


@torch.no_grad()
def embed_documents(
    records: list[dict[str, object]],
    model: torch.nn.Module,
    tokenizer: BaselineTokenizer,
    cfg: dict,
    device: torch.device,
    objective: str,
    chunk_batch_size: int,
    max_chunks: int | None,
) -> tuple[torch.Tensor, list[list[str]], list[str]]:
    encoder = model.encoder
    chunk_length = resolve_chunk_length(cfg, objective)
    pad_token_id = int(cfg["model"]["pad_token_id"])

    document_embeddings = []
    labels = []
    document_ids = []

    for record in records:
        tokenized = tokenizer.tokenize(record["text"])
        input_ids = tokenized["input_ids"]
        chunks = []
        masks = []

        for start in range(0, input_ids.size(0), chunk_length):
            chunk = input_ids[start:start + chunk_length]
            if chunk.numel() == 0:
                continue
            padded_chunk, attn_mask = pad_tokens(
                x=chunk,
                output_attn_mask=True,
                max_length=chunk_length,
                pad_token_id=pad_token_id,
            )
            chunks.append(padded_chunk)
            masks.append(attn_mask)
            if max_chunks is not None and len(chunks) >= max_chunks:
                break

        if not chunks:
            padded_chunk, attn_mask = pad_tokens(
                x=torch.tensor([pad_token_id], dtype=torch.long),
                output_attn_mask=True,
                max_length=chunk_length,
                pad_token_id=pad_token_id,
            )
            chunks = [padded_chunk]
            masks = [attn_mask]

        chunk_embeddings = []
        for start in range(0, len(chunks), chunk_batch_size):
            batch_ids = torch.stack(chunks[start:start + chunk_batch_size]).to(device)
            batch_masks = torch.stack(masks[start:start + chunk_batch_size]).to(device)
            hidden = encoder(input_ids=batch_ids, attention_mask=batch_masks).last_hidden_state
            chunk_embeddings.append(mean_pooling(hidden, batch_masks).cpu())

        doc_embedding = torch.cat(chunk_embeddings, dim=0).mean(dim=0)
        document_embeddings.append(doc_embedding)
        labels.append(record["labels"])
        document_ids.append(record["document_id"])

    return torch.stack(document_embeddings), labels, document_ids


def retrieval_recall_at_k(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: list[list[str]],
    gallery_labels: list[list[str]],
    k: int,
    jaccard_threshold: float,
) -> float:
    sim = F.normalize(query_embeddings, dim=1) @ F.normalize(gallery_embeddings, dim=1).T
    topk = sim.topk(k=min(k, gallery_embeddings.size(0)), dim=1).indices

    correct = 0
    for query_idx, neighbor_indices in enumerate(topk):
        query_codes = set(query_labels[query_idx])
        hit = False
        for neighbor_idx in neighbor_indices.tolist():
            neighbor_codes = set(gallery_labels[neighbor_idx])
            union = query_codes | neighbor_codes
            if not union:
                continue
            jaccard = len(query_codes & neighbor_codes) / len(union)
            if jaccard >= jaccard_threshold:
                hit = True
                break
        correct += int(hit)
    return correct / max(1, len(query_labels))


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


def save_umap_html(
    embeddings: torch.Tensor,
    labels: list[str],
    colors: list[str],
    output_path: str,
    title: str,
) -> None:
    reducer = UMAP(n_components=2, random_state=13)
    projection = reducer.fit_transform(embeddings.numpy())

    min_x, min_y = projection.min(axis=0)
    max_x, max_y = projection.max(axis=0)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    points = []
    for idx, ((x, y), label, color) in enumerate(zip(projection, labels, colors)):
        svg_x = 40 + 720 * ((x - min_x) / span_x)
        svg_y = 40 + 420 * ((y - min_y) / span_y)
        safe_label = (
            label.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        points.append(
            f'<circle cx="{svg_x:.2f}" cy="{svg_y:.2f}" r="4" fill="{color}" fill-opacity="0.72">'
            f"<title>{idx}: {safe_label}</title></circle>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: Georgia, serif; background:#f7f4ef; color:#1f1c18; margin:24px; }}
.card {{ background:#fffdf8; border:1px solid #ddd4c8; border-radius:16px; padding:20px; max-width:860px; }}
.legend span {{ display:inline-block; margin-right:16px; }}
.dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; margin-right:6px; }}
svg {{ background:#fff; border:1px solid #ddd4c8; border-radius:12px; }}
</style></head>
<body><div class="card"><h1>{title}</h1>
<p>UMAP projection of document embeddings. Hover points for split and labels.</p>
<div class="legend">
<span><span class="dot" style="background:#c25b37"></span>train</span>
<span><span class="dot" style="background:#3d6f9e"></span>validation</span>
<span><span class="dot" style="background:#5b8f41"></span>test</span>
</div>
<svg width="800" height="500" viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
{''.join(points)}
</svg></div></body></html>"""
    Path(output_path).write_text(html, encoding="utf-8")


def run_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: list[list[str]],
    eval_embeddings: torch.Tensor,
    eval_labels: list[list[str]],
    classes: list[str],
) -> dict[str, float]:
    mlb = MultiLabelBinarizer(classes=classes)
    y_train = mlb.fit_transform(train_labels)
    y_eval = mlb.transform(eval_labels)

    classifier = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    )
    classifier.fit(train_embeddings.numpy(), y_train)
    y_pred = classifier.predict(eval_embeddings.numpy())

    return {
        "micro_f1": f1_score(y_eval, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_eval, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg, device, model, checkpoint_label = load_model_and_cfg(
        args.cfg,
        args.checkpoint,
        args.baseline_untrained,
        args.objective,
    )
    tokenizer = BaselineTokenizer(cfg=cfg)
    splits = load_codiesp_splits(args.dataset_name, args.dataset_config)
    top_codes = select_top_codes(splits["train"], args.top_k_codes)
    allowed_codes = set(top_codes)

    train_records = filter_labels(splits["train"], allowed_codes)
    validation_records = filter_labels(splits["validation"], allowed_codes)
    test_records = filter_labels(splits["test"], allowed_codes)

    train_embeddings, train_labels, _ = embed_documents(
        train_records,
        model,
        tokenizer,
        cfg,
        device,
        args.objective,
        args.chunk_batch_size,
        args.max_chunks,
    )
    validation_embeddings, validation_labels, _ = embed_documents(
        validation_records,
        model,
        tokenizer,
        cfg,
        device,
        args.objective,
        args.chunk_batch_size,
        args.max_chunks,
    )
    test_embeddings, test_labels, _ = embed_documents(
        test_records,
        model,
        tokenizer,
        cfg,
        device,
        args.objective,
        args.chunk_batch_size,
        args.max_chunks,
    )

    metrics = {
        "checkpoint": checkpoint_label,
        "objective": args.objective,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "top_k_codes": args.top_k_codes,
        "retrieval_jaccard_threshold": args.retrieval_jaccard_threshold,
        "num_train_docs": len(train_records),
        "num_validation_docs": len(validation_records),
        "num_test_docs": len(test_records),
        "retrieval_recall_at_1": retrieval_recall_at_k(
            test_embeddings,
            train_embeddings,
            test_labels,
            train_labels,
            k=1,
            jaccard_threshold=args.retrieval_jaccard_threshold,
        ),
        "retrieval_recall_at_5": retrieval_recall_at_k(
            test_embeddings,
            train_embeddings,
            test_labels,
            train_labels,
            k=5,
            jaccard_threshold=args.retrieval_jaccard_threshold,
        ),
        "retrieval_recall_at_10": retrieval_recall_at_k(
            test_embeddings,
            train_embeddings,
            test_labels,
            train_labels,
            k=10,
            jaccard_threshold=args.retrieval_jaccard_threshold,
        ),
    }
    metrics.update(
        {
            f"validation_{k}": v
            for k, v in run_linear_probe(
                train_embeddings,
                train_labels,
                validation_embeddings,
                validation_labels,
                top_codes,
            ).items()
        }
    )
    metrics.update(
        {
            f"test_{k}": v
            for k, v in run_linear_probe(
                train_embeddings,
                train_labels,
                test_embeddings,
                test_labels,
                top_codes,
            ).items()
        }
    )
    metrics.update(geometry_summary(torch.cat([train_embeddings, validation_embeddings, test_embeddings], dim=0)))

    print(json.dumps(metrics, indent=2))

    if args.umap_output is not None:
        all_embeddings = torch.cat([train_embeddings, validation_embeddings, test_embeddings], dim=0).cpu()
        all_labels = (
            [f"train | {', '.join(labels)}" for labels in train_labels]
            + [f"validation | {', '.join(labels)}" for labels in validation_labels]
            + [f"test | {', '.join(labels)}" for labels in test_labels]
        )
        all_colors = (
            ["#c25b37"] * len(train_labels)
            + ["#3d6f9e"] * len(validation_labels)
            + ["#5b8f41"] * len(test_labels)
        )
        save_umap_html(
            embeddings=all_embeddings,
            labels=all_labels,
            colors=all_colors,
            output_path=args.umap_output,
            title=f"UMAP - CodiEsp - {args.objective}",
        )

    if args.output is not None:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
