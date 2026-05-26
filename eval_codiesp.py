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

from src.utils import load_cfg, mean_pooling, pad_tokens, validate_cfg
from src.data.byte_tokenizer import BaselineTokenizer
from src.model.model import ByteModernBertEncoder
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

    if args.output is not None:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
