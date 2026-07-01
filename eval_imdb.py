import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.byte_tokenizer import BaselineTokenizer
from src.model.cnn_byte_model import CnnByteModernBertEncoder
from src.model.model import ByteModernBertEncoder
from src.mlm.model import ByteModernBertMlm
from src.utils import load_cfg, pad_tokens, pool_encoder_output, validate_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen byte encoders on IMDB sentiment with a linear probe."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=str, help="Path to a saved training checkpoint.")
    source_group.add_argument(
        "--baseline-untrained",
        action="store_true",
        help="Evaluate a randomly initialized encoder with no checkpoint loaded.",
    )
    parser.add_argument("--objective", type=str, choices=["lejepa", "mlm"], required=True)
    parser.add_argument("--cfg", type=str, default="cfg.yml", help="Path to the config file.")
    parser.add_argument("--dataset-name", type=str, default="stanfordnlp/imdb", help="Hugging Face dataset id.")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset configuration.",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["linear", "mlp"],
        default="linear",
        help="Classifier trained on top of frozen embeddings.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of the train split used for model selection.",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
        help="Candidate C values for linear probe selection.",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension used when --probe-type mlp.",
    )
    parser.add_argument(
        "--mlp-alpha-values",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3],
        help="Candidate alpha values for MLP probe selection.",
    )
    parser.add_argument("--chunk-batch-size", type=int, default=16, help="Number of chunks encoded at once.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional maximum chunks per document.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train examples for faster experiments.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on test examples for faster experiments.",
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


def load_imdb_splits(
    dataset_name: str,
    dataset_config: str | None,
    max_train_samples: int | None,
    max_test_samples: int | None,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if dataset_config is None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, dataset_config)

    train_split = dataset["train"].shuffle(seed=seed)
    test_split = dataset["test"].shuffle(seed=seed)

    if max_train_samples is not None:
        train_split = train_split.select(range(min(max_train_samples, len(train_split))))
    if max_test_samples is not None:
        test_split = test_split.select(range(min(max_test_samples, len(test_split))))

    def normalize(split) -> list[dict[str, object]]:
        records = []
        for example in split:
            text = example["text"]
            label = int(example["label"])
            if text is None or not str(text).strip():
                continue
            records.append({"text": text, "label": label})
        return records

    return normalize(train_split), normalize(test_split)


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
) -> tuple[torch.Tensor, np.ndarray]:
    encoder = model.encoder
    chunk_length = resolve_chunk_length(cfg, objective)
    pad_token_id = int(cfg["model"]["pad_token_id"])

    document_embeddings = []
    labels = []

    for record in tqdm(records, desc="embedding documents"):
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
            fallback = torch.tensor([pad_token_id], dtype=torch.long)
            padded_chunk, attn_mask = pad_tokens(
                x=fallback,
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
            pooled = pool_encoder_output(encoder, batch_ids, batch_masks).cpu()
            chunk_embeddings.append(pooled)

        doc_embedding = torch.cat(chunk_embeddings, dim=0).mean(dim=0)
        document_embeddings.append(doc_embedding)
        labels.append(int(record["label"]))

    return torch.stack(document_embeddings), np.asarray(labels, dtype=np.int64)


class StandardScalerTorch:
    """Feature standardization fit on train data, applied to any split."""

    def fit(self, x: torch.Tensor) -> "StandardScalerTorch":
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) / self.std_


class LinearHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MlpHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchProbe:
    """Binary classifier head trained on frozen embeddings with CUDA acceleration.

    Carves off a small validation split internally for early stopping, mirroring
    what MLPClassifier(early_stopping=True) did, so the same routine works for
    both the linear and MLP heads.
    """

    def __init__(
        self,
        model_fn,
        weight_decay: float,
        device: torch.device,
        lr: float = 1e-3,
        max_epochs: int = 200,
        batch_size: int = 256,
        patience: int = 15,
        val_fraction: float = 0.1,
        seed: int = 13,
        desc: str = "probe",
    ):
        self.model_fn = model_fn
        self.weight_decay = weight_decay
        self.device = device
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.desc = desc
        self.scaler = StandardScalerTorch()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchProbe":
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=self.val_fraction, random_state=self.seed, stratify=y,
        )
        x_train = torch.as_tensor(x_train, dtype=torch.float32)
        x_val = torch.as_tensor(x_val, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train).to(self.device)
        x_val = self.scaler.transform(x_val).to(self.device)
        y_train = y_train.to(self.device)
        y_val_t = y_val_t.to(self.device)

        generator = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        self.model = self.model_fn(x_train.size(1)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        best_state = None
        best_val_acc = -1.0
        epochs_without_improvement = 0

        progress = tqdm(range(self.max_epochs), desc=self.desc, leave=False)
        for _ in progress:
            self.model.train()
            running_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            running_loss /= len(loader.dataset)

            self.model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(self.model(x_val))
                val_acc = ((val_probs > 0.5).float() == y_val_t).float().mean().item()
            progress.set_postfix(loss=f"{running_loss:.4f}", val_acc=f"{val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        with torch.no_grad():
            self.model.eval()
            val_scores = torch.sigmoid(self.model(x_val)).cpu().numpy()
        val_preds = (val_scores > 0.5).astype(np.int64)
        self.val_metrics_ = compute_metrics(y_val, val_preds, val_scores)
        return self

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor = self.scaler.transform(torch.as_tensor(x, dtype=torch.float32)).to(self.device)
        scores = torch.sigmoid(self.model(tensor)).cpu().numpy()
        return np.stack([1.0 - scores, scores], axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] > 0.5).astype(np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "average_precision": average_precision_score(y_true, y_score),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def select_probe(
    train_embeddings: torch.Tensor,
    train_labels: np.ndarray,
    probe_type: str,
    c_values: list[float],
    mlp_hidden_dim: int,
    mlp_alpha_values: list[float],
    val_size: float,
    seed: int,
    device: torch.device,
):
    if np.unique(train_labels).size < 2:
        raise ValueError("Train labels must contain at least two classes to fit a probe.")

    train_embeddings_np = train_embeddings.numpy()

    if probe_type == "linear":
        candidates = [{"weight_decay": 1.0 / value} for value in c_values]
        model_fn = lambda dim: LinearHead(dim)  # noqa: E731
    else:
        candidates = [{"weight_decay": value} for value in mlp_alpha_values]
        model_fn = lambda dim: MlpHead(dim, mlp_hidden_dim)  # noqa: E731

    best_probe = None
    best_params = {}
    best_metrics = None
    best_score = -float("inf")

    for params in tqdm(candidates, desc=f"selecting {probe_type} probe"):
        probe = TorchProbe(
            model_fn=model_fn,
            weight_decay=params["weight_decay"],
            device=device,
            val_fraction=val_size,
            seed=seed,
            desc=f"{probe_type} wd={params['weight_decay']:.4g}",
        )
        probe.fit(train_embeddings_np, train_labels)
        score = probe.val_metrics_["accuracy"]

        if score > best_score:
            best_score = score
            best_probe = probe
            best_params = params
            best_metrics = probe.val_metrics_

    return best_probe, best_params, best_metrics


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
        args.objective,
    )
    tokenizer = BaselineTokenizer(cfg=cfg)

    train_records, test_records = load_imdb_splits(
        args.dataset_name,
        args.dataset_config,
        args.max_train_samples,
        args.max_test_samples,
        args.seed,
    )
    train_embeddings, train_labels = embed_documents(
        train_records,
        model,
        tokenizer,
        cfg,
        device,
        args.objective,
        args.chunk_batch_size,
        args.max_chunks,
    )
    test_embeddings, test_labels = embed_documents(
        test_records,
        model,
        tokenizer,
        cfg,
        device,
        args.objective,
        args.chunk_batch_size,
        args.max_chunks,
    )

    probe, selected_params, validation_metrics = select_probe(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        probe_type=args.probe_type,
        c_values=args.c_values,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_alpha_values=args.mlp_alpha_values,
        val_size=args.val_size,
        seed=args.seed,
        device=device,
    )

    test_pred = probe.predict(test_embeddings.numpy())
    test_score = probe.predict_proba(test_embeddings.numpy())[:, 1]
    test_metrics = compute_metrics(test_labels, test_pred, test_score)

    cosine_pos = F.cosine_similarity(
        train_embeddings[: min(len(train_embeddings), len(test_embeddings))],
        test_embeddings[: min(len(train_embeddings), len(test_embeddings))],
        dim=1,
    ).mean().item()

    metrics = {
        "checkpoint": checkpoint_label,
        "objective": args.objective,
        "dataset_name": args.dataset_name,
        "probe_type": args.probe_type,
        "num_train_docs": len(train_records),
        "num_test_docs": len(test_records),
        "selected_params": selected_params,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_f1": validation_metrics["f1"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_average_precision": validation_metrics["average_precision"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_average_precision": test_metrics["average_precision"],
        "test_roc_auc": test_metrics["roc_auc"],
        "mean_train_test_prefix_cosine": cosine_pos,
    }
    metrics.update(geometry_summary(torch.cat([train_embeddings, test_embeddings], dim=0)))

    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
