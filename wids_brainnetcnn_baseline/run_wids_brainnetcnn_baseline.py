from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


SEED = 42
N_SPLITS = 5
BATCH_SIZE = 16
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 5
VAL_SIZE = 0.15
THRESHOLD = 0.5


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "EDA" / "TRAIN_NEW"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_fc_csv() -> Path:
    candidates = sorted(DATA_DIR.glob("TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No FC CSV found under {DATA_DIR}")
    return max(candidates, key=lambda p: p.stat().st_size)


def fisher_z_transform(x: np.ndarray, clip_value: float = 0.999) -> np.ndarray:
    x = np.clip(x, -clip_value, clip_value)
    return np.arctanh(x)


def edge_vectors_to_matrices(edge_array: np.ndarray, n_roi: int) -> np.ndarray:
    matrices = np.zeros((edge_array.shape[0], n_roi, n_roi), dtype=np.float32)
    triu_i, triu_j = np.triu_indices(n_roi, k=1)
    matrices[:, triu_i, triu_j] = edge_array
    matrices[:, triu_j, triu_i] = edge_array
    return matrices


def infer_n_roi(n_edges: int) -> int:
    n_roi = int((1 + math.sqrt(1 + 8 * n_edges)) / 2)
    if n_roi * (n_roi - 1) // 2 != n_edges:
        raise ValueError(f"Cannot infer ROI count from {n_edges} edges")
    return n_roi


def load_wids_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    fc_csv = pick_fc_csv()
    label_xlsx = list(DATA_DIR.glob("TRAINING_SOLUTIONS*.xlsx"))[0]
    meta_xlsx = list(DATA_DIR.glob("TRAIN_QUANTITATIVE_METADATA*.xlsx"))[0]

    df_fc = pd.read_csv(fc_csv)
    df_label = pd.read_excel(label_xlsx)
    df_meta = pd.read_excel(meta_xlsx)
    df_all = df_fc.merge(df_label, on="participant_id").merge(df_meta, on="participant_id")

    fc_columns = [col for col in df_fc.columns if col != "participant_id"]
    n_roi = infer_n_roi(len(fc_columns))

    fc_raw = df_all[fc_columns].to_numpy(dtype=np.float32)
    fc_z = fisher_z_transform(fc_raw)
    fc_matrices = edge_vectors_to_matrices(fc_z, n_roi)
    x_all = fc_matrices[:, None, :, :].astype(np.float32)
    y_adhd = df_all["ADHD_Outcome"].to_numpy(dtype=np.int64)
    y_sex = df_all["Sex_F"].to_numpy(dtype=np.int64)
    return df_all, x_all, y_adhd, y_sex


class FCMatrixDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class EdgeToEdgeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_roi: int) -> None:
        super().__init__()
        self.n_roi = n_roi
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, n_roi))
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(n_roi, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_features = self.row_conv(x).repeat(1, 1, 1, self.n_roi)
        col_features = self.col_conv(x).repeat(1, 1, self.n_roi, 1)
        return row_features + col_features


class BrainNetCNN(nn.Module):
    def __init__(self, n_roi: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.e2e1 = EdgeToEdgeBlock(1, 32, n_roi)
        self.e2e2 = EdgeToEdgeBlock(32, 64, n_roi)
        self.e2n = nn.Conv2d(64, 128, kernel_size=(1, n_roi))
        self.n2g = nn.Conv2d(128, 256, kernel_size=(n_roi, 1))
        self.act = nn.LeakyReLU(0.33)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.e2e1(x)))
        x = self.drop(self.act(self.e2e2(x)))
        x = self.drop(self.act(self.e2n(x)))
        x = self.drop(self.act(self.n2g(x)))
        return self.head(x).squeeze(-1)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    probs = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets).astype(int)
    return float(np.mean(losses)), y_true, y_prob


def run_single_fold(x_train_full: np.ndarray, y_train_full: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, n_roi: int) -> tuple[dict[str, float], np.ndarray]:
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=VAL_SIZE,
        stratify=y_train_full,
        random_state=SEED,
    )

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    train_loader = DataLoader(FCMatrixDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FCMatrixDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FCMatrixDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = BrainNetCNN(n_roi, dropout=DROPOUT).to(DEVICE)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], dtype=torch.float32).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, y_val_true, y_val_prob = evaluate(model, val_loader, loss_fn)
        val_metrics = compute_binary_metrics(y_val_true, y_val_prob, threshold=THRESHOLD)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, y_test_true, y_test_prob = evaluate(model, test_loader, loss_fn)
    test_metrics = compute_binary_metrics(y_test_true, y_test_prob, threshold=THRESHOLD)
    test_metrics["test_loss"] = test_loss
    test_metrics["best_val_roc_auc"] = float(best_auc)
    return test_metrics, y_test_prob


def run_cross_validation(x: np.ndarray, y: np.ndarray, task_name: str, participant_ids: np.ndarray, n_roi: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_rows = []
    oof_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        print(f"\n===== {task_name} | fold {fold}/{N_SPLITS} =====")
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        fold_metrics, test_prob = run_single_fold(x_train, y_train, x_test, y_test, n_roi=n_roi)
        fold_rows.append({"task": task_name, "fold": fold, **fold_metrics})

        for local_idx, global_idx in enumerate(test_idx):
            oof_rows.append(
                {
                    "task": task_name,
                    "fold": fold,
                    "participant_id": str(participant_ids[global_idx]),
                    "y_true": int(y[global_idx]),
                    "y_prob": float(test_prob[local_idx]),
                }
            )

        print(
            f"[{task_name}] fold={fold} acc={fold_metrics['accuracy']:.4f} "
            f"bal_acc={fold_metrics['balanced_accuracy']:.4f} auc={fold_metrics['roc_auc']:.4f}"
        )

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def summarize_cv_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    summary = fold_df[["accuracy", "balanced_accuracy", "f1", "roc_auc", "test_loss"]].agg(["mean", "std"]).T
    summary = summary.reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary


def format_summary(summary_df: pd.DataFrame, task_name: str, model_name: str = "BrainNetCNN") -> pd.DataFrame:
    out = summary_df.copy()
    out["task"] = task_name
    out["model"] = model_name
    out["display"] = out.apply(lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1)
    return out[["task", "model", "metric", "mean", "std", "display"]]


def main() -> None:
    seed_everything(SEED)
    df_all, x_all, y_adhd, y_sex = load_wids_data()
    participant_ids = df_all["participant_id"].to_numpy()
    n_roi = x_all.shape[-1]

    adhd_fold_df, adhd_oof_df = run_cross_validation(x_all, y_adhd, "ADHD", participant_ids, n_roi)
    sex_fold_df, sex_oof_df = run_cross_validation(x_all, y_sex, "Sex", participant_ids, n_roi)

    adhd_summary_df = summarize_cv_results(adhd_fold_df)
    sex_summary_df = summarize_cv_results(sex_fold_df)
    summary_table = pd.concat(
        [
            format_summary(adhd_summary_df, task_name="ADHD"),
            format_summary(sex_summary_df, task_name="Sex"),
        ],
        ignore_index=True,
    )

    adhd_fold_df.to_csv(OUTPUT_DIR / "brainnetcnn_adhd_fold_metrics.csv", index=False)
    adhd_oof_df.to_csv(OUTPUT_DIR / "brainnetcnn_adhd_oof_predictions.csv", index=False)
    sex_fold_df.to_csv(OUTPUT_DIR / "brainnetcnn_sex_fold_metrics.csv", index=False)
    sex_oof_df.to_csv(OUTPUT_DIR / "brainnetcnn_sex_oof_predictions.csv", index=False)
    summary_table.to_csv(OUTPUT_DIR / "brainnetcnn_summary_table.csv", index=False)

    print("\nSaved outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
