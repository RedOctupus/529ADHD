import json
import math
import os
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


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "EDA" / "TRAIN_NEW"
RESULT_ROOT = ROOT / "modling" / "brainnetcnn_results" / "sex_tuning"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

CHANNEL_PRESETS = {
    "small": (16, 32, 64, 128),
    "medium": (24, 48, 96, 192),
    "base": (32, 64, 128, 256),
}

CONFIGS = {
    "optuna_tuned": {
        "optimizer": "AdamW",
        "learning_rate": 0.00018891200276189413,
        "weight_decay": 3.982994169892413e-06,
        "dropout": 0.2233618039413735,
        "batch_size": 16,
        "max_epochs": 25,
        "patience": 6,
        "val_size": 0.15,
        "n_splits": 5,
        "channel_preset": "base",
        "use_batchnorm": True,
        "use_scheduler": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "threshold_metric": "balanced_accuracy",
    },
    "manual_tuned": {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 5e-4,
        "dropout": 0.40,
        "batch_size": 16,
        "max_epochs": 35,
        "patience": 8,
        "val_size": 0.15,
        "n_splits": 5,
        "channel_preset": "medium",
        "use_batchnorm": True,
        "use_scheduler": True,
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "threshold_metric": "balanced_accuracy",
    },
}


def fisher_z_transform(x, clip_value=0.999):
    x = np.clip(x, -clip_value, clip_value)
    return np.arctanh(x)


def edge_vectors_to_matrices(edge_array, n_roi):
    matrices = np.zeros((edge_array.shape[0], n_roi, n_roi), dtype=np.float32)
    triu_i, triu_j = np.triu_indices(n_roi, k=1)
    matrices[:, triu_i, triu_j] = edge_array
    matrices[:, triu_j, triu_i] = edge_array
    return matrices


class FCMatrixDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EdgeToEdgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_roi):
        super().__init__()
        self.n_roi = n_roi
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, n_roi))
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(n_roi, 1))

    def forward(self, x):
        row_features = self.row_conv(x).repeat(1, 1, 1, self.n_roi)
        col_features = self.col_conv(x).repeat(1, 1, self.n_roi, 1)
        return row_features + col_features


class BrainNetCNN(nn.Module):
    def __init__(self, n_roi, channels=(32, 64, 128, 256), dropout=0.3, use_batchnorm=False):
        super().__init__()
        e2e1_c, e2e2_c, e2n_c, n2g_c = channels
        self.e2e1 = EdgeToEdgeBlock(1, e2e1_c, n_roi)
        self.e2e2 = EdgeToEdgeBlock(e2e1_c, e2e2_c, n_roi)
        self.e2n = nn.Conv2d(e2e2_c, e2n_c, kernel_size=(1, n_roi))
        self.n2g = nn.Conv2d(e2n_c, n2g_c, kernel_size=(n_roi, 1))
        self.use_batchnorm = use_batchnorm

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(e2e1_c)
            self.bn2 = nn.BatchNorm2d(e2e2_c)
            self.bn3 = nn.BatchNorm2d(e2n_c)
            self.bn4 = nn.BatchNorm2d(n2g_c)

        self.act = nn.LeakyReLU(0.33)
        self.drop = nn.Dropout(dropout)
        hidden = max(64, n2g_c // 2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n2g_c, hidden),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _block(self, x, layer, bn=None):
        x = layer(x)
        if bn is not None:
            x = bn(x)
        return self.drop(self.act(x))

    def forward(self, x):
        x = self._block(x, self.e2e1, self.bn1 if self.use_batchnorm else None)
        x = self._block(x, self.e2e2, self.bn2 if self.use_batchnorm else None)
        x = self._block(x, self.e2n, self.bn3 if self.use_batchnorm else None)
        x = self._block(x, self.n2g, self.bn4 if self.use_batchnorm else None)
        return self.head(x).squeeze(-1)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def search_best_threshold(y_true, y_prob, metric="balanced_accuracy"):
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.2, 0.8, 61):
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        score = metrics[metric]
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def collect_predictions(model, loader, loss_fn):
    model.eval()
    losses = []
    probs = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(yb.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets).astype(int)
    return float(np.mean(losses)), y_true, y_prob


def build_optimizer(cfg, model):
    if cfg["optimizer"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    return torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])


def run_single_fold(x_train_full, y_train_full, x_test, y_test, cfg):
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=cfg["val_size"],
        stratify=y_train_full,
        random_state=SEED,
    )

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    train_loader = DataLoader(
        FCMatrixDataset(x_train, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        FCMatrixDataset(x_val, y_val),
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        FCMatrixDataset(x_test, y_test),
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    model = BrainNetCNN(
        N_ROI,
        channels=CHANNEL_PRESETS[cfg["channel_preset"]],
        dropout=cfg["dropout"],
        use_batchnorm=cfg["use_batchnorm"],
    ).to(device)

    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], dtype=torch.float32).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(cfg, model)
    scheduler = None
    if cfg["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg["scheduler_factor"],
            patience=cfg["scheduler_patience"],
        )

    best_auc = -1.0
    best_state = None
    best_threshold = 0.5
    wait = 0
    history_rows = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, y_val_true, y_val_prob = collect_predictions(model, val_loader, loss_fn)
        val_auc = roc_auc_score(y_val_true, y_val_prob)
        val_threshold, val_bal_acc = search_best_threshold(y_val_true, y_val_prob, metric=cfg["threshold_metric"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_threshold": val_threshold,
                "val_balanced_accuracy": val_bal_acc,
            }
        )

        print(
            f"      epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_threshold={val_threshold:.2f}"
        )

        if scheduler is not None:
            scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = deepcopy(model.state_dict())
            best_threshold = val_threshold
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"      early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_loss, y_test_true, y_test_prob = collect_predictions(model, test_loader, loss_fn)
    test_metrics = compute_binary_metrics(y_test_true, y_test_prob, threshold=best_threshold)
    test_metrics["test_loss"] = test_loss
    test_metrics["best_val_roc_auc"] = best_auc
    test_metrics["selected_threshold"] = best_threshold
    return test_metrics, y_test_prob, pd.DataFrame(history_rows)


def run_cross_validation(x, y, cfg, config_name):
    output_dir = RESULT_ROOT / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=SEED)
    fold_rows = []
    oof_rows = []
    history_frames = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        print(f"  fold {fold}/{cfg['n_splits']}")
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        fold_metrics, test_prob, history_df = run_single_fold(x_train, y_train, x_test, y_test, cfg)
        print(
            f"    test accuracy={fold_metrics['accuracy']:.4f} "
            f"bal_acc={fold_metrics['balanced_accuracy']:.4f} "
            f"f1={fold_metrics['f1']:.4f} auc={fold_metrics['roc_auc']:.4f} "
            f"threshold={fold_metrics['selected_threshold']:.2f}"
        )

        fold_rows.append({"task": "Sex", "config_name": config_name, "fold": fold, **fold_metrics})
        history_df["fold"] = fold
        history_df["config_name"] = config_name
        history_frames.append(history_df)

        for i, idx in enumerate(test_idx):
            oof_rows.append(
                {
                    "task": "Sex",
                    "config_name": config_name,
                    "fold": fold,
                    "participant_id": df_all.iloc[idx]["participant_id"],
                    "y_true": int(y_test[i]),
                    "y_prob": float(test_prob[i]),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    history_df = pd.concat(history_frames, ignore_index=True)

    summary_df = fold_df[
        ["accuracy", "balanced_accuracy", "f1", "roc_auc", "test_loss", "selected_threshold"]
    ].agg(["mean", "std"]).T
    summary_df = summary_df.reset_index()
    summary_df.columns = ["metric", "mean", "std"]
    summary_df["config_name"] = config_name

    fold_df.to_csv(output_dir / "sex_fold_metrics.csv", index=False)
    oof_df.to_csv(output_dir / "sex_oof_predictions.csv", index=False)
    history_df.to_csv(output_dir / "sex_training_history.csv", index=False)
    summary_df.to_csv(output_dir / "sex_summary.csv", index=False)
    with (output_dir / "sex_cfg.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return fold_df, summary_df


print("Loading data...")
print("device:", device)
fc_csv = sorted(
    DATA_DIR.glob("TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson*.csv"),
    key=lambda p: p.stat().st_size,
    reverse=True,
)[0]
label_xlsx = list(DATA_DIR.glob("TRAINING_SOLUTIONS*.xlsx"))[0]
meta_xlsx = list(DATA_DIR.glob("TRAIN_QUANTITATIVE_METADATA*.xlsx"))[0]

df_fc = pd.read_csv(fc_csv)
df_label = pd.read_excel(label_xlsx)
df_meta = pd.read_excel(meta_xlsx)

df_all = df_fc.merge(df_label, on="participant_id").merge(df_meta, on="participant_id")
fc_columns = [col for col in df_fc.columns if col != "participant_id"]
N_ROI = int((1 + math.sqrt(1 + 8 * len(fc_columns))) / 2)

fc_raw = df_all[fc_columns].to_numpy(dtype=np.float32)
fc_z = fisher_z_transform(fc_raw)
fc_matrices = edge_vectors_to_matrices(fc_z, N_ROI)
X_all = fc_matrices[:, None, :, :].astype(np.float32)
y_sex = df_all["Sex_F"].to_numpy(dtype=np.int64)

comparison_rows = []
for config_name, cfg in CONFIGS.items():
    print(f"\n=== Running {config_name} ===")
    fold_df, summary_df = run_cross_validation(X_all, y_sex, cfg, config_name)
    summary_map = {
        row["metric"]: {"mean": row["mean"], "std": row["std"]}
        for _, row in summary_df.iterrows()
    }
    comparison_rows.append(
        {
            "config_name": config_name,
            "accuracy_mean": summary_map["accuracy"]["mean"],
            "accuracy_std": summary_map["accuracy"]["std"],
            "balanced_accuracy_mean": summary_map["balanced_accuracy"]["mean"],
            "balanced_accuracy_std": summary_map["balanced_accuracy"]["std"],
            "f1_mean": summary_map["f1"]["mean"],
            "f1_std": summary_map["f1"]["std"],
            "roc_auc_mean": summary_map["roc_auc"]["mean"],
            "roc_auc_std": summary_map["roc_auc"]["std"],
            "test_loss_mean": summary_map["test_loss"]["mean"],
            "test_loss_std": summary_map["test_loss"]["std"],
            "selected_threshold_mean": summary_map["selected_threshold"]["mean"],
            "selected_threshold_std": summary_map["selected_threshold"]["std"],
        }
    )

comparison_df = pd.DataFrame(comparison_rows).sort_values("roc_auc_mean", ascending=False)
comparison_path = RESULT_ROOT / "sex_tuning_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\n=== Final comparison ===")
print(comparison_df.to_string(index=False))
print(f"\nSaved comparison to {comparison_path}")
