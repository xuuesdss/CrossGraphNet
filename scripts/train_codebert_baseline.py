import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


class CodeDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        code_field: str = "code",
        label_field: str = "label",
        max_length: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.code_field = code_field
        self.label_field = label_field
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item[self.code_field]
        label = int(item[self.label_field])

        enc = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)


class CodeBERTBaseline(nn.Module):
    def __init__(self,model_name="/home/xu/FedVulGuard/CodeBert", frozen: bool = True,
                 cls_hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name,local_files_only=True)
        self.classifier = MLPClassifier(input_dim=768, hidden_dim=cls_hidden_dim, dropout=dropout)

        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]   # [CLS]
        logits = self.classifier(cls_vec)
        return logits


@dataclass
class Metrics:
    loss: float
    f1: float
    auc: float
    precision: float
    recall: float


def compute_metrics(y_true, y_prob, y_pred) -> Metrics:
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return Metrics(
        loss=0.0,
        f1=f1,
        auc=auc,
        precision=precision,
        recall=recall,
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    metrics = compute_metrics(y_true, y_prob, y_pred)
    metrics.loss = total_loss / len(loader.dataset)
    return metrics


def train_one_epoch(model, loader, device, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="/home/xu/FedVulGuard/CodeBert")
    parser.add_argument("--code_field", type=str, default="code")
    parser.add_argument("--label_field", type=str, default="label")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,local_files_only=True,use_fast=False)

    train_data = read_jsonl(args.train_path)
    val_data = read_jsonl(args.val_path)
    test_data = read_jsonl(args.test_path)

    train_ds = CodeDataset(
        train_data, tokenizer,
        code_field=args.code_field,
        label_field=args.label_field,
        max_length=args.max_length
    )
    val_ds = CodeDataset(
        val_data, tokenizer,
        code_field=args.code_field,
        label_field=args.label_field,
        max_length=args.max_length
    )
    test_ds = CodeDataset(
        test_data, tokenizer,
        code_field=args.code_field,
        label_field=args.label_field,
        max_length=args.max_length
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CodeBERTBaseline(
        model_name=args.model_name,
        frozen=args.frozen,
        cls_hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.frozen:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps
        )
    else:
        encoder_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                classifier_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": args.encoder_lr},
                {"params": classifier_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay
        )
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps
        )

    best_val_f1 = -1.0
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        val_metrics = evaluate(model, val_loader, device, criterion)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "val_f1": val_metrics.f1,
            "val_auc": val_metrics.auc,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
        }
        history.append(record)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_f1={val_metrics.f1:.4f} "
            f"val_auc={val_metrics.auc:.4f}"
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[Early Stop] patience={args.patience}")
                break

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
    test_metrics = evaluate(model, test_loader, device, criterion)

    result = {
        "seed": args.seed,
        "frozen": args.frozen,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_loss": test_metrics.loss,
        "test_f1": test_metrics.f1,
        "test_auc": test_metrics.auc,
        "test_precision": test_metrics.precision,
        "test_recall": test_metrics.recall,
    }

    save_json(history, os.path.join(args.output_dir, "history.json"))
    save_json(result, os.path.join(args.output_dir, "result.json"))

    print("\n===== Final Test =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()