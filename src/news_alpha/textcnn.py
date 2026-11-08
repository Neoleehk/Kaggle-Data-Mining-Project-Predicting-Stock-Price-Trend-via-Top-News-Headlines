from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
import re
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts: Iterable[str], min_freq: int = 2, max_vocab: int = 20000) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> np.ndarray:
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)]
    if len(ids) >= max_len:
        return np.asarray(ids[:max_len], dtype=np.int64)
    out = np.zeros(max_len, dtype=np.int64)
    out[: len(ids)] = ids
    return out


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: dict[str, int], max_len: int) -> None:
        self.inputs = np.vstack([encode_text(t, vocab, max_len) for t in texts])
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.inputs[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)  # [B, L, E]
        emb = emb.transpose(1, 2)  # [B, E, L]
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        feat = torch.cat(pooled, dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)


@dataclass(frozen=True)
class TrainConfig:
    max_len: int = 128
    embed_dim: int = 128
    num_filters: int = 128
    dropout: float = 0.3
    min_freq: int = 2
    max_vocab: int = 20000
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


def _device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_and_predict_proba(
    train_texts: list[str],
    train_y: np.ndarray,
    val_texts: list[str],
    val_y: np.ndarray,
    config: TrainConfig,
    device: str | None = None,
) -> np.ndarray:
    set_seed(config.seed)
    run_device = _device(device)

    vocab = build_vocab(train_texts, min_freq=config.min_freq, max_vocab=config.max_vocab)
    train_ds = TextDataset(train_texts, train_y, vocab, config.max_len)
    val_ds = TextDataset(val_texts, val_y, vocab, config.max_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_filters=config.num_filters,
        dropout=config.dropout,
    ).to(run_device)

    counts = np.bincount(train_y.astype(np.int64), minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(
        [counts.sum() / (2 * counts[0]), counts.sum() / (2 * counts[1])],
        dtype=torch.float32,
        device=run_device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    for _ in range(config.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(run_device)
            y = y.to(run_device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(run_device)
                y = y.to(run_device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * len(y)
                total_count += len(y)
        val_loss = total_loss / max(1, total_count)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(run_device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)


def train_and_predict_infer(
    train_texts: list[str],
    train_y: np.ndarray,
    early_stop_texts: list[str],
    early_stop_y: np.ndarray,
    infer_texts: list[str],
    config: TrainConfig,
    device: str | None = None,
) -> np.ndarray:
    set_seed(config.seed)
    run_device = _device(device)

    vocab = build_vocab(train_texts, min_freq=config.min_freq, max_vocab=config.max_vocab)
    train_ds = TextDataset(train_texts, train_y, vocab, config.max_len)
    early_ds = TextDataset(early_stop_texts, early_stop_y, vocab, config.max_len)
    infer_labels = np.zeros(len(infer_texts), dtype=np.int64)
    infer_ds = TextDataset(infer_texts, infer_labels, vocab, config.max_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    early_loader = DataLoader(early_ds, batch_size=config.batch_size, shuffle=False)
    infer_loader = DataLoader(infer_ds, batch_size=config.batch_size, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        num_filters=config.num_filters,
        dropout=config.dropout,
    ).to(run_device)

    counts = np.bincount(train_y.astype(np.int64), minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = torch.tensor(
        [counts.sum() / (2 * counts[0]), counts.sum() / (2 * counts[1])],
        dtype=torch.float32,
        device=run_device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    for _ in range(config.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(run_device)
            y = y.to(run_device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for x, y in early_loader:
                x = x.to(run_device)
                y = y.to(run_device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * len(y)
                total_count += len(y)
        val_loss = total_loss / max(1, total_count)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in infer_loader:
            x = x.to(run_device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p)
    return np.concatenate(probs, axis=0)
