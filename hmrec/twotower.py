from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hmrec.splits import ensure_before_cut

MASK_VALUE = -1e9


@dataclass
class Embeddings:
    article_ids: np.ndarray
    item_vectors: np.ndarray
    customer_vectors: np.ndarray
    has_history: np.ndarray
    losses: list


class TwoTower(nn.Module):
    def __init__(self, n_customers, n_items, dim):
        super().__init__()
        self.customers = nn.Embedding(n_customers, dim)
        self.items = nn.Embedding(n_items, dim)
        nn.init.normal_(self.customers.weight, std=0.1)
        nn.init.normal_(self.items.weight, std=0.1)

    def encode_customers(self, ids):
        return F.normalize(self.customers(ids), dim=-1)

    def encode_items(self, ids):
        return F.normalize(self.items(ids), dim=-1)


def build_pairs(history):
    article_ids, item_index = np.unique(
        history["article_id"].to_numpy(), return_inverse=True
    )
    customer_index = history["customer_idx"].to_numpy().astype("int64")
    counts = np.bincount(item_index, minlength=len(article_ids))
    log_q = np.log(counts / counts.sum())
    return customer_index, item_index.astype("int64"), article_ids, log_q


def in_batch_loss(user_vecs, item_vecs, customer_ids, item_ids, log_q, temperature, use_logq):
    logits = user_vecs @ item_vecs.T / temperature
    if use_logq:
        logits = logits - log_q[item_ids].unsqueeze(0)
    same_item = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)
    same_customer = customer_ids.unsqueeze(0) == customer_ids.unsqueeze(1)
    diagonal = torch.eye(len(item_ids), dtype=torch.bool)
    false_negative = (same_item | same_customer) & ~diagonal
    logits = logits.masked_fill(false_negative, MASK_VALUE)
    targets = torch.arange(len(item_ids))
    return F.cross_entropy(logits, targets)


def fit(
    history,
    cut,
    n_customers,
    dim=32,
    epochs=8,
    batch_size=1024,
    lr=0.01,
    temperature=0.1,
    use_logq=True,
    seed=42,
):
    ensure_before_cut(history, cut)
    torch.manual_seed(seed)

    customer_index, item_index, article_ids, log_q = build_pairs(history)
    model = TwoTower(n_customers, len(article_ids), dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    customers_t = torch.from_numpy(customer_index)
    items_t = torch.from_numpy(item_index)
    log_q_t = torch.from_numpy(log_q).float()
    n_rows = len(customers_t)

    losses = []
    for epoch in range(epochs):
        order = torch.randperm(n_rows)
        total = 0.0
        batches = 0
        for start in range(0, n_rows, batch_size):
            batch = order[start : start + batch_size]
            if len(batch) < 2:
                continue
            c = customers_t[batch]
            i = items_t[batch]
            loss = in_batch_loss(
                model.encode_customers(c),
                model.encode_items(i),
                c,
                i,
                log_q_t,
                temperature,
                use_logq,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        losses.append(total / max(batches, 1))
        print(f"epoch {epoch + 1}/{epochs} loss {losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        item_vectors = model.encode_items(torch.arange(len(article_ids))).numpy()
        customer_vectors = model.encode_customers(torch.arange(n_customers)).numpy()

    has_history = np.zeros(n_customers, dtype=bool)
    has_history[customer_index] = True

    return Embeddings(article_ids, item_vectors, customer_vectors, has_history, losses)