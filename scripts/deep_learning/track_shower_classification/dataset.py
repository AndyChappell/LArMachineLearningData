import h5py, torch, torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset
import h5py


class LArTPCSequenceDataset(Dataset):
    """
    Dataset for flat HDF5 LArTPC data.

    HDF5 structure:
        - hits:      (total_hits, F)
        - labels:    (total_hits,)
        - event_ptr: (num_events + 1,)

    Each item returns:
        {
            "hits":   (N_hits, F)
            "labels": (N_hits,)
        }
    """

    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.file = None

        # Read metadata only
        with h5py.File(self.h5_path, "r") as hf:
            self.num_events = hf["event_ptr"].shape[0] - 1

    def __len__(self):
        return self.num_events

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
            self.hits = self.file["hits"]
            self.labels = self.file["labels"]
            self.event_ptr = self.file["event_ptr"]

    def __getitem__(self, idx):
        self._ensure_open()

        if idx >= self.num_events:
            raise IndexError("Index out of range")

        start = self.event_ptr[idx]
        end = self.event_ptr[idx + 1]

        hits = torch.from_numpy(self.hits[start:end]).float()
        labels = torch.from_numpy(self.labels[start:end]).long()

        return {
            "hits": hits,
            "labels": labels,
        }


import torch


def collate_fn_pad(batch):
    """
    Pads variable-length hit sequences to max length in batch.

    Returns:
        hits:   (B, N_max, F)
        labels: (B, N_max)
        mask:   (B, N_max)  (True = valid)
    """

    hits = [b["hits"] for b in batch]
    labels = [b["labels"] for b in batch]

    lengths = [h.shape[0] for h in hits]
    max_len = max(lengths)
    feat_dim = hits[0].shape[1]

    B = len(batch)

    padded_hits = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    padded_labels = torch.full((B, max_len), -1, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (h, l) in enumerate(zip(hits, labels)):
        n = h.shape[0]
        padded_hits[i, :n] = h
        padded_labels[i, :n] = l
        mask[i, :n] = True

    return {
        "hits": padded_hits,
        "labels": padded_labels,
        "mask": mask,
    }


def compute_class_weights(dataloader, num_classes, device="cpu"):
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for batch in dataloader:
        labels = batch["labels"]

        # --- Handle list vs tensor ---
        if isinstance(labels, list):
            labels = torch.cat(labels, dim=0)

        # --- Ensure 1D ---
        labels = labels.reshape(-1)

        labels = labels.to(torch.long)
        # Ignore padding
        valid = (labels >= 0)
        labels = labels[valid]

        if labels.numel() == 0:
            continue

        counts = torch.bincount(labels, minlength=num_classes)
        class_counts += counts

    class_counts = class_counts.float().clamp(min=1)
    weights = class_counts.sum() / (num_classes * class_counts)

    return weights.to(device)
