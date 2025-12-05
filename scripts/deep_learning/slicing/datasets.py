try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except Exception:
    from tqdm import tqdm


import h5py, torch, torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class LArTPCSequenceDataset(Dataset):
    """
    Dataset for pre-processed LArTPC event data (hdf5 format).
    Each event group must contain:
      - "hits"         : (N_hits, F)  float32 input features
      - "slice_labels" : (N_hits,)    int32 slice ids (>=0), -1 for background
      - "cp_labels"    : (N_hits,)    int32 cp flags, 1 for CP, -1 for background
    The dataset lazily opens the HDF5 file per worker (safe with DataLoader num_workers>0).
    """
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.file = None

        # count number of events
        with h5py.File(self.h5_path, "r") as hf:
            self.event_keys = sorted(hf["events"].keys(), key=lambda x: int(x))
            self.num_events = len(self.event_keys)

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        if idx >= self.num_events:
            raise IndexError("Index out of range")

        event_name = self.event_keys[idx]
        group = self.file[f"events/{event_name}"]

        hits = torch.from_numpy(group["hits"][()]).float()
        slice_labels = torch.from_numpy(group["slice_labels"][()]).long()
        cp_labels = torch.from_numpy(group["cp_labels"][()]).long()

        return {
            "hits": hits,
            "slice_labels": slice_labels,
            "cp_labels": cp_labels
        }


def collate_fn(batch):
    """
    Collate variable-length events into a padded batch.

    Each batch element is a dict:
        "hits": (N, F)
        "slice_labels": (N,)
        "cp_labels": (N,)

    Returns:
        hits:         (B, Nmax, F)
        slice_labels: (B, Nmax)  padded with -1
        cp_labels:    (B, Nmax)  padded with -1
        mask:         (B, Nmax)  bool mask of valid entries
    """

    hits_list = [b["hits"] for b in batch]
    slice_list = [b["slice_labels"] for b in batch]
    cp_list    = [b["cp_labels"] for b in batch]

    max_len = max(h.shape[0] for h in hits_list)

    padded_hits = []
    padded_slice = []
    padded_cp = []
    masks = []

    for h, sl, cl in zip(hits_list, slice_list, cp_list):
        pad_len = max_len - h.shape[0]

        # Pad hits: shape (N, F), so pad (features=0, sequence=pad_len)
        padded_hits.append(F.pad(h, (0, 0, 0, pad_len)))

        # Pad labels on the right
        padded_slice.append(F.pad(sl, (0, pad_len), value=-1))
        padded_cp.append(F.pad(cl, (0, pad_len), value=-1))

        # Mask: True for real hits, False for padded
        masks.append(torch.cat([
            torch.ones(h.shape[0], dtype=torch.bool),
            torch.zeros(pad_len, dtype=torch.bool)
        ]))

    return {
        "hits": torch.stack(padded_hits),          # (B, Nmax, F)
        "slice_labels": torch.stack(padded_slice), # (B, Nmax)
        "cp_labels": torch.stack(padded_cp),       # (B, Nmax)
        "mask": torch.stack(masks)                 # (B, Nmax)
    }