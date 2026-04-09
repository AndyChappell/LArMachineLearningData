import h5py, math, os, torch, uproot
import awkward as ak, numpy as np

try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except Exception:
    from tqdm import tqdm


import os
import math
import h5py
import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm


def root_to_hdf5(in_filename, treename, out_filename):
    """
    Convert ROOT LArTPC hit data to an efficient HDF5 format.

    Input branches expected:
        - 'xx', 'zz'              : Cartesian hit coordinates (relative to vertex)
        - 'r', 'cosTheta', 'sinTheta' : Polar coordinates
        - 'width'                 : hit width (drift direction)
        - 'adc'                   : (already log-scaled)
        - 'semantic_label'        : per-hit class label

    Output HDF5 datasets:
        - hits:      (total_hits, F)
        - labels:    (total_hits,)
        - event_ptr: (num_events + 1,)

    where:
        event i corresponds to hits[event_ptr[i]:event_ptr[i+1]]

    Args:
        in_filename (str): Input ROOT file
        treename (str): ROOT tree name
        out_filename (str): Output HDF5 file
    """

    if os.path.exists(out_filename):
        print(f"{out_filename} already exists. Skipping processing.")
        with h5py.File(out_filename, 'r') as hf:
            num_events = hf['event_ptr'].shape[0] - 1
        return out_filename, num_events

    branches = [
        'xx', 'zz',
        'r', 'cosTheta', 'sinTheta',
        'width', 'adc',
        'semantic_label'
    ]

    hits_list = []
    labels_list = []
    event_ptr = [0]

    # Get number of entries for progress tracking
    with uproot.open(f"{in_filename}:{treename}") as tree:
        num_entries = tree.num_entries
        step_size = max(1, num_entries // 10)

    for arrays in tqdm(
        uproot.iterate(
            f"{in_filename}:{treename}",
            branches,
            step_size=step_size,
            library='ak'
        ),
        total=int(math.ceil(num_entries / step_size)),
        desc="Processing ROOT → HDF5"
    ):

        # Build feature tensor per hit
        hits_ak = ak.concatenate([
            arrays['xx'][..., None],
            arrays['zz'][..., None],
            arrays['r'][..., None],
            arrays['cosTheta'][..., None],
            arrays['sinTheta'][..., None],
            arrays['width'][..., None],
            arrays['adc'][..., None]
        ], axis=-1)

        labels_ak = arrays['semantic_label']

        # Loop over events (variable-length)
        for i in range(len(hits_ak)):
            event_hits = ak.to_numpy(hits_ak[i])
            if event_hits.shape[0] == 0:
                continue

            event_labels = ak.to_numpy(labels_ak[i] - 1)  # adjust if needed

            hits_list.append(event_hits)
            labels_list.append(event_labels)

            event_ptr.append(event_ptr[-1] + event_hits.shape[0])

    # Concatenate all events
    hits_all = np.concatenate(hits_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    event_ptr = np.array(event_ptr, dtype=np.int64)

    # Write to HDF5
    with h5py.File(out_filename, 'w') as hf:
        hf.create_dataset("hits", data=hits_all, compression="lzf")
        hf.create_dataset("labels", data=labels_all, compression="lzf")
        hf.create_dataset("event_ptr", data=event_ptr)

    print(f"Saved {len(event_ptr) - 1} events to {out_filename}")
    return out_filename, len(event_ptr) - 1
