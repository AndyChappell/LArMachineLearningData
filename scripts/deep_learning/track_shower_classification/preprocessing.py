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
        - 'x', 'z'                : Cartesian hit coordinates (relative to vertex)
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
        'x_rel', 'z_rel', 'x_abs', 'z_abs', 'width', 'adc',
        'r', 'cos_theta', 'sin_theta',
        'wire_pitch', 'wire_angle',
        'semantic_label'
    ]

    event_ptr = [0]
    num_features = 11

    # Get number of entries for progress tracking
    with uproot.open(f"{in_filename}:{treename}") as tree:
        num_entries = tree.num_entries
        step_size = max(1, num_entries // 10)

    # Write incrementally — avoids holding entire dataset in memory at once.
    with h5py.File(out_filename, 'w') as hf:
        ds_hits = hf.create_dataset(
            "hits", shape=(0, num_features), maxshape=(None, num_features),
            dtype=np.float32, compression="lzf", chunks=(4096, num_features)
        )
        ds_labels = hf.create_dataset(
            "labels", shape=(0,), maxshape=(None,),
            dtype=np.int64, compression="lzf", chunks=(4096,)
        )

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
            # Build feature array for entire chunk at once
            hits_ak = ak.concatenate([
                arrays['x_rel'][..., None],
                arrays['z_rel'][..., None],
                arrays['x_abs'][..., None],
                arrays['z_abs'][..., None],
                arrays['width'][..., None],
                arrays['adc'][..., None],
                arrays['r'][..., None],
                arrays['cos_theta'][..., None],
                arrays['sin_theta'][..., None],
                arrays['wire_pitch'][..., None],
                arrays['wire_angle'][..., None]
            ], axis=-1)

            labels_ak = arrays['semantic_label']

            # Per-event lengths for event_ptr — cheap, no full flatten needed
            lengths = np.asarray(ak.num(labels_ak, axis=1))
            nonempty = lengths > 0

            if not np.any(nonempty):
                continue

            # Flatten chunk to numpy in one call
            hits_np   = ak.to_numpy(ak.flatten(hits_ak[nonempty],   axis=1)).astype(np.float32)
            raw_labels = ak.to_numpy(ak.flatten(labels_ak[nonempty], axis=1))

            assert raw_labels.min() >= 1, (
                f"Expected 1-indexed labels but got min={raw_labels.min()}."
            )
            labels_np = (raw_labels - 1).astype(np.int64)

            # Append to resizable datasets
            n_existing = ds_hits.shape[0]
            n_new      = hits_np.shape[0]
            ds_hits.resize(n_existing + n_new, axis=0)
            ds_labels.resize(n_existing + n_new, axis=0)
            ds_hits[n_existing:]   = hits_np
            ds_labels[n_existing:] = labels_np

            # Build event_ptr increments from non-empty event lengths only
            for length in lengths[nonempty]:
                event_ptr.append(event_ptr[-1] + int(length))

        event_ptr_np = np.array(event_ptr, dtype=np.int64)
        hf.create_dataset("event_ptr", data=event_ptr_np)

    print(f"Saved {len(event_ptr) - 1} events to {out_filename}")
    return out_filename, len(event_ptr) - 1
