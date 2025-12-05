import numpy as np

def compute_instance_metrics(cluster_ids, slice_labels):
    """
    Inputs:
        cluster_ids: (N,) predicted cluster assignment (tensor or array)
        slice_labels: (N,) ground truth instance labels (tensor or array)
                      -1 for background
    """

    # Ensure CPU NumPy arrays
    if hasattr(cluster_ids, "device"):
        cluster_ids = cluster_ids.detach().cpu().numpy()
    else:
        cluster_ids = np.asarray(cluster_ids)

    if hasattr(slice_labels, "device"):
        slice_labels = slice_labels.detach().cpu().numpy()
    else:
        slice_labels = np.asarray(slice_labels)

    # Mask to ignore background (-1)
    mask = slice_labels >= 0
    pred = cluster_ids[mask]
    truth = slice_labels[mask]

    # -------------------------------
    # Adjusted Rand Index
    # -------------------------------
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(truth, pred)

    # -------------------------------
    # Purity / Efficiency / F1
    # -------------------------------
    unique_pred = np.unique(pred)
    purity_list, eff_list, f1_list = [], [], []

    for c in unique_pred:
        pred_mask = pred == c
        truth_in_cluster = truth[pred_mask]

        if len(truth_in_cluster) == 0:
            continue

        # Dominant true label
        vals, counts = np.unique(truth_in_cluster, return_counts=True)
        dominant = vals[np.argmax(counts)]
        tp = np.max(counts)

        # Purity
        purity = tp / len(truth_in_cluster)
        purity_list.append(purity)

        # Efficiency (recall)
        total_true = np.sum(truth == dominant)
        eff = tp / total_true
        eff_list.append(eff)

        # F1 score
        f1 = 2 * purity * eff / (purity + eff + 1e-12)
        f1_list.append(f1)

    # -------------------------------
    # Background rejection
    # -------------------------------
    bg_mask = slice_labels < 0
    bg_pred = cluster_ids[bg_mask]

    num_bg = len(bg_pred)
    # background is never assigned cluster -1 by OC, but if you ever want:
    bg_correct = np.sum(bg_pred == -1) if -1 in unique_pred else 0
    bg_rejection = bg_correct / num_bg if num_bg > 0 else 1.0

    return {
        "ARI": float(ari),
        "purity_mean": float(np.mean(purity_list)) if purity_list else 0.0,
        "eff_mean": float(np.mean(eff_list)) if eff_list else 0.0,
        "f1_mean": float(np.mean(f1_list)) if f1_list else 0.0,
        "bg_rejection": bg_rejection,
    }
