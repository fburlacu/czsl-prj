"""
GDE/LDE conditional compositional classification for 3-attribute triplets
(attr1, attr2, obj) = (color, material, object).

Prediction is hierarchical and geodesic:
  Stage 1 - pick the object by comparing the image against composed object
            embeddings (Exp_mu(v_o) for GDE; context+v_o for LDE).
  Stage 2 - conditioned on the chosen object, compare the image against the
            candidate triplets sharing that object and read off (color, material).

Scoring always routes through the factorizer's combine_ideal_words, so GDE
(geodesic) and LDE (flat) genuinely differ.

Open vs closed world is controlled by the candidate triplet pool
(test_dataset.triplets, which composition_dataset expands to the full product
when open_world=True). NOTE: a factored predictor scores primitives, not whole
combinations, so closed and open world give near-identical numbers by design -
this is expected, not a bug. Focus on closed world.

Metrics returned (no AUC): per-dimension accuracy (color/material/object) and
full-triplet accuracy, each split into seen / unseen by whether the true triplet
appeared in training.
"""

from math import exp

import numpy as np
import torch


def compute_logits(image_embs, label_embs, logit_scale=exp(0.07)):
    logit_scale = min(logit_scale, 100.0)
    return (logit_scale * image_embs @ label_embs.t()).to("cpu")


# -----------------------------------------------------------------------------
# Candidate pool (open/closed world lives here)
# -----------------------------------------------------------------------------

def build_candidate_triplets(test_dataset):
    """
    Triplets the classifier may predict.
    closed world: train+test triplets (test_dataset.triplets as parsed)
    open  world : full color x material x object product
                  (composition_dataset sets self.triplets = full_triplets when
                   open_world=True, so reading .triplets is the single source of truth)
    """
    return list(test_dataset.triplets)


# -----------------------------------------------------------------------------
# Compose candidate embeddings through the factorizer (GDE or LDE)
# -----------------------------------------------------------------------------

def compose_object_embeddings(factorizer, objs, device):
    """Exp_mu(v_o) for GDE, context+v_o for LDE - one embedding per object."""
    o_idx = torch.tensor([factorizer.obj2idx[o] for o in objs],
                         device=factorizer.device)
    o_IW = factorizer.obj_IW[o_idx]
    composed = factorizer.combine_ideal_words(o_IW)
    return composed.to(device)


def compose_triplet_embeddings(factorizer, triplets, device):
    """Composed embedding for each full (color, material, object) triplet."""
    a1_idx = torch.tensor([factorizer.attr1_idx[a1] for a1, a2, o in triplets],
                          device=factorizer.device)
    a2_idx = torch.tensor([factorizer.attr2_idx[a2] for a1, a2, o in triplets],
                          device=factorizer.device)
    o_idx  = torch.tensor([factorizer.obj2idx[o]   for a1, a2, o in triplets],
                          device=factorizer.device)

    a1_IW = factorizer.attr1_IW[a1_idx]
    a2_IW = factorizer.attr2_IW[a2_idx]
    o_IW  = factorizer.obj_IW[o_idx]

    composed = factorizer.combine_ideal_words(a1_IW, a2_IW, o_IW)
    return composed.to(device)


# -----------------------------------------------------------------------------
# Conditional prediction
# -----------------------------------------------------------------------------

def conditional_predict(factorizer, image_embs, candidate_triplets, device):
    cand_objs = sorted(set(o for (_a1, _a2, o) in candidate_triplets))

    # Stage 1: object
    obj_embs = compose_object_embeddings(factorizer, cand_objs, device)   # (|O|, d)
    obj_scores = compute_logits(image_embs, obj_embs)                     # (N, |O|)
    best_obj_local = obj_scores.argmax(dim=1)
    best_obj_names = [cand_objs[i] for i in best_obj_local.tolist()]

    # Precompute composed embeddings for all candidate triplets, score once
    composed_all = compose_triplet_embeddings(factorizer, candidate_triplets, device)
    full_scores = compute_logits(image_embs, composed_all)               # (N, T)

    triplets_by_obj = {}
    for j, t in enumerate(candidate_triplets):
        triplets_by_obj.setdefault(t[2], []).append((j, t))

    # Stage 2: attributes conditioned on predicted object
    a1_pred, a2_pred, o_pred_names = [], [], []
    for i in range(len(image_embs)):
        obj = best_obj_names[i]
        cols_triplets = triplets_by_obj[obj]
        cols = torch.tensor([j for j, _t in cols_triplets])
        sub_scores = full_scores[i, cols]
        best_local = sub_scores.argmax().item()
        chosen = cols_triplets[best_local][1]
        a1_pred.append(factorizer.attr1_idx[chosen[0]])
        a2_pred.append(factorizer.attr2_idx[chosen[1]])
        o_pred_names.append(chosen[2])

    return {
        "attr1_pred": torch.LongTensor(a1_pred),
        "attr2_pred": torch.LongTensor(a2_pred),
        "obj_pred_names": o_pred_names,
    }


# -----------------------------------------------------------------------------
# Metric
# -----------------------------------------------------------------------------

def compute_metrics(pred, all_triplets_true, test_dataset):
    seen_set = set(test_dataset.train_triplets)
    seen_ids = [i for i, t in enumerate(all_triplets_true) if t in seen_set]
    unseen_ids = [i for i, t in enumerate(all_triplets_true) if t not in seen_set]

    a1_true = torch.LongTensor([test_dataset.attr1_idx[a1] for a1, a2, o in all_triplets_true])
    a2_true = torch.LongTensor([test_dataset.attr2_idx[a2] for a1, a2, o in all_triplets_true])
    o_true  = torch.LongTensor([test_dataset.obj2idx[o]   for a1, a2, o in all_triplets_true])

    a1_pred = pred["attr1_pred"]
    a2_pred = pred["attr2_pred"]
    o_pred  = torch.LongTensor([test_dataset.obj2idx[o] for o in pred["obj_pred_names"]])

    def split_acc(p, t):
        c = (p == t).numpy()
        return {
            "all":    float(c.mean()),
            "seen":   float(c[seen_ids].mean())   if seen_ids   else 0.0,
            "unseen": float(c[unseen_ids].mean()) if unseen_ids else 0.0,
        }

    a1 = split_acc(a1_pred, a1_true)
    a2 = split_acc(a2_pred, a2_true)
    ob = split_acc(o_pred,  o_true)

    triplet_correct = ((a1_pred == a1_true) &
                       (a2_pred == a2_true) &
                       (o_pred  == o_true)).numpy()

    return {
        # color
        "attr1_acc":          a1["all"],
        "seen_attr1_acc":     a1["seen"],
        "unseen_attr1_acc":   a1["unseen"],
        # material
        "attr2_acc":          a2["all"],
        "seen_attr2_acc":     a2["seen"],
        "unseen_attr2_acc":   a2["unseen"],
        # object
        "obj_acc":            ob["all"],
        "seen_obj_acc":       ob["seen"],
        "unseen_obj_acc":     ob["unseen"],
        # full triplet
        "triplet_acc":        float(triplet_correct.mean()),
        "seen_triplet_acc":   float(triplet_correct[seen_ids].mean())   if seen_ids   else 0.0,
        "unseen_triplet_acc": float(triplet_correct[unseen_ids].mean()) if unseen_ids else 0.0,
    }


# -----------------------------------------------------------------------------
# Entry point used by classification.py main()
# -----------------------------------------------------------------------------

def run_conditional_eval(factorizer, test_dataset, device):
    image_embs, all_triplets_true = test_dataset.load_all_image_embs()
    image_embs = image_embs.to(device)

    candidate_triplets = build_candidate_triplets(test_dataset)
    pred = conditional_predict(factorizer, image_embs, candidate_triplets, device)
    return compute_metrics(pred, all_triplets_true, test_dataset)