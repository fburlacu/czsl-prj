import argparse
import json
import os
import torch
import numpy as np
import random
from tqdm import tqdm
from math import exp
from collections import defaultdict
from scipy.stats import hmean


from datasets.read_datasets import DATASET_PATHS
from datasets.composition_dataset import CompositionDatasetEmbeddings
from utils.utils import set_seed
from factorizers import FACTORIZERS


def accuracy(y_pred, y_true):
    n_correct = torch.eq(y_pred, y_true).sum().item()
    n_tot = y_true.size(0)
    return n_correct / n_tot


class Evaluator:
    '''
    Part of the code for this class is taken from:
    https://github.com/tttyuntian/vlm_primitive_concepts/blob/main/vlm_concept/mit_states/train_retrieval_model.py
    '''
    def __init__(self, dset: CompositionDatasetEmbeddings):
        self.dset = dset

        if dset.phase == 'train':
            test_pairs_set = set(dset.train_pairs)
            test_pairs_gt  = set(dset.train_pairs)
        elif dset.phase == 'val':
            test_pairs_set = set(dset.val_pairs + dset.train_pairs)
            test_pairs_gt  = set(dset.val_pairs)
        else:
            test_pairs_set = set(dset.test_pairs + dset.train_pairs)
            test_pairs_gt  = set(dset.test_pairs)

        if not dset.open_world:
            self.closed_mask = torch.BoolTensor(
                [1 if pair in test_pairs_set else 0 for pair in dset.pairs]
            )

        self.seen_pairs_set = set(dset.train_pairs)
        mask = [1 if pair in self.seen_pairs_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        self.pairs_idx2ao_idx = torch.LongTensor([
            (dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs
        ])

    def get_attr_obj_from_pairs(self, pairs):
        attrs = self.pairs_idx2ao_idx[pairs, 0]
        objs  = self.pairs_idx2ao_idx[pairs, 1]
        return attrs, objs

    def evaluate(self, y_pred_topk, y_true, seen_ids, unseen_ids):
        correct = torch.eq(y_pred_topk, y_true.unsqueeze(1)).any(1).numpy()
        all_acc    = np.mean(correct)
        seen_acc   = np.mean(correct[seen_ids])
        unseen_acc = np.mean(correct[unseen_ids])
        return {
            "all_acc": all_acc,
            "seen_acc": seen_acc,
            "unseen_acc": unseen_acc,
            "harmonic_mean": hmean([seen_acc, unseen_acc], axis=0),
            "macro_average_acc": (seen_acc + unseen_acc) * 0.5,
        }

    def predict(self, scores, topk, bias=0.0):
        scores = scores.clone()
        scores[:, ~self.seen_mask] += bias

        if not self.dset.open_world:
            scores[:, ~self.closed_mask] = -1e10

        _, pair_preds = scores.topk(topk, dim=1)
        attr_preds, obj_preds = self.get_attr_obj_from_pairs(pair_preds)
        return pair_preds, attr_preds, obj_preds

    def predict_sequential(self, scores_attr, scores_obj, attr_f2d, obj_f2d):
        attr_preds = attr_f2d[scores_attr.argmax(dim=1)]
        obj_preds  = obj_f2d[scores_obj.argmax(dim=1)]
        return attr_preds, obj_preds

    def predict_top_k(self, img_embs, scores_attr, scores_obj, attr_f2d, obj_f2d, factorizer, pick_top_k):
        logit_scale = min(exp(0.07), 100.0)
        N   = scores_attr.shape[0]
        s_a = scores_attr.shape[1]
        s_o = scores_obj.shape[1]
        device = img_embs.device

        full_scores = (
            scores_attr[:, :, None] + scores_obj[:, None, :]
        ).view(N, -1)

        _, topk_ = full_scores.topk(pick_top_k, dim=1)

        new_a = topk_ // s_o
        new_o = topk_ % s_o

        new_attr_IW = factorizer.attr_IW.to(device)[new_a]
        new_obj_IW  = factorizer.obj_IW.to(device)[new_o]

        NK = N * pick_top_k
        cand_embs    = factorizer.combine_ideal_words(new_attr_IW.view(NK, -1), new_obj_IW.view(NK, -1))
        img_expanded = img_embs.unsqueeze(1).expand(N, pick_top_k, -1).reshape(NK, -1)
        exact_scores = (logit_scale * (img_expanded * cand_embs).sum(dim=1)).view(N, pick_top_k)
        best_k = exact_scores.argmax(dim=1)

        attr_preds = attr_f2d[new_a[torch.arange(N), best_k]]
        obj_preds  = obj_f2d[new_o[torch.arange(N), best_k]]
        return attr_preds, obj_preds

    def get_overall_metrics(self, features, all_pairs_true, topk_list=[1], progress_bar=True):
        labels = torch.LongTensor([self.dset.pair2idx[pair] for pair in all_pairs_true])

        seen_ids   = [i for i, p in enumerate(all_pairs_true) if p in self.seen_pairs_set]
        unseen_ids = [i for i, p in enumerate(all_pairs_true) if p not in self.seen_pairs_set]

        overall_metrics = {}
        for topk in topk_list:
            pair_preds, attr_preds, obj_preds = self.predict(features, topk=topk, bias=0.)
            attr_true, obj_true = self.get_attr_obj_from_pairs(labels)

            unbiased_pair_acc = self.evaluate(pair_preds, labels,    seen_ids, unseen_ids)['all_acc']
            attr_acc          = self.evaluate(attr_preds, attr_true, seen_ids, unseen_ids)['all_acc']
            obj_acc           = self.evaluate(obj_preds,  obj_true,  seen_ids, unseen_ids)['all_acc']

            pair_preds, _, _ = self.predict(features, topk=topk, bias=1e3)
            full_unseen_metrics = self.evaluate(pair_preds, labels, seen_ids, unseen_ids)

            correct_scores  = features[np.arange(len(features)), labels][unseen_ids]
            max_seen_scores = features[unseen_ids][:, self.seen_mask].topk(topk, dim=1)[0][:, topk-1]
            pairs_correct   = torch.eq(pair_preds, labels.unsqueeze(1)).any(1).numpy()
            unseen_correct  = pairs_correct[unseen_ids]
            unseen_score_diff = max_seen_scores - correct_scores
            correct_unseen_score_diff = unseen_score_diff[unseen_correct] - 1e-4
            correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
            magic_binsize = 20
            bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
            bias_list = correct_unseen_score_diff[::bias_skip]

            all_metrics = []
            for bias in tqdm(bias_list, disable=not progress_bar):
                pair_preds, _, _ = self.predict(features, topk=topk, bias=bias)
                metrics = self.evaluate(pair_preds, labels, seen_ids, unseen_ids)
                all_metrics.append(metrics)
            all_metrics.append(full_unseen_metrics)

            seen_accs        = np.array([m['seen_acc']      for m in all_metrics])
            unseen_accs      = np.array([m['unseen_acc']    for m in all_metrics])
            best_seen_acc    = max(m['seen_acc']             for m in all_metrics)
            best_unseen_acc  = max(m['unseen_acc']           for m in all_metrics)
            best_harmonic_mean = max(m['harmonic_mean']      for m in all_metrics)
            auc = np.trapz(seen_accs, unseen_accs)

            overall_metrics[topk] = {
                "unbiased_pair_acc":   unbiased_pair_acc,
                "attr_acc":            attr_acc,
                "obj_acc":             obj_acc,
                "best_seen_acc":       best_seen_acc,
                "best_unseen_acc":     best_unseen_acc,
                "best_harmonic_mean":  best_harmonic_mean,
                "auc":                 auc,
            }
        return overall_metrics

    def get_fast_metrics(self, features, all_pairs_true, topk_list=[1]):
        '''Compute all metrics except auc. Much faster than get_overall_metrics.'''
        labels = torch.LongTensor([self.dset.pair2idx[pair] for pair in all_pairs_true])
        attr_true, obj_true = self.get_attr_obj_from_pairs(labels)

        seen_ids   = [i for i, p in enumerate(all_pairs_true) if p in self.seen_pairs_set]
        unseen_ids = [i for i, p in enumerate(all_pairs_true) if p not in self.seen_pairs_set]

        fast_metrics = {}
        for topk in topk_list:
            pair_preds, attr_preds, obj_preds = self.predict(features, topk=topk, bias=0.)
            unbiased_pair_acc = self.evaluate(pair_preds, labels,    seen_ids, unseen_ids)['all_acc']
            attr_acc          = self.evaluate(attr_preds, attr_true, seen_ids, unseen_ids)['all_acc']
            obj_acc           = self.evaluate(obj_preds,  obj_true,  seen_ids, unseen_ids)['all_acc']

            pair_preds, _, _ = self.predict(features, topk=topk, bias=-1e3)
            best_seen_acc = self.evaluate(pair_preds, labels, seen_ids, unseen_ids)['seen_acc']

            pair_preds, _, _ = self.predict(features, topk=topk, bias=1e3)
            best_unseen_acc = self.evaluate(pair_preds, labels, seen_ids, unseen_ids)['unseen_acc']

            fast_metrics[topk] = {
                "unbiased_pair_acc": unbiased_pair_acc,
                "attr_acc":          attr_acc,
                "obj_acc":           obj_acc,
                "best_seen_acc":     best_seen_acc,
                "best_unseen_acc":   best_unseen_acc,
            }
        return fast_metrics

    def get_sequential_metrics(self, scores_attr, scores_obj, all_pairs_true, attr_f2d, obj_f2d):
        labels = torch.LongTensor([self.dset.pair2idx[pair] for pair in all_pairs_true])
        attr_true, obj_true = self.get_attr_obj_from_pairs(labels)

        seen_ids   = [i for i, p in enumerate(all_pairs_true) if p in self.seen_pairs_set]
        unseen_ids = [i for i, p in enumerate(all_pairs_true) if p not in self.seen_pairs_set]

        attr_preds, obj_preds = self.predict_sequential(
            scores_attr, scores_obj, attr_f2d=attr_f2d, obj_f2d=obj_f2d
        )

        attr_acc = self.evaluate(attr_preds.unsqueeze(1), attr_true, seen_ids, unseen_ids)['all_acc']
        obj_acc  = self.evaluate(obj_preds.unsqueeze(1),  obj_true,  seen_ids, unseen_ids)['all_acc']

        attr_correct = torch.eq(attr_preds, attr_true)
        obj_correct  = torch.eq(obj_preds,  obj_true)
        pair_correct = (attr_correct & obj_correct).numpy()

        pair_acc        = np.mean(pair_correct)
        seen_pair_acc   = np.mean(pair_correct[seen_ids])   if seen_ids   else 0.0
        unseen_pair_acc = np.mean(pair_correct[unseen_ids]) if unseen_ids else 0.0

        return {
            "attr_acc":         attr_acc,
            "obj_acc":          obj_acc,
            "pair_acc":         pair_acc,
            "seen_pair_acc":    seen_pair_acc,
            "unseen_pair_acc":  unseen_pair_acc,
        }

    def get_top_k_metrics(self, img_embs, scores_attr, scores_obj, all_pairs_true, attr_f2d, obj_f2d, factorizer, pick_top_k):
        labels = torch.LongTensor([self.dset.pair2idx[pair] for pair in all_pairs_true])
        attr_true, obj_true = self.get_attr_obj_from_pairs(labels)

        seen_ids   = [i for i, p in enumerate(all_pairs_true) if p in self.seen_pairs_set]
        unseen_ids = [i for i, p in enumerate(all_pairs_true) if p not in self.seen_pairs_set]

        attr_preds, obj_preds = self.predict_top_k(
            img_embs, scores_attr, scores_obj, attr_f2d, obj_f2d, factorizer, pick_top_k
        )

        attr_acc = self.evaluate(attr_preds.unsqueeze(1), attr_true, seen_ids, unseen_ids)['all_acc']
        obj_acc  = self.evaluate(obj_preds.unsqueeze(1),  obj_true,  seen_ids, unseen_ids)['all_acc']

        attr_correct = torch.eq(attr_preds, attr_true)
        obj_correct  = torch.eq(obj_preds,  obj_true)
        pair_correct = (attr_correct & obj_correct).numpy()

        pair_acc        = np.mean(pair_correct)
        seen_pair_acc   = np.mean(pair_correct[seen_ids])   if seen_ids   else 0.0
        unseen_pair_acc = np.mean(pair_correct[unseen_ids]) if unseen_ids else 0.0

        return {
            "attr_acc":         attr_acc,
            "obj_acc":          obj_acc,
            "pair_acc":         pair_acc,
            "seen_pair_acc":    seen_pair_acc,
            "unseen_pair_acc":  unseen_pair_acc,
        }


def select_n_embs_per_pair(embeddings, all_pairs, n: int):
    '''Randomly selects up to n embeddings for each (attr, obj) pair.'''
    unique_pairs = sorted(set(all_pairs))
    pair_idx2img_embs = {pair: [] for pair in unique_pairs}
    for i, pair in enumerate(all_pairs):
        pair_idx2img_embs[pair].append(embeddings[i])

    selected_embs, selected_all_pairs = [], []
    for pair in unique_pairs:
        pair_reps = pair_idx2img_embs[pair]
        k = min(n, len(pair_reps))
        sampled_reps = random.sample(pair_reps, k)
        selected_embs       += sampled_reps
        selected_all_pairs  += [pair] * k
    selected_embs = torch.stack(selected_embs)
    return selected_embs, selected_all_pairs


def compute_logits(image_embs, label_embs):
    logit_scale = exp(0.07)
    logit_scale = logit_scale if logit_scale <= 100.0 else 100.0
    logits = logit_scale * image_embs @ label_embs.t()
    return logits.to('cpu')


def compute_weights(embs_for_IW, all_pairs_IW, train_dataset, use_clip_score=False, temperature=0.01, probs_type='clip'):
    device = embs_for_IW.device
    if len(set(all_pairs_IW)) == len(all_pairs_IW):
        weights = None
    else:
        if use_clip_score:
            text_embs = train_dataset.load_text_embs(all_pairs_IW)
            logits = torch.sum(embs_for_IW * text_embs, dim=1)
            T = temperature
            if probs_type == 'clip':
                weights = torch.exp(logits / float(T))
            elif probs_type == 'SigLIP':
                logit_bias = -16.54513931274414
                weights = torch.sigmoid(logits / float(T) + logit_bias)
        else:
            weights = torch.ones(len(all_pairs_IW)).float().to(device)

        _, inverse = np.unique(all_pairs_IW, axis=0, return_inverse=True)
        inverse = torch.LongTensor(inverse).to(device)
        group_sums = torch.bincount(inverse, weights=weights).float()
        weights /= group_sums[inverse]
    return weights


def main(config: argparse.Namespace, verbose=False):
    if config.experiment_name != 'clip' and config.modality_IW is None:
        raise Exception("Argument --modality_IW is required when --experiment_name!='clip'.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    dataset_path = DATASET_PATHS[config.dataset]
    model_info = {
        'model_architecture': config.model_architecture,
        'model_pretraining':  config.model_pretraining
    }

    test_dataset  = CompositionDatasetEmbeddings(dataset_path, phase=config.test_phase,
                                                 open_world=config.open_world, **model_info)
    train_dataset = CompositionDatasetEmbeddings(dataset_path, phase='train', **model_info)

    if verbose:
        scenario = 'open world' if config.open_world else 'closed world'
        print(f"Experiment name : {config.experiment_name}   IW modality : {config.modality_IW}    Scenario : {scenario}")
        print(f'Running on      : {device}')
        print(f"Dataset         : {config.dataset}")
        print(test_dataset)

    all_results = []
    for e in range(config.n_exp):
        if verbose:
            print(f'Running experiment {e+1}/{config.n_exp}')

        if config.experiment_name == 'clip':
            test_pair_embs = test_dataset.load_text_embs(test_dataset.pairs)
        else:
            # 1) Prepare embeddings for ideal word computation
            if config.modality_IW == 'text':
                all_pairs_IW = train_dataset.full_pairs
                embs_for_IW  = train_dataset.load_text_embs(all_pairs_IW)
            elif config.modality_IW == 'valid text':
                all_pairs_IW = train_dataset.train_pairs
                embs_for_IW  = train_dataset.load_text_embs(all_pairs_IW)
            elif config.modality_IW == 'image':
                embs_for_IW, all_pairs_IW = train_dataset.load_all_image_embs()
                if config.n_images is not None:
                    embs_for_IW, all_pairs_IW = select_n_embs_per_pair(
                        embs_for_IW, all_pairs_IW, n=config.n_images
                    )

            # 2) Compute noise distribution
            if 'CW' in config.experiment_name:
                name, _, T = config.experiment_name.split('_')
                probs_type = 'SigLIP' if 'SigLIP' in config.model_architecture else 'clip'
                weights = compute_weights(embs_for_IW, all_pairs_IW, train_dataset,
                                          use_clip_score=True, temperature=T, probs_type=probs_type)
            else:
                name = config.experiment_name
                weights = compute_weights(embs_for_IW, all_pairs_IW, train_dataset,
                                          use_clip_score=False)

            # 3) Build factorizer
            Factorizer = FACTORIZERS[name]
            factorizer = Factorizer(embs_for_IW, all_pairs_IW, weights)

            # 4) Compute pair / individual representations
            if config.sequential or config.top_k:
                attr_emb, obj_emb = factorizer.compute_ideal_words_approximation_sequential()
            else:
                test_pair_embs = factorizer.compute_ideal_words_approximation(
                    target_pairs=test_dataset.pairs
                )

        # Compute predictions
        image_embs, all_pairs_true = test_dataset.load_all_image_embs()
        image_embs = image_embs.to(device)

        evaluator = Evaluator(test_dataset)

        if config.sequential:
            scores_attr = compute_logits(image_embs, attr_emb.to(device))
            scores_obj  = compute_logits(image_embs, obj_emb.to(device))

            attr_f2d = torch.LongTensor([test_dataset.attr2idx[a] for a in factorizer.attrs]).to(device)
            obj_f2d  = torch.LongTensor([test_dataset.obj2idx[o]  for o in factorizer.objs ]).to(device)

            result = evaluator.get_sequential_metrics(
                scores_attr, scores_obj, all_pairs_true,
                attr_f2d=attr_f2d, obj_f2d=obj_f2d
            )
        elif config.top_k:
            scores_attr = compute_logits(image_embs, attr_emb.to(device))
            scores_obj  = compute_logits(image_embs, obj_emb.to(device))

            attr_f2d = torch.LongTensor([test_dataset.attr2idx[a] for a in factorizer.attrs]).to(device)
            obj_f2d  = torch.LongTensor([test_dataset.obj2idx[o]  for o in factorizer.objs ]).to(device)

            result = evaluator.get_top_k_metrics(
                image_embs,
                scores_attr, scores_obj,
                all_pairs_true,
                attr_f2d=attr_f2d, obj_f2d=obj_f2d,
                factorizer=factorizer,
                pick_top_k=config.pick_top_k,
            )
        else:
            test_pair_embs = test_pair_embs.to(device)
            logits = compute_logits(image_embs, test_pair_embs)
            result = evaluator.get_overall_metrics(logits, all_pairs_true, progress_bar=False)[1]

        all_results.append(result)

    # Combine results of multiple experiments
    if config.n_exp > 1:
        all_stats = list(all_results[0].keys())
        result = defaultdict(list)
        for res in all_results:
            for stat in all_stats:
                result[stat + ' (list)'].append(res[stat])
        for stat in all_stats:
            result[stat + ' (mean)'] = np.mean(result[stat + ' (list)'])
            result[stat + ' (std)']  = np.std(result[stat  + ' (list)'])
        result = dict(result)

    if config.result_path is not None:
        with open(config.result_path, 'w+') as fp:
            json.dump({'config': vars(config), 'result': result}, fp, indent=4)

    if verbose:
        to_float = lambda v: [float(x) for x in v] if isinstance(v, list) else float(v)
        print({k: to_float(v) for k, v in result.items()})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",            type=str, required=True)
    parser.add_argument("--model_architecture", type=str, default="ViT-L-14")
    parser.add_argument("--model_pretraining",  type=str, default="openai")
    parser.add_argument("--experiment_name",    type=str)
    parser.add_argument("--modality_IW",        type=str, default=None,
                        choices=['text', 'valid text', 'image'])
    parser.add_argument("--n_images",           type=int, default=None)
    parser.add_argument("--open_world",         action="store_true")
    parser.add_argument("--n_exp",              type=int, default=1)
    parser.add_argument("--test_phase",         type=str, default="test")
    parser.add_argument("--result_path",        type=str, default=None)
    parser.add_argument("--sequential",         action="store_true",
                        help="score attr and obj independently, then combine")
    parser.add_argument("--top_k",              action="store_true",
                        help="sequential top-k followed by GDE reranking")
    parser.add_argument("--pick_top_k",         type=int, default=20)

    config = parser.parse_args()
    main(config, verbose=True)
