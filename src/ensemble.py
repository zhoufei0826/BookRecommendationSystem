"""Member C: compare and blend B's ItemKNN with ALS on a shared split.

python src/ensemble.py validate --out-dir outputs/c_fusion/seed42
python src/ensemble.py submit --config reports/c_selected_model.json

All candidates use the complete training item catalog. Only score batches are
dense; no full users-by-items prediction matrix is retained. Validation selects
a configuration; submit requires that saved configuration and refits on full data.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate, validate_recommendations
from implicit_als import ImplicitALS, _targets_and_history
from item_knn import compute_similarity, truncate_neighbors


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def standardize(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), 1e-8)


def weighted_history(matrix, popularity, beta):
    """B's pop_i**(-beta) history weights, normalized separately per user."""
    history = matrix.astype(np.float64).multiply(
        np.maximum(popularity, 1).astype(np.float64) ** (-beta)).tocsr()
    if beta > 0:
        totals = np.asarray(history.sum(axis=1)).ravel()
        history = history.multiply((1 / np.maximum(totals, 1e-12))[:, None]).tocsr()
    return history


def top_items(scores, seen, popularity, top_k, positive_only=False):
    """Rank after seen filtering. Empty KNN candidates use unseen popularity."""
    scores = np.asarray(scores).copy()
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite model scores')
    scores[seen] = -1.0 if positive_only else -np.inf
    valid = scores > 0 if positive_only else np.isfinite(scores)
    count = min(top_k, int(valid.sum()))
    if count == top_k:
        # Match B's selection procedure, including its treatment of equal scores.
        selected = np.argpartition(scores, -top_k)[-top_k:]
        selected = selected[np.argsort(scores[selected])[::-1]].tolist()
    elif count:
        ordered = np.argsort(scores)[::-1]
        selected = ordered[valid[ordered]][:count].tolist()
    else:
        selected = []
    if len(selected) < top_k:
        excluded = set(seen) | set(selected)
        for item in np.lexsort((np.arange(len(popularity)), -popularity)):
            if item not in excluded:
                selected.append(int(item))
                excluded.add(int(item))
            if len(selected) == top_k:
                break
    if len(selected) != top_k:
        raise ValueError('Insufficient unseen items for requested Top-K')
    return np.asarray(selected, dtype=np.int32)


def make_predictions(model, neighbors, users, blends, beta=0.5, top_k=10, batch_size=256):
    users = np.asarray(users, dtype=np.int32)
    positions = np.searchsorted(model.user_ids, users)
    if np.any(positions >= len(model.user_ids)) or np.any(model.user_ids[positions] != users):
        raise ValueError('Requested users must occur in the training data')
    for weight, pop_weight in blends.values():
        if not (0 <= weight <= 1 and 0 <= pop_weight <= 1):
            raise ValueError('Blend weights must be between 0 and 1')
    history = weighted_history(model.matrix, model.popularity, beta)
    pop_scores = standardize(np.log1p(model.popularity)[None, :])
    predictions = {name: np.empty((len(users), top_k), dtype=np.int32) for name in blends}
    for start in range(0, len(users), batch_size):
        stop = min(start + batch_size, len(users))
        rows = positions[start:stop]
        knn_raw = (history[rows] @ neighbors).toarray()
        knn_z = standardize(knn_raw)
        als_z = standardize(model.user_factors[rows] @ model.item_factors.T)
        for name, (als_weight, pop_weight) in blends.items():
            als_part = (1 - pop_weight) * als_z + pop_weight * pop_scores
            scores = (1 - als_weight) * knn_z + als_weight * als_part
            if als_weight == 0:
                scores = knn_raw
            for local, row in enumerate(rows):
                seen = model.matrix.indices[model.matrix.indptr[row]:model.matrix.indptr[row + 1]]
                chosen = top_items(scores[local], seen, model.popularity, top_k,
                                   positive_only=(als_weight == 0))
                predictions[name][start + local] = model.item_ids[chosen]
        print(f'Fusion ranking {stop}/{len(users)}', flush=True) if stop % 8192 < batch_size or stop == len(users) else None
    return predictions


def as_dict(users, items):
    return {int(uid): row.astype(int).tolist() for uid, row in zip(users, items)}


def write_recs(path, users, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'user_id': np.repeat(users, items.shape[1]),
                  'item_id': items.ravel()}).to_csv(path, index=False)


def compare_predictions(users, baseline, selected, targets, history):
    """Paired user differences and activity strata defined using training only."""
    activity = np.array([len(history[int(uid)]) for uid in users])
    lower, upper = np.quantile(activity, [1 / 3, 2 / 3])
    masks = {'low': activity <= lower, 'middle': (activity > lower) & (activity <= upper),
             'high': activity > upper}
    base_hits = np.array([len(set(row) & targets[int(uid)]) for uid, row in zip(users, baseline)])
    best_hits = np.array([len(set(row) & targets[int(uid)]) for uid, row in zip(users, selected)])
    overlaps = [len(set(a) & set(b)) for a, b in zip(baseline, selected)]
    rows = []
    for name, mask in masks.items():
        count = int(mask.sum())
        actual = sum(len(targets[int(uid)]) for uid in users[mask])
        denom = count * baseline.shape[1] + actual
        rows.append({'group': name, 'users': count,
                     'knn_micro_f1': 2 * int(base_hits[mask].sum()) / denom if denom else 0,
                     'selected_micro_f1': 2 * int(best_hits[mask].sum()) / denom if denom else 0})
    return {'activity_cutoffs': [float(lower), float(upper)], 'activity_groups': rows,
            'additional_hits': int((best_hits - base_hits).sum()),
            'users_improved': int((best_hits > base_hits).sum()),
            'users_worse': int((best_hits < base_hits).sum()),
            'users_equal': int((best_hits == base_hits).sum()),
            'mean_top10_overlap_with_knn': float(np.mean(overlaps))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'submit'])
    parser.add_argument('--config', help='Frozen JSON configuration; required for submission')
    parser.add_argument('--train-path')
    parser.add_argument('--val-path', default='outputs/val_split.csv')
    parser.add_argument('--out-dir', default='outputs/c_fusion/seed42')
    parser.add_argument('--output', default='data/submission_fusion.csv')
    parser.add_argument('--selected-config-out', default='reports/c_selected_model.json')
    args = parser.parse_args()
    if args.mode == 'submit' and not args.config:
        parser.error('submit requires --config chosen on validation data')
    config = {'als': {'factors': 128, 'iterations': 12, 'alpha': 40.0,
                      'regularization': 0.1, 'cg_steps': 3, 'seed': 42},
              'knn': {'neighbors': 100, 'beta': 0.5}, 'top_k': 10}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    frozen = args.config is not None
    train_path = args.train_path or ('outputs/train_split.csv' if args.mode == 'validate' else 'data/train_dataset.csv')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    train = pd.read_csv(train_path, dtype={'user_id': 'int32', 'item_id': 'int32'})
    model = ImplicitALS(**config['als']).fit(train)
    print('Building B-001 cosine neighbors', flush=True)
    similarity = compute_similarity(model.matrix, 'cosine')
    neighbors = truncate_neighbors(similarity, config['knn']['neighbors'])
    del similarity
    train_seconds = time.perf_counter() - clock
    metadata = {'train_path': train_path, 'train_sha256': file_hash(train_path),
                'configuration': config, 'train_seconds': train_seconds,
                'code_sha256': {f: file_hash(Path('src') / f) for f in
                                ['ensemble.py', 'implicit_als.py', 'item_knn.py', 'evaluate.py']}}

    if args.mode == 'validate':
        validation = pd.read_csv(args.val_path, dtype={'user_id': 'int32', 'item_id': 'int32'})
        targets, seen = _targets_and_history(model, validation)
        if any(items & seen.get(uid, set()) for uid, items in targets.items()):
            raise ValueError('Training and validation interactions overlap')
        users = np.array(sorted(targets), dtype=np.int32)
        blends = {'knn': (0.0, 0.0), 'als': (1.0, 0.0), 'als_pop': (1.0, 0.25)}
        if frozen:
            blends['selected'] = (config['blend']['als_weight'], config['blend']['als_pop_weight'])
        else:
            for weight in [0.1, 0.25, 0.5]:
                blends[f'mix_als_{weight}'] = (weight, 0.0)
            blends['mix_als_pop_0.25'] = (0.25, 0.25)
        predictions = make_predictions(model, neighbors, users, blends, config['knn']['beta'], config['top_k'])
        results = {}
        for name, items in predictions.items():
            metrics = evaluate(as_dict(users, items), targets, top_k=config['top_k'], user_train_items=seen)
            metrics['coverage'] = float(len(np.unique(items)) / len(model.item_ids))
            metrics['als_weight'], metrics['als_pop_weight'] = blends[name]
            results[name] = metrics
            print(f'{name}: micro-F1={metrics["micro_f1"]:.6f}, macro-F1={metrics["macro_f1"]:.6f}', flush=True)
        selected = 'selected' if frozen else max(results, key=lambda label: results[label]['micro_f1'])
        if not frozen:
            config['blend'] = {'als_weight': blends[selected][0], 'als_pop_weight': blends[selected][1]}
            config['selected_on'] = {'train_sha256': metadata['train_sha256'],
                                     'val_sha256': file_hash(args.val_path),
                                     'label': selected, 'micro_f1': results[selected]['micro_f1']}
            save_json(args.selected_config_out, config)
        metadata.update({'val_path': args.val_path, 'val_sha256': file_hash(args.val_path),
                         'configuration_locked': frozen, 'selected': selected,
                         'metrics': results, 'total_seconds': time.perf_counter() - clock,
                         'paired_analysis': compare_predictions(users, predictions['knn'], predictions[selected], targets, seen)})
        save_json(out / 'evaluation.json', metadata)
        np.savez_compressed(out / 'recommendations.npz', user_ids=users, **predictions)
        write_recs(out / 'selected_recs.csv', users, predictions[selected])
        print(f'Saved validation to {out}; selected={selected}', flush=True)
    else:
        users = pd.read_csv('data/test_dataset.csv', dtype={'user_id': 'int32'}).user_id.to_numpy()
        blend = config['blend']
        predictions = make_predictions(model, neighbors, users,
                                       {'selected': (blend['als_weight'], blend['als_pop_weight'])},
                                       config['knn']['beta'], config['top_k'])['selected']
        _, seen = _targets_and_history(model, train.iloc[:0])
        validate_recommendations(as_dict(users, predictions), users, config['top_k'], seen)
        write_recs(args.output, users, predictions)
        metadata.update({'output': args.output, 'output_sha256': file_hash(args.output),
                         'test_sha256': file_hash('data/test_dataset.csv'),
                         'rows': int(predictions.size), 'total_seconds': time.perf_counter() - clock})
        save_json(out / 'submission_info.json', metadata)
        print(f'Saved full-data submission candidate: {args.output}', flush=True)


if __name__ == '__main__':
    main()
