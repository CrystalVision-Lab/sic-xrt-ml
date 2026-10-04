"""Compare full inverse-frequency training against a preserved balanced baseline on fixed validation rows."""
import argparse
import csv
import hashlib
import json
import uuid
from pathlib import Path

import torch

from .center_comparison import best_record
from .patch_classifier import (
    CLASSES,
    checkpoint_crop,
    classification_metrics,
    dataset_info,
    file_hash,
    inverse_frequency_weights,
    plot_history,
    read_csv,
    train,
    write_json,
)
from .validation_review import generate_review


def validation_fingerprint(rows):
    content = json.dumps(sorted(rows, key=lambda r: r['patch_id']), sort_keys=True,
                         ensure_ascii=True, separators=(',', ':'))
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def checked_predictions(review, rows, checkpoint_sha, manifest_sha):
    meta = json.loads((review/'summary.json').read_text(encoding='utf-8'))
    if meta['split'] != 'val' or meta['test_evaluated'] is not False or \
            meta['checkpoint_sha256'] != checkpoint_sha or meta['manifest_sha256'] != manifest_sha:
        raise ValueError('Validation predictions do not match checkpoint and dataset')
    predictions = read_csv(review/'val_predictions.csv')
    canonical = {r['patch_id']: r for r in rows}
    by_id = {r['patch_id']: r for r in predictions}
    if len(predictions) != len(by_id) or set(by_id) != set(canonical) or \
            any(any(r.get(k) != value for k, value in canonical[ident].items()) or
                r.get('prediction') not in CLASSES for ident, r in by_id.items()):
        raise ValueError('Validation prediction rows or original targets changed')
    matrix = [[0]*len(CLASSES) for _ in CLASSES]
    for row in predictions:
        matrix[CLASSES.index(row['label'])][CLASSES.index(row['prediction'])] += 1
    if classification_metrics(matrix) != meta['metrics']:
        raise ValueError('Validation CSV and recorded metrics differ')
    return by_id, meta


def plot_comparison(output, records):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ['Balanced 438' if records[0]['training_samples'] == 438 else 'Balanced subset', 'Full weighted']
    axes[0].bar(names, [r['metrics']['macro_f1'] for r in records])
    axes[0].set_title('Best validation macro F1')
    axes[0].set_ylim(0, 1)
    for name, record in zip(names, records, strict=True):
        axes[1].plot(CLASSES, [record['metrics']['per_class'][c]['f1'] for c in CLASSES], marker='o', label=name)
    axes[1].set_title('Validation F1 per class')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)


def compare_full_training(dataset_root, baseline_run, baseline_review, output_root, *, device='cuda'):
    baseline_run, baseline_review, output_root = (Path(p).resolve() for p in
                                                 (baseline_run, baseline_review, output_root))
    config, baseline = best_record(baseline_run)
    if checkpoint_crop(config) is not None or not config['balanced_train'] or config.get('class_weighting', 'none') != 'none':
        raise ValueError('Baseline must be unweighted balanced training with full patch input')
    info = dataset_info(Path(dataset_root), config['size'], balanced=False)
    reference = dataset_info(Path(dataset_root), config['size'], balanced=True)
    if info['manifest_sha256'] != config['manifest_sha256'] or \
            reference['train_csv_sha256'] != config['train_csv_sha256'] or \
            reference['counts'] != config['counts'] or info['wafer_groups'] != config['wafer_groups']:
        raise ValueError('Baseline and candidate dataset contracts differ')
    if config['device'] != device or config['torch'] != str(torch.__version__):
        raise ValueError('Comparison device and PyTorch version must match the baseline')
    if any(output_root == p or output_root.is_relative_to(p) for p in (info['root'], baseline_run, baseline_review)):
        raise ValueError('Comparison outputs must be outside the dataset and baseline')
    checkpoint = torch.load(baseline_run/'best_model.pt', map_location='cpu', weights_only=True)
    if checkpoint['config'] != config or checkpoint['epoch'] != baseline['selected_epoch'] or \
            checkpoint['validation'] != baseline['metrics']:
        raise ValueError('Baseline checkpoint and best validation record differ')
    prior, prior_meta = checked_predictions(baseline_review, info['rows']['val'],
                                           baseline['checkpoint_sha256'], info['manifest_sha256'])
    if prior_meta['selected_epoch'] != baseline['selected_epoch'] or \
            prior_meta['metrics']['confusion_matrix'] != baseline['metrics']['confusion_matrix'] or \
            validation_fingerprint(reference['rows']['val']) != validation_fingerprint(info['rows']['val']):
        raise ValueError('Baseline validation data or predictions differ')
    protected = {str(p): file_hash(p) for p in baseline_run.rglob('*') if p.is_file()}
    for name in ('summary.json', 'validation.json', 'provenance.json', '기록/samples.csv', '기록/train_balanced_128.csv'):
        p = info['root']/name
        if p.exists():
            protected[str(p)] = file_hash(p)
    weights = inverse_frequency_weights(info['rows']['train'])
    output = output_root/('full_training_comparison_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    plan = {'schema_version': 1, 'task': 'fixed_validation_full_training_comparison',
        'research_only': True, 'baseline_run': str(baseline_run), 'baseline_review': str(baseline_review),
        'manifest_sha256': info['manifest_sha256'], 'validation_rows_sha256': validation_fingerprint(info['rows']['val']),
        'class_weighting': 'inverse_frequency', 'class_weights': weights,
        'class_weight_basis': 'all_training_rows_only', 'training_counts': info['counts']['train'],
        'wafer_groups': info['wafer_groups'], 'validation_samples': len(info['rows']['val']),
        'epochs': config['epochs'], 'seed': config['seed'], 'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'], 'size': config['size'],
        'device': device, 'torch': str(torch.__version__), 'validation_loss_weighted': False,
        'test_evaluated': False, 'labels_changed': False, 'coordinates_changed': False,
        'protected_artifact_hashes': protected,
        'interpretation': 'Combined change in data amount, loss weighting and optimizer update count; reused validation wafer, not independent generalization',
        'deterministic_cuda_guaranteed': False}
    write_json(output/'plan.json', plan)
    try:
        result = train(info['root'], output/'runs', epochs=config['epochs'], batch_size=config['batch_size'],
                       learning_rate=config['learning_rate'], size=config['size'], balanced=False,
                       seed=config['seed'], device=device, class_weighting='inverse_frequency')
        run = Path(result['run_dir'])
        fig = plot_history(run)
        import matplotlib.pyplot as plt
        plt.close(fig)
        review = generate_review(run, info['root'], device=device, batch_size=config['batch_size'])
        candidate_config, candidate = best_record(run)
        current, current_meta = checked_predictions(review, info['rows']['val'],
                                                   candidate['checkpoint_sha256'], info['manifest_sha256'])
        if current_meta['selected_epoch'] != candidate['selected_epoch'] or \
                current_meta['metrics']['confusion_matrix'] != candidate['metrics']['confusion_matrix']:
            raise ValueError('Candidate checkpoint and predictions differ')
        history = json.loads((run/'history.json').read_text(encoding='utf-8'))
        if any(row['train']['samples'] != len(info['rows']['train']) or
               row['val']['samples'] != len(info['rows']['val']) for row in history):
            raise ValueError('Not all fixed training and validation rows were used each epoch')
        baseline.update(strategy='balanced_subset', training_samples=len(reference['rows']['train']),
                        optimizer_steps_per_epoch=(len(reference['rows']['train'])+config['batch_size']-1)//config['batch_size'])
        candidate.update(strategy='full_inverse_frequency', training_samples=len(info['rows']['train']),
                         optimizer_steps_per_epoch=candidate_config['optimizer_steps_per_epoch'],
                         validation_review=str(review))
        records = [baseline, candidate]
        paired = []
        for ident in sorted(prior):
            old, new = prior[ident], current[ident]
            paired.append({'patch_id': ident, 'label': old['label'], 'baseline_prediction': old['prediction'],
                'full_weighted_prediction': new['prediction'], 'baseline_correct': old['label']==old['prediction'],
                'full_weighted_correct': new['label']==new['prediction']})
        with (output/'paired_validation_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
        if any(file_hash(Path(p)) != sha for p, sha in protected.items()):
            raise ValueError('Dataset or baseline artifacts changed during comparison')
        change = candidate['metrics']['macro_f1']-baseline['metrics']['macro_f1']
        summary = plan | {'status': 'completed', 'output_dir': str(output), 'records': records,
            'macro_f1_change': change, 'full_weighted_improved': change > 0,
            'old_TSD_to_TED': baseline['metrics']['confusion_matrix'][2][1],
            'new_TSD_to_TED': candidate['metrics']['confusion_matrix'][2][1],
            'paired_improved': sum(not p['baseline_correct'] and p['full_weighted_correct'] for p in paired),
            'paired_regressed': sum(p['baseline_correct'] and not p['full_weighted_correct'] for p in paired),
            'validation_labels_preserved': True, 'baseline_hashes_preserved': True,
            'test_evaluated': False, 'automatic_model_promotion': False}
        write_json(output/'comparison.json', summary)
        with (output/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            fields = ['strategy', 'training_samples', 'selected_epoch', 'macro_f1', 'accuracy',
                      *[c+'_f1' for c in CLASSES], 'run_dir']
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for r in records:
                writer.writerow({k:r[k] for k in ('strategy','training_samples','selected_epoch','run_dir')} |
                    {k:r['metrics'][k] for k in ('macro_f1','accuracy')} |
                    {c+'_f1':r['metrics']['per_class'][c]['f1'] for c in CLASSES})
        plot_comparison(output, records)
        return summary
    except BaseException as exc:
        write_json(output/'comparison.json', plan | {'status':'failed', 'error_type':type(exc).__name__, 'error':str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'baseline', 'baseline-review', 'outputs'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = parser.parse_args()
    print(json.dumps(compare_full_training(args.dataset,args.baseline,args.baseline_review,args.outputs,
                                          device=args.device), ensure_ascii=False))


if __name__ == '__main__':
    main()
