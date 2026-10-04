"""Fixed validation-only comparison of center crops against a saved baseline."""

import csv
import json
import uuid
from pathlib import Path

from .patch_classifier import (
    CLASSES,
    checkpoint_crop,
    dataset_info,
    file_hash,
    plot_history,
    train,
    validate_crop,
    write_json,
)


def best_record(run):
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    history = json.loads((run / 'history.json').read_text(encoding='utf-8'))
    if result['status'] != 'completed' or len(history) != config['epochs']:
        raise ValueError('Comparison needs a completed baseline')
    best = max(history, key=lambda r: r['val']['macro_f1'])
    return config, {'run_dir': str(run), 'input_size': config.get('center_crop') or config['size'],
                    'selected_epoch': best['epoch'], 'metrics': best['val'],
                    'checkpoint_sha256': file_hash(run / 'best_model.pt')}


def plot_comparison(root, records):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = [str(r['input_size']) for r in records]
    axes[0].bar(names, [r['metrics']['macro_f1'] for r in records])
    axes[0].set_title('Best validation macro F1')
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel('Input pixels')
    for record in records:
        history = json.loads((Path(record['run_dir']) / 'history.json').read_text(encoding='utf-8'))
        axes[1].plot([r['epoch'] for r in history], [r['val']['macro_f1'] for r in history], label=record['input_size'])
        axes[2].plot(CLASSES, [record['metrics']['per_class'][c]['f1'] for c in CLASSES], marker='o', label=record['input_size'])
    axes[1].set_title('Validation macro F1 by epoch')
    axes[1].set_xlabel('Epoch')
    axes[2].set_title('Validation F1 per class')
    for axis in axes[1:]:
        axis.set_ylim(0, 1)
        axis.legend(title='Input pixels')
        axis.grid(alpha=.3)
    figure.tight_layout()
    figure.savefig(root / 'comparison.png', dpi=150)
    plt.close(figure)
    return root / 'comparison.png'


def compare_centers(dataset_root, baseline_run, output_root, *, crops=(64, 32), device='cuda'):
    baseline_run, output_root = Path(baseline_run).resolve(), Path(output_root).resolve()
    config, baseline = best_record(baseline_run)
    if checkpoint_crop(config) is not None:
        raise ValueError('Reference must use the full source patch')
    info = dataset_info(Path(dataset_root), config['size'], config['balanced_train'])
    if (info['manifest_sha256'] != config['manifest_sha256']
            or info['train_csv_sha256'] != config['train_csv_sha256']
            or info['wafer_groups'] != config['wafer_groups'] or info['counts'] != config['counts']):
        raise ValueError('Baseline and candidates must use identical data and splits')
    if config['device'] != device:
        raise ValueError('Comparison device must match baseline')
    import torch

    if config['torch'] != str(torch.__version__):
        raise ValueError('Comparison PyTorch version must match baseline')
    if not crops or len(set(crops)) != len(crops):
        raise ValueError('Specify unique center sizes before running')
    for crop in crops:
        validate_crop(config['size'], crop)
        if crop is None or crop >= config['size']:
            raise ValueError('Candidates must use smaller center inputs')
    if output_root == info['root'] or info['root'] in output_root.parents:
        raise ValueError('Comparison output must be outside the dataset')
    output = output_root / ('center_comparison_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    preserved = {name: file_hash(baseline_run / name) for name in ('best_model.pt', 'config.json', 'history.json', 'result.json')}
    plan = {'schema_version': 1, 'task': 'fixed_center_input_comparison', 'research_only': True,
            'input_sizes': [config['size'], *crops], 'manifest_sha256': info['manifest_sha256'],
            'train_csv_sha256': info['train_csv_sha256'], 'baseline_run': str(baseline_run),
            'baseline_hashes': preserved, 'device': device, 'torch': str(torch.__version__),
            'seed': config['seed'], 'epochs': config['epochs'], 'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'], 'dataset_review_status': info['review_status'],
            'class_weighting': config.get('class_weighting', 'none'),
            'selection_split': 'val', 'test_evaluated': False,
            'interpretation': 'Exploratory comparison on reused validation wafer; no independent generalization claim',
            'deterministic_cuda_guaranteed': False}
    write_json(output / 'plan.json', plan)
    records = [baseline]
    try:
        for crop in crops:
            print(f'Comparing center {crop} pixels; all other settings fixed', flush=True)
            result = train(info['root'], output, epochs=config['epochs'], batch_size=config['batch_size'],
                           learning_rate=config['learning_rate'], size=config['size'], balanced=config['balanced_train'],
                           seed=config['seed'], device=device, center_crop=crop,
                           class_weighting=config.get('class_weighting', 'none'))
            run = Path(result['run_dir'])
            plot_history(run)
            _, record = best_record(run)
            records.append(record)
            write_json(output / 'progress.json', records)
        if any(file_hash(baseline_run / name) != expected for name, expected in preserved.items()):
            raise ValueError('Baseline artifacts changed during comparison')
        best = max(records, key=lambda r: r['metrics']['macro_f1'])
        summary = dict(plan, status='completed', records=records, best_input_size=best['input_size'],
                       output_dir=str(output), improvement=best['metrics']['macro_f1']-baseline['metrics']['macro_f1'])
        write_json(output / 'comparison.json', summary)
        with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            fields = ['input_size', 'selected_epoch', 'macro_f1', 'accuracy', *[f'{c}_f1' for c in CLASSES], 'run_dir']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({'input_size': record['input_size'], 'selected_epoch': record['selected_epoch'],
                                 'macro_f1': record['metrics']['macro_f1'], 'accuracy': record['metrics']['accuracy'],
                                 'run_dir': record['run_dir'],
                                 **{f'{c}_f1': record['metrics']['per_class'][c]['f1'] for c in CLASSES}})
        plot_comparison(output, records)
        write_json(output / 'status.json', {'status': 'completed', 'test_evaluated': False})
        return summary
    except BaseException as error:
        write_json(output / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                           'error_type': type(error).__name__, 'completed_candidates': len(records)-1})
        raise
