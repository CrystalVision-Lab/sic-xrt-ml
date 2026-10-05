"""Recorded follow-up on development domain shifts; retains the first comparison."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .grouped_development import (
    WAFERS,
    DevelopmentImages,
    fit,
    load_development_model,
    predictions,
    split_wafer,
    verify_cohort,
)
from .patch_classifier import classification_metrics, file_hash, to_tensor, write_json

RECIPES = (
    {'name': 'normalized_center_sqrt', 'variant': 'center32', 'sampling_power': .5,
     'normalization': 'local_contrast_v1'},
    {'name': 'normalized_context_sqrt', 'variant': 'center32_context128', 'sampling_power': .5,
     'normalization': 'local_contrast_v1'},
)


def run_followup(root, previous, output, epochs=12, seed=42, device='cuda'):
    root, previous, output = (Path(p).resolve() for p in (root, previous, output))
    if output.exists() or output.is_relative_to(root):
        raise ValueError('New output outside cohort required')
    contract, rows = verify_cohort(root)
    first = json.loads((previous/'result.json').read_text(encoding='utf-8'))
    first_plan = json.loads((previous/'plan.json').read_text(encoding='utf-8'))
    if (first['status'] != 'completed' or first_plan['manifest_sha256'] != contract['manifest_sha256']
            or first_plan['epochs'] != epochs or first_plan['seed'] != seed):
        raise ValueError('First development comparison is incompatible')
    output.mkdir(parents=True)
    write_json(output/'plan.json', {'schema_version': 1, 'task': 'contrast_normalization_development_followup',
        'created_at': datetime.now(UTC).isoformat(), 'prior_result_sha256': file_hash(previous/'result.json'),
        'manifest_sha256': contract['manifest_sha256'], 'recipes': list(RECIPES), 'epochs': epochs, 'seed': seed,
        'reason': 'First development folds exposed cross-wafer/annealing appearance failures; investigate contrast invariance',
        'selection': 'maximum_pooled_OOF_macro_F1_across_original_and_two_new_recipes',
        'adaptive_development_experiment': True, 'independent_final_test': False, 'test_evaluated': False})
    cache = DevelopmentImages(root, rows)
    results = list(first['comparison'])
    for recipe in RECIPES:
        folds, oof = [], []
        for heldout in WAFERS:
            train, valid = split_wafer(rows, heldout)
            run = output/f'{recipe["name"]}_holdout{heldout}'
            model = fit(cache, train, recipe, contract, run, epochs, seed, device, heldout)
            pred, metrics = predictions(model, cache, valid, device)
            write_json(run/'heldout_predictions.json', pred)
            folds.append({'wafer': heldout, 'metrics': metrics, 'checkpoint_sha256': file_hash(run/'best_model.pt')})
            oof.extend(pred)
            write_json(output/'progress.json', {'recipe': recipe, 'completed_folds': folds, 'test_evaluated': False})
            print(f'FOLD DONE {recipe["name"]} W{heldout}: {json.dumps(metrics)}', flush=True)
            del model
        if len(oof) != len(rows) or len({p['patch_id'] for p in oof}) != len(rows):
            raise ValueError('OOF coverage failure')
        item = {'recipe': recipe, 'folds': folds, 'pooled_OOF': classification_metrics(
            np.sum([f['metrics']['confusion_matrix'] for f in folds], axis=0))}
        results.append(item)
        write_json(output/f'{recipe["name"]}_oof.json', oof)
        write_json(output/'comparison.json', results)
    selected = max(results, key=lambda r: r['pooled_OOF']['macro_f1'])
    final_run = output/'development_model'
    model = fit(cache, list(range(len(rows))), selected['recipe'], contract, final_run, epochs, seed, device, None)
    reloaded, _ = load_development_model(final_run, device)
    probe = torch.stack([to_tensor(cache.images[i]) for i in range(3)]).to(device)
    model.eval()
    with torch.inference_mode():
        if not torch.allclose(model(probe), reloaded(probe), atol=1e-6):
            raise ValueError('Reloaded checkpoint differs')
    verify_cohort(root)
    result = {'status': 'completed', 'output_dir': str(output), 'previous_output': str(previous),
              'comparison': results, 'selected_recipe': selected['recipe'], 'selected_OOF': selected['pooled_OOF'],
              'model_run': str(final_run), 'checkpoint_sha256': file_hash(final_run/'best_model.pt'),
              'model_loader': 'sic_xrt_ml.training.grouped_development.load_development_model',
              'serialization_verified': True, 'adaptive_development_experiment': True,
              'test_evaluated': False, 'independent_final_test': False, 'research_only': True,
              'expert_ground_truth': False, 'limitations': first['limitations']}
    write_json(output/'result.json', result)
    return result
