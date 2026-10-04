"""Read-only train/val input diagnostics and a predeclared center-input experiment."""

import argparse
import csv
import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tifffile
import torch

from .center_comparison import best_record, compare_centers
from .full_training_comparison import checked_predictions, validation_fingerprint
from .patch_classifier import (
    CLASSES,
    checkpoint_crop,
    classification_metrics,
    dataset_info,
    file_hash,
    inverse_frequency_weights,
    to_tensor,
    write_json,
)
from .validation_review import generate_review


def pixel_features(image):
    """Unit-range descriptors of supplied pixels, never physical type/position truth."""
    gray = to_tensor(image).numpy().mean(0)
    size = gray.shape[0]
    if gray.shape != (size, size) or size < 16:
        raise ValueError('Diagnostics need square inputs >=16 pixels')
    width = min(16, size//2)
    start = (size-width)//2
    mask = np.ones((size, size), dtype=bool)
    mask[start:start+width, start:start+width] = False
    values = gray
    background = values[mask]
    core = values[start:start+width, start:start+width]
    smooth = sum(gray[dy:dy+size-2,dx:dx+size-2] for dy in range(3) for dx in range(3))/9
    peak_y, peak_x = np.unravel_index(np.argmax(smooth), smooth.shape)
    distance = np.hypot(peak_x+1-(size-1)/2, peak_y+1-(size-1)/2)
    return {'background_median': float(np.median(background)),
            'background_std': float(np.std(background)),
            'center_q99_minus_background_q99': float(np.quantile(core,.99)-np.quantile(background,.99)),
            'center_mean_minus_background_median': float(core.mean()-np.median(background)),
            'brightest_3x3_distance_pixels': float(distance)}


def grouped_counts(rows):
    groups = defaultdict(Counter)
    for r in rows:
        groups[(r['split'],r['wafer'],r.get('phase','unknown'),r.get('source_id','unknown'))][r['label']] += 1
    return [{'split':k[0],'wafer':k[1],'phase':k[2],'source_id':k[3],
             'counts':dict(counts)} for k, counts in sorted(groups.items())]


def aggregate_features(features):
    groups = defaultdict(list)
    for row in features:
        groups[(row['split'],row['label'],row['fine_label'],row['phase'])].append(row)
    names = ('background_median','background_std','center_q99_minus_background_q99',
             'center_mean_minus_background_median','brightest_3x3_distance_pixels')
    return [{'split':key[0],'label':key[1],'fine_label':key[2],'phase':key[3],'samples':len(rows),
             'features':{name:dict(zip(('q25','median','q75'),
                 map(float,np.quantile([r[name] for r in rows],[.25,.5,.75])),strict=True)) for name in names}}
            for key, rows in sorted(groups.items())]


def subtype_metrics(predictions):
    groups = defaultdict(lambda: np.zeros((3,3),dtype=np.int64))
    for row in predictions.values():
        groups[row.get('provider_fine_label',row['label'])][CLASSES.index(row['label']),CLASSES.index(row['prediction'])] += 1
    return {name:classification_metrics(matrix) for name,matrix in sorted(groups.items())}


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_run(run, info, *, device):
    config, record = best_record(run)
    checkpoint_crop(config)
    if (config['manifest_sha256'] != info['manifest_sha256'] or
            config['train_csv_sha256'] != info['train_csv_sha256'] or
            config['counts'] != info['counts'] or config['wafer_groups'] != info['wafer_groups'] or
            config['device'] != device or config['torch'] != str(torch.__version__) or
            config['balanced_train'] or config.get('class_weighting') != 'inverse_frequency' or
            config.get('class_weights') != inverse_frequency_weights(info['rows']['train'])):
        raise ValueError('Completed run must match full weighted dataset and runtime')
    checkpoint = torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
    if (checkpoint['config'] != config or checkpoint['epoch'] != record['selected_epoch'] or
            checkpoint['validation'] != record['metrics']):
        raise ValueError('Checkpoint and best completed record differ')
    history = json.loads((run/'history.json').read_text(encoding='utf-8'))
    if any(r['train']['samples'] != len(info['rows']['train']) or
           r['val']['samples'] != len(info['rows']['val']) for r in history):
        raise ValueError('Incomplete fixed train/validation sample use')
    return config, record


def plot_features(output, features):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for axis, name in zip(axes, ('background_median','center_q99_minus_background_q99'),strict=True):
        groups = [[r[name] for r in features if r['split']==split and r['fine_label']=='TSD_a']
                  for split in ('train','val')]
        if all(groups):
            axis.boxplot(groups,showfliers=False)
            axis.set_xticks([1,2],['Train TSD_a','Val TSD_a'])
        axis.set_title(name.replace('_',' '))
        axis.set_ylabel('Unit-range pixel value')
        axis.grid(alpha=.3)
    fig.suptitle('Supplied TSD_a pixel descriptors; not physical type evidence')
    fig.tight_layout()
    fig.savefig(output/'pixel_descriptors.png',dpi=150)
    plt.close(fig)


def diagnose_and_compare(dataset, baseline, review, outputs, *, crops=(64,32), device='cuda'):
    dataset, baseline, review, outputs = (Path(p).resolve() for p in (dataset,baseline,review,outputs))
    config = json.loads((baseline/'config.json').read_text(encoding='utf-8'))
    info = dataset_info(dataset,config['size'],balanced=False)
    config, record = checked_run(baseline,info,device=device)
    if checkpoint_crop(config) is not None:
        raise ValueError('Baseline must use full source input')
    if any(outputs==p or outputs.is_relative_to(p) for p in (dataset,baseline,review)):
        raise ValueError('Output must be outside dataset and prior artifacts')
    prior, meta = checked_predictions(review,info['rows']['val'],record['checkpoint_sha256'],info['manifest_sha256'])
    if meta['selected_epoch'] != record['selected_epoch'] or meta['metrics']['confusion_matrix'] != record['metrics']['confusion_matrix']:
        raise ValueError('Prior predictions and baseline metrics differ')
    from .patch_classifier import validate_crop
    if not crops or len(set(crops)) != len(crops):
        raise ValueError('Predeclare unique crops')
    for crop in crops:
        validate_crop(config['size'],crop)
        if crop is None or crop >= config['size']:
            raise ValueError('Crops must be smaller than full input')
    protected = {str(p):file_hash(p) for folder in (baseline,review) for p in folder.rglob('*') if p.is_file()}
    for name in ('summary.json','validation.json','provenance.json','output_hashes.json','기록/samples.csv'):
        p = dataset/name
        if p.is_file():
            protected[str(p)] = file_hash(p)
    out = outputs/('tsd_diagnostics_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    plan = {'schema_version':1,'task':'tsd_distribution_and_center_diagnosis','research_only':True,
            'manifest_sha256':info['manifest_sha256'],'validation_rows_sha256':validation_fingerprint(info['rows']['val']),
            'crops':list(crops),'baseline_run':str(baseline),'baseline_review':str(review),
            'baseline_checkpoint_sha256':record['checkpoint_sha256'],'class_weights':config['class_weights'],
            'class_weighting':'inverse_frequency','train_samples':len(info['rows']['train']),
            'validation_samples':len(info['rows']['val']),'test_evaluated':False,
            'labels_changed':False,'coordinates_changed':False,'protected_hashes':protected,
            'limitations':['Provider labels are not expert ground truth','Reused validation; exploratory only',
                           'Pixel descriptors do not identify physical defect types',
                           'Center crop changes context and spatial resolution of CNN pooling',
                           'No phase/source causal isolation; no deterministic CUDA guarantee']}
    write_json(out/'plan.json',plan)
    try:
        features = []
        for split in ('train','val'):
            for row in info['rows'][split]:
                path = (dataset/row['path']).resolve()
                if not path.is_relative_to(dataset) or file_hash(path) != row['sha256']:
                    raise ValueError('Diagnostic patch integrity failed')
                image = tifffile.imread(path)
                if image.shape[:2] != (config['size'],config['size']) or str(image.dtype) != row['dtype']:
                    raise ValueError('Diagnostic pixel contract changed')
                features.append({'patch_id':row['patch_id'],'split':split,'wafer':row['wafer'],
                    'phase':row.get('phase','unknown'),'label':row['label'],
                    'fine_label':row.get('provider_fine_label',row['label']),
                    'prediction':prior[row['patch_id']]['prediction'] if split=='val' else '',
                    **pixel_features(image)})
        domains = grouped_counts(info['rows']['train']+info['rows']['val'])
        diagnostic = {'source_counts':domains,'feature_groups':aggregate_features(features),
                      'baseline_subtype_metrics':subtype_metrics(prior),'test_pixels_read':False}
        write_csv(out/'pixel_features.csv',features)
        write_json(out/'input_diagnostics.json',diagnostic)
        plot_features(out,features)
        print('Train/val descriptors saved; starting predeclared center comparisons',flush=True)
        centers = compare_centers(dataset,baseline,out/'center_runs',crops=crops,device=device)
        records = [record | {'validation_review':str(review),'subtype_metrics':subtype_metrics(prior)}]
        paired = {ident:{'patch_id':ident,'label':r['label'],'fine_label':r.get('provider_fine_label',r['label']),
                         'prediction_full':r['prediction']} for ident,r in prior.items()}
        for candidate in centers['records'][1:]:
            run = Path(candidate['run_dir'])
            _, verified = checked_run(run,info,device=device)
            report = generate_review(run,dataset,device=device,batch_size=config['batch_size'])
            predictions, meta = checked_predictions(report,info['rows']['val'],verified['checkpoint_sha256'],info['manifest_sha256'])
            if meta['selected_epoch'] != verified['selected_epoch'] or meta['metrics']['confusion_matrix'] != verified['metrics']['confusion_matrix']:
                raise ValueError('Candidate predictions do not reproduce saved metrics')
            records.append(verified | {'validation_review':str(report),'subtype_metrics':subtype_metrics(predictions)})
            for ident,r in predictions.items():
                paired[ident]['prediction_'+str(verified['input_size'])] = r['prediction']
        if any(file_hash(Path(p)) != sha for p,sha in protected.items()):
            raise ValueError('Prior input or artifacts changed')
        write_csv(out/'paired_validation_predictions.csv',list(paired.values()))
        completed = plan | {'status':'completed','output_dir':str(out),'diagnostics':diagnostic,
                            'records':records,'baseline_hashes_preserved':True,'automatic_model_promotion':False}
        write_json(out/'diagnosis.json',completed)
        return completed
    except BaseException as exc:
        write_json(out/'diagnosis.json',plan | {'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','baseline','review','outputs'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args = parser.parse_args()
    print(json.dumps(diagnose_and_compare(args.dataset,args.baseline,args.review,args.outputs,device=args.device),ensure_ascii=False))


if __name__ == '__main__':
    main()
