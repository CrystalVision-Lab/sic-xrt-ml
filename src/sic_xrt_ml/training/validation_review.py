"""Read-only val predictions and a local research review report."""

import argparse
import base64
import csv
import html
import io
import json
import random
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .patch_classifier import (
    CLASSES,
    PatchDataset,
    SmallPatchCNN,
    checkpoint_crop,
    classification_metrics,
    crop_image,
    dataset_info,
    file_hash,
    to_tensor,
    write_json,
)


def select_examples(rows, per_pair=12, seed=42):
    """Half highest scores, half seeded random remainder; four correct per class."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row['label'], row['prediction']].append(row)
    selected = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: (-r['model_score'], r['patch_id']))
        if key[0] == key[1]:
            selected.extend(rng.sample(items, min(4, len(items))))
        else:
            top_count = min((per_pair + 1)//2, len(items))
            selected.extend(items[:top_count])
            selected.extend(rng.sample(items[top_count:], min(per_pair-top_count, len(items)-top_count)))
    return selected


def preview_data(root, row, center_crop=None):
    # Full precision remains in TIFF; this PNG is only a normalized display preview.
    if file_hash(root / row['path']) != row['sha256']:
        raise ValueError('Preview patch integrity check failed')
    image = crop_image(tifffile.imread(root / row['path']), center_crop)
    rgb = to_tensor(image).permute(1, 2, 0).numpy()
    buffer = io.BytesIO()
    Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


PAGE = r'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>학습 결과 검수</title><style>
body{font:16px system-ui,sans-serif;margin:32px auto;max-width:1300px;padding:0 20px;color:#182434;background:#f4f6fa}
h1{margin-bottom:8px}p{line-height:1.7}button,select,textarea{font:inherit;padding:8px;border:1px solid #b7c2cf;border-radius:6px}
button{background:#fff;cursor:pointer;margin:4px}button.primary{background:#1765c1;color:white}
table{border-collapse:collapse;background:white}td,th{padding:9px 18px;border:1px solid #ccd4df;text-align:center}
.controls{position:sticky;top:0;background:#f4f6fa;padding:12px 0;z-index:2;border-bottom:1px solid #ccd4df}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;margin-top:20px}
article{background:white;border:1px solid #d7dfeb;border-radius:12px;padding:18px}article[hidden]{display:none}
.photo{position:relative;width:256px;height:256px;margin:14px auto;background:#eee}
.photo img{width:100%;height:100%;image-rendering:pixelated}.cross{position:absolute;width:16px;height:16px;transform:translate(-50%,-50%);pointer-events:none}
.cross:before,.cross:after{content:'';position:absolute;background:#ff2020;box-shadow:0 0 1px white}
.cross:before{width:16px;height:1px;top:7px}.cross:after{height:16px;width:1px;left:7px}
.hide-center .cross{display:none}.detail{font-size:13px;color:#576678;overflow-wrap:anywhere;line-height:1.6}
textarea{box-sizing:border-box;width:100%;height:60px;margin-top:8px}select{width:100%}
.badge{display:inline-block;background:#fff2db;padding:4px 8px;border-radius:5px}.correct{background:#e5f5eb}
</style><h1>학습 결과 검수</h1>
<h2>먼저 TED → TSD와 BPD → TSD 이미지를 확인하세요</h2>
<p>PLACEHOLDER_INTRO</p>PLACEHOLDER_TABLE
<p>패치 중심에 원래 표시한 결함이 있는지, 원래 라벨이 맞는지 확인하세요. 작은 패치만으로 확신하기 어려우면 <b>판단 보류</b>를 선택하세요.
모델과 다르게 보인다는 이유만으로 라벨을 바꾸지 마세요. 모델 점수는 보정된 확률이 아닙니다.</p>
<div class="controls"><button onclick="filter('TED','TSD')">① TED → TSD</button><button onclick="filter('BPD','TSD')">② BPD → TSD</button>
<button onclick="filter('wrong')">틀린 사례 전체</button><button onclick="filter('correct')">맞힌 비교 사례</button><button onclick="filter('all')">전체</button>
<label><input type="checkbox" checked onchange="document.body.classList.toggle('hide-center',!this.checked)">중심 표시</label>
<button class="primary" onclick="saveReview()">검수 의견 저장</button><span id="progress"></span></div>
<div class="grid">PLACEHOLDER_CARDS</div>
<script>
const provenance=PLACEHOLDER_PROVENANCE;
function filter(label,prediction){for(const c of document.querySelectorAll('article')){
 const wrong=c.dataset.label!==c.dataset.prediction;
 c.hidden=label==='all'?false:label==='wrong'?!wrong:label==='correct'?wrong:!(c.dataset.label===label&&c.dataset.prediction===prediction);}}
function progress(){const checked=[...document.querySelectorAll('article')].filter(c=>c.querySelector('select').value!=='unreviewed').length;
 document.getElementById('progress').textContent='확인 '+checked+' / '+document.querySelectorAll('article').length;}
document.addEventListener('change',progress);
function saveReview(){const entries=[...document.querySelectorAll('article')].map(c=>({patch_id:c.dataset.id,
 original_label:c.dataset.label,model_prediction:c.dataset.prediction,decision:c.querySelector('select').value,note:c.querySelector('textarea').value}));
 const body=JSON.stringify({...provenance,reviewed_at:new Date().toISOString(),entries},null,2);
 const url=URL.createObjectURL(new Blob([body],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');
 a.href=url;a.download='validation_review_'+new Date().toISOString().replaceAll(':','-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
filter('TED','TSD');progress();
</script></html>'''


def make_report(root, predictions, metrics, metadata):
    selected = select_examples(predictions)
    cards = []
    def escape(value):
        return html.escape(str(value), quote=True)
    for row in selected:
        wrong = row['label'] != row['prediction']
        input_size = metadata.get('center_crop') or int(row['size'])
        margin = (int(row['size']) - input_size)//2
        center_x = (float(row['x']) - int(row['left']) - margin + .5) / input_size * 100
        center_y = (float(row['y']) - int(row['top']) - margin + .5) / input_size * 100
        cards.append(f'''<article data-id="{escape(row['patch_id'])}" data-label="{escape(row['label'])}" data-prediction="{escape(row['prediction'])}">
<b>원래 라벨 {escape(row['label'])} → 예측 {escape(row['prediction'])}</b>
<p><span class="badge {'correct' if not wrong else ''}">{'일치' if not wrong else '불일치'}</span> 모델 점수 {row['model_score']:.3f}</p>
<div class="photo"><img alt="모델 입력 패치" src="data:image/png;base64,{preview_data(root,row,metadata.get('center_crop'))}"><span class="cross" style="left:{center_x}%;top:{center_y}%"></span></div>
<p class="detail">웨이퍼 {escape(row['wafer'])} · 좌표 ({escape(row['x'])}, {escape(row['y'])})<br>
원본: {escape(row.get('source_image',''))}<br>패치: {escape(row['patch_id'])}</p>
<select aria-label="검수 판단"><option value="unreviewed">아직 확인 안 함</option><option value="label_plausible">원래 라벨이 타당해 보임</option>
<option value="label_suspect">원래 라벨 확인 필요</option><option value="center_unclear">중심 결함이 불명확함</option><option value="uncertain">판단 보류</option></select>
<textarea aria-label="검수 메모" placeholder="관찰한 모양이나 확인할 점"></textarea></article>''')
    table = '<table><tr><th>원래 라벨 ＼ 예측</th>' + ''.join(f'<th>{c}</th>' for c in CLASSES) + '</tr>'
    for label, values in zip(CLASSES, metrics['confusion_matrix'], strict=True):
        table += f'<tr><th>{label}</th>' + ''.join(f'<td>{v}</td>' for v in values) + '</tr>'
    table += '</table>'
    intro = (f"선택된 {metadata['selected_epoch']}회차 모델 · 검증 {len(predictions):,}장 · "
             f"검증 macro F1 {metrics['macro_f1']:.3f}. 라벨 검수 상태: {escape(metadata['dataset_review_status'])}.<br>"
             f"표시되는 모델 입력: {metadata.get('input_size', '원본 패치')} 픽셀. 표시용 확대만 적용하며 모델 입력 리사이즈는 없습니다.<br>"
             f"여기에는 {len(selected)}개 비교 사례를 표시합니다. 혼동 쌍별 높은 점수와 무작위 사례를 섞었으므로 전체 비율을 대표하지 않습니다.<br>"
             '확인한 뒤 <b>검수 의견 저장</b>을 눌러 파일을 다운로드하세요. 새로고침/닫기 전 저장하세요. 의견은 원본 라벨에 자동 반영되지 않습니다.')
    page = PAGE.replace('PLACEHOLDER_INTRO', intro).replace('PLACEHOLDER_TABLE', table)
    page = page.replace('PLACEHOLDER_CARDS', ''.join(cards))
    return page.replace('PLACEHOLDER_PROVENANCE', json.dumps(metadata, ensure_ascii=True).replace('<', r'\u003c')), selected


def generate_review(run_dir, dataset_root, *, batch_size=32, device='cuda'):
    run_dir, dataset_root = Path(run_dir).resolve(), Path(dataset_root).resolve()
    if run_dir == dataset_root or dataset_root in run_dir.parents:
        raise ValueError('Report output must be outside the dataset')
    checkpoint_path = run_dir / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    center_crop = checkpoint_crop(config)
    info = dataset_info(dataset_root, config['size'], config['balanced_train'])
    if info['manifest_sha256'] != config['manifest_sha256'] or tuple(config['classes']) != CLASSES:
        raise ValueError('Checkpoint dataset or class contract differs')
    sources = dataset_root / '기록/sources.json'
    source_map = {s['source_id']: s.get('image', s.get('locator', ''))
                  for s in json.loads(sources.read_text(encoding='utf-8'))} if sources.exists() else {}
    rows = info['rows']['val']
    model = SmallPatchCNN().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    predictions, matrix = [], np.zeros((3, 3), dtype=np.int64)
    loader = DataLoader(PatchDataset(dataset_root, rows, config['size'], center_crop), batch_size=batch_size, num_workers=0)
    offset = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            scores = model(inputs.to(device)).softmax(1).cpu().numpy()
            for row, target, score in zip(rows[offset:offset+len(targets)], targets.tolist(), scores, strict=True):
                predicted = int(score.argmax())
                matrix[target, predicted] += 1
                predictions.append(dict(row, prediction=CLASSES[predicted], model_score=float(score[predicted]),
                                        source_image=source_map.get(row['source_id'], ''),
                                        **{f'score_{c}': float(score[i]) for i, c in enumerate(CLASSES)}))
            offset += len(targets)
    metrics = classification_metrics(matrix)
    output = run_dir / ('validation_review_' + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    metadata = {'schema_version': 1, 'task': 'validation_prediction_review', 'split': 'val',
                'research_only': True, 'dataset_review_status': info['review_status'],
                'manifest_sha256': info['manifest_sha256'], 'checkpoint_sha256': file_hash(checkpoint_path),
                'selected_epoch': checkpoint['epoch'], 'selection': 'per_pair_half_high_score_half_random_seed42_correct4',
                'center_crop': center_crop, 'input_size': center_crop or config['size'],
                'preprocessing': config['preprocessing'],
                'scores_calibrated': False, 'original_labels_modified': False, 'test_evaluated': False}
    page, selected = make_report(dataset_root, predictions, metrics, metadata)
    with (output / 'val_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write_json(output / 'summary.json', dict(metadata, metrics=metrics, files_predicted=len(predictions),
                                            preview_ids=[r['patch_id'] for r in selected]))
    (output / '검수.html').write_text(page, encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    print(generate_review(args.run, args.dataset, device=args.device))


if __name__ == '__main__':
    main()
