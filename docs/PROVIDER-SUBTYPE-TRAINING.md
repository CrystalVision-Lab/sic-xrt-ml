# 제공자 세부 종류 데이터셋 학습·평가 (#51)

sic-xrt-data-tools가 만든 `provider_subtype_point_dataset` `schema_version=1` 폴더(생산자 계약: sic-xrt-data-tools `docs/provider-subtype-dataset-v1.md`, PR #36)로 큰 종류(BPD/TED/TSD)와 방향 세부 종류(TED a~f, TSD a~c)를 함께 예측하는 분류기를 학습·평가하고, Analyzer가 받는 `frozen_xrt_patch_classifier` v1 계약을 지키는 ONNX 후보를 만든다. 이 저장소는 생산자 코드를 import하거나 복사하지 않고 폴더 형식(JSON/CSV/TIFF)만 읽는다. 모델 계약은 [model-contract.md](model-contract.md)를 본다.

라벨은 **제공자 라벨**이다(`labels_source=provider`, `expert_verified=false`, `research_only=true`). 아래 수치는 전문가 확정 정답에 대한 성능이 아니다.

## 1. 명령

```powershell
$py = ".\.venv\Scripts\python.exe"
$ds = "C:\XRT-datasets\provider-subtype-v1"
$out = "C:\XRT-models\provider-subtype-v1"
& $py -m sic_xrt_ml validate $ds
& $py -m sic_xrt_ml train $ds --config configs/provider-subtype.json --output $out\final\run
& $py -m sic_xrt_ml evaluate $ds --checkpoint $out\final\run\checkpoint.pt --output $out\final\evaluation --existing-bundle <앱 동봉 research 모델 폴더>
& $py -m sic_xrt_ml export --checkpoint $out\final\run\checkpoint.pt --evaluation $out\final\evaluation\evaluation.json --output $out\final\candidate-subtype
& $py -m sic_xrt_ml export --checkpoint $out\final\run\checkpoint.pt --evaluation $out\final\evaluation\evaluation.json --output $out\final\candidate-class-only --coarse-only
& $py -m sic_xrt_ml crossval $ds --config configs/provider-subtype.json --output $out\crossval --existing-bundle <앱 동봉 research 모델 폴더>
```

- 명령 이름은 v1/v2와 같다. `manifest.json`의 `schema`가 `provider_subtype_point_dataset`이면 세부 종류 파이프라인(`data/provider_subtype.py`, `training/subtype.py`, `evaluation/subtype_eval.py`, `export/subtype_bundle.py`)으로 간다. `loading.prepare`는 지금처럼 `verified_xrt_dataset` v1/v2만 받는다(v1/v2 동작 변화 없음).
- GPU: `configs/provider-subtype.json`의 `"device": "cuda"`. CUDA PyTorch가 없으면 `"cpu"`로 바꾼다(같은 결과를 기대하지 말 것: 장치가 다르면 부동소수 결과가 다르다).
- 실행 산출물(checkpoint, evaluation.json, predictions.json, 후보 번들)은 Git 밖(`C:\XRT-models\...`)에 둔다.

## 2. 로더 검증

- `schema`/`schema_version=1`, `research_only=true`, `labels_source="provider"`, `expert_verified=false`가 아니면 거부한다(`LABEL_PROVENANCE`).
- `classes=["BPD","TED","TSD"]`, `subtypes`, `fine_labels`(10개, 순서 고정)와 `patch`(128, uint8, RGB, `TIFF uncompressed`, 리사이즈·정규화 없음)가 정확히 같아야 한다.
- `output_hashes.json`: `manifest.json`·`samples.csv`(=`manifest.samples_sha256`)·`images.json`·`points.csv`·`roi_files.csv`의 SHA-256, 그리고 사용하는 **모든 패치 파일**의 SHA-256을 다시 계산한다. `patches_128/` 아래 파일 목록은 `samples.csv`와 정확히 같아야 한다. `output_hashes.json`에 없는 파일이 있으면 거부한다(`UNREFERENCED_FILE`, 예: `BUILD_FAILED.json`). 단 `preview/`는 표시 전용이라 무시한다(해시 목록에 없는 `preview/mean_by_area_stage_display_only.png` 포함).
- 행: `fine_label = label[_subtype]`, 경로 `patches_128/<split>/<fine_label>/<patch_id>.tif`, `center = floor(x, y)`, `left/top = center - 64`, `labels_source=provider`, `expert_verified=False`, `area == wafer`, 영상(`image_id`)의 Area가 `images.json`과 같음, 중복 patch_id·중복 파일 SHA 거부.
- 분할: 행의 split은 `manifest.split_map[area]`와 같아야 하고, 한 Area가 두 split에 나오면 거부한다(`AREA_SPLIT_MAP`/`AREA_SPLIT_LEAKAGE`). `split_map`의 `val`은 이 저장소의 `validation`이다. `counts.by_split_class_subtype`를 다시 센다.
- 패치: TIFF 매직, Pillow `format=TIFF`, `mode=RGB`(8-bit), 128×128, 단일 프레임, `compression=raw`. 회색·16-bit·압축·크기 다른 패치는 거부한다.
- `dataset_hash` = `manifest.json` 바이트의 SHA-256. manifest가 `samples_sha256`으로 `samples.csv`를, `samples.csv`가 패치마다 SHA-256을 고정하므로 이 값 하나가 학습 입력 전체를 고정한다.
- 준비도: 큰 종류마다 split별 최소 개수(기본 train 10 / validation 5 / test 5), 10개 세부 라벨이 모두 train에 1개 이상, split마다 Area 1개 이상(교차 검증 fold는 train의 큰 종류만 gate, 제외 Area에 없는 라벨은 보고만). 생산자 manifest에 `synthetic_fixture: true`(테스트용 확장 키)가 있으면 실제 학습을 거부하고 `--synthetic-smoke`만 허용한다.

## 3. 학습 선택

모든 선택은 validation(Area 2)만 보고 했다(실측 근거는 4절). test(Area 8)는 최종 평가에만 읽었다. 단, EMA BatchNorm 결함(4절)을 고치기 전 최종 run도 test를 한 번 평가했다(큰 종류 accuracy 0.928, TED→BPD 오분류 121개). 결함은 교차 검증 fold의 validation 진단으로 찾았고, 수정 뒤 후보 비교를 validation만으로 다시 했으며 선택된 설정(구조·가중치·EMA 값)은 수정 전과 같다. 수정 전 결과는 Git 밖 `superseded-ema-bn-bug-*`에만 남기고 비교에 쓰지 않는다.

- 구조: `xrt_local_contrast_subtype_cnn_mbc/v1`. 입력 안 국소 대비 정규화(v1과 같은 9×9 창) → 3×3 conv 두 개 + BatchNorm + ReLU + 2×2 max pool 4단(32/64/128/256 채널, 128→8 px) → 8×8 지도의 중앙 4×4 평균 → dropout 0.2 → 10개 세부 라벨. 약 1.18M parameter.
- 가중치 EMA(decay 0.998)로 validation·선택·export를 하고, EMA 모델의 BatchNorm 통계는 매 epoch 끝에 train 패치로 다시 추정한다.

- **증강 없음.** TED a~f / TSD a~c는 방향 세부 종류이므로 좌우·상하 뒤집기와 회전은 라벨을 바꾼다. 설정의 `augmentation`은 `"none"`만 허용한다.
- 손실: 세부 라벨 10개 Cross Entropy에 클래스 가중치. `balanced` = `n/(K·n_c)`, `sqrt_balanced` = 그 제곱근을 train 표본 가중 평균 1로 다시 맞춘 값, `none` = 가중치 없음.
- 모델 선택: 같은 클래스 가중치를 쓴 validation loss 최소 epoch(`best_model`). 조기 종료 `patience`.
- 큰 종류 점수 = 같은 큰 종류에 속한 세부 softmax 점수의 합. 큰 종류 예측은 그 합의 argmax(ONNX `probabilities`와 같음)다.
- 결정성: seed, `torch.use_deterministic_algorithms(True)`, 전역 평균 풀링은 `mean`(CUDA backward 결정적). 같은 장치에서 epoch 단위 재개 결과가 처음부터 학습과 같다(테스트로 확인).

## 4. 구조·가중치 선택 근거 (validation = Area 2만)

모두 같은 split(train Area 1·9 = 13,528, validation Area 2 = 1,442), seed 2026, batch 64, AdamW lr 1e-3, StepLR(10, 0.5), 최대 30 epoch, patience 6. 표의 값은 validation loss 최소 epoch의 validation 수치다. `f_F1`은 validation에 있는 세부 라벨만의 macro F1이다. validation의 TED_e 29, TED_f 20, TSD_b 6, TSD_c 2개는 표본이 적어 `f_F1`이 크게 흔들린다(±0.05 정도 차이는 잡음으로 본다).

| run | 구조 | 가중치 | EMA | 선택 epoch | val loss | 큰 종류 acc | 큰 종류 bal.acc | 큰 종류 macro F1 | 세부 acc | 세부 f_F1 | 큰 종류 맞을 때 세부 acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 trunk 8/16/32 | `_s` | sqrt | - | 17/30 (patience 30) | 2.276 | 0.757 | 0.582 | 0.533 | 0.091 | 0.067 | 0.124 |
| GroupNorm | `_m` | sqrt | - | 27/30 | 0.850 | 0.946 | 0.902 | 0.909 | 0.860 | 0.489 | 0.899 |
| BatchNorm | `_mb` | sqrt | - | 6/12 | 0.808 | 0.926 | 0.894 | 0.879 | 0.861 | 0.524 | 0.924 |
| BatchNorm | `_mb` | sqrt | 0.998 | 6/12 | 0.659 | 0.960 | 0.907 | 0.926 | 0.908 | 0.595 | 0.934 |
| BatchNorm (seed 2027) | `_mb` | sqrt | 0.998 | 6/12 | 0.666 | 0.960 | 0.908 | 0.925 | 0.911 | 0.605 | 0.939 |
| BatchNorm + 중앙 풀링 | `_mbc` | sqrt | - | 4/10 | 0.796 | 0.954 | 0.908 | 0.919 | 0.869 | 0.594 | 0.904 |
| **BatchNorm + 중앙 풀링 (채택)** | `_mbc` | sqrt | 0.998 | 4/10 | 0.672 | 0.960 | 0.903 | 0.923 | 0.915 | **0.631** | **0.945** |
| BatchNorm + 중앙 풀링 (seed 2027) | `_mbc` | sqrt | 0.998 | 5/11 | 0.689 | 0.959 | 0.902 | 0.922 | 0.915 | 0.627 | 0.949 |
| BatchNorm + 중앙 풀링 | `_mbc` | none | 0.998 | 6/12 | 0.278* | 0.967 | 0.920 | 0.937 | 0.917 | 0.576 | 0.941 |
| BatchNorm + 중앙 풀링 | `_mbc` | balanced | 0.998 | 4/10 | 1.407* | 0.935 | 0.841 | 0.866 | 0.863 | 0.572 | 0.914 |

\* 가중치가 다르면 val loss의 척도가 달라 run 사이 비교에 쓰지 않는다.

선택 기준은 새 과제인 세부 종류 성능(validation 세부 f_F1, 큰 종류가 맞을 때 세부 정확도)이고, 큰 종류 balanced accuracy가 크게 나빠지지 않는지를 함께 봤다.

- **작은 v1 trunk(`_s`, 6,746 parameter)는 과소적합**: 30 epoch에도 세부 정확도 0.09. 더 큰 trunk(약 1.18M parameter)를 새 구조 id로 추가했다.
- **BatchNorm vs GroupNorm**: GroupNorm은 수렴이 느려(27 epoch에 val loss 0.850, BatchNorm은 6 epoch에 0.808) 같은 epoch 예산에서 더 나쁘다.
- **EMA**: EMA 없는 BatchNorm run은 epoch마다 validation이 크게 흔들렸다(5·7·9 epoch에 큰 종류 정확도 0.61·0.12·0.21). 가중치 EMA(decay 0.998, 약 2.4 epoch 지평)를 쓰면 `_mb` 세부 f_F1 0.524 → 0.595, `_mbc` 0.594 → 0.631.
  - EMA 모델의 BatchNorm running 통계는 매 epoch 끝에 train 패치로 다시 추정한다(`recompute_batch_norm`). 처음 구현은 EMA 가중치에 원래 가중치의 running 통계를 평균해 붙였는데, Area 1을 제외한 교차 검증 fold에서 EMA 모델이 validation 전체를 한 클래스로 예측했다(원래 가중치 0.937, EMA 0.123, 통계 재추정 후 0.937). 이 결함을 고친 뒤 EMA run을 모두 다시 돌렸고 표는 수정 후 값이다. 수정 전 run과 그 최종 평가는 Git 밖 `superseded-*` 폴더에 남겼고 결과 비교에 쓰지 않는다.
- **클래스 가중치**: `balanced`는 TSD_c 가중치가 135배로 극단적이고 큰 종류 balanced accuracy가 0.841로 가장 나빴다. `none`은 큰 종류는 가장 좋지만 세부 f_F1이 낮았다. 중간값 `sqrt_balanced`(TSD_c 14배, TED_f 8.9배)를 썼다.
- **중앙 풀링(`_mbc`)**: 8×8 특징 지도의 중앙 4×4만 평균한다(라벨 점이 패치 중심이고 가장자리는 이웃 결함일 수 있음). 두 seed 모두 `_mb`보다 세부 f_F1(0.631/0.627 vs 0.595/0.605)과 세부 정확도(0.945/0.949 vs 0.934/0.939)가 높고 큰 종류는 비슷했다(0.903/0.902 vs 0.907/0.908). 최종 설정 = `configs/provider-subtype.json`(`_mbc`, sqrt_balanced, EMA 0.998, seed 2026).
- 같은 설정·seed는 같은 결과를 낸다(중단한 BatchNorm run의 6 epoch 수치와 다시 돌린 `_mb` run이 소수점까지 같음, 최종 학습과 선택 run도 같음).

### 방향 민감도 확인 (train/validation만, test 미사용)

최종 모델에 train/validation 패치를 좌우 반전해 넣었을 때 정답률(세부 라벨 기준):

| 라벨 (Area) | 원본 | 좌우 반전 |
|---|---|---|
| TED_a (1 / 9) | 0.885 / 0.968 | 0.022 / 0.000 |
| TED_b (1 / 9) | 0.966 / 0.992 | 0.014 / 0.018 |
| TED_c (1 / 9) | 0.990 / 0.977 | 0.065 / 0.031 |
| TED_d (1) | 0.967 | 0.034 |
| TSD_a (1 / 9) | 0.994 / 0.997 | 0.975 / 0.864 |
| TSD_b (1 / 9) | 0.990 / 0.993 | 0.936 / 0.950 |
| TSD_a (2, ROI가 원본 JPEG 격자라 `hflip`으로 보정된 점, n=829) | 0.989 | 0.887 |

TED 세부 종류는 좌우 반전에 거의 완전히 바뀐다. 뒤집기 증강을 쓰면 안 되고, 반전된 입력(EXIF 방향을 무시하고 디코딩한 제공 JPEG)에서는 TED 세부 예측이 틀린다. TSD a/b는 좌우 반전에 거의 변하지 않는다(좌우 대칭이 아닌 다른 차이로 구분되는 것으로 보인다). Area 2 TSD_a는 원본(TIFF 격자) 쪽이 더 잘 맞아 data-tools의 `hflip` 좌표 보정과 일치한다.

## 5. 최종 test 결과 (Area 8, 1,762 패치, 1회 평가)

최종 run: `configs/provider-subtype.json`, 코드 `597307b`, 4 epoch 선택(10 epoch에서 조기 종료). 데이터셋 `dataset_hash` `f23fc2c4…05fd`(= manifest.json SHA-256). 선택 epoch의 validation 수치는 4절 채택 행과 같다.

### 큰 종류 3클래스 (세부 점수 합의 argmax) — 앱 동봉 연구 모델과 같은 패치 비교

| 지표 | 새 모델 | 동봉 모델 `spatial64-c4-24368b8f6613` | 차이 (patch bootstrap 95%) |
|---|---|---|---|
| accuracy | 0.994 | 0.988 | +0.006 [+0.001, +0.012] |
| balanced accuracy | 0.956 | 0.963 | −0.007 [−0.062, +0.050] |
| macro F1 | 0.960 | 0.908 | +0.052 [+0.009, +0.100] |

| 클래스 | support | 새 P / R / F1 | 동봉 P / R / F1 |
|---|---|---|---|
| BPD | 32 | 0.903 / 0.875 / 0.889 | 0.617 / 0.906 / 0.734 |
| TED | 1,170 | 0.997 / 0.995 / 0.996 | 0.997 / 0.984 / 0.991 |
| TSD | 560 | 0.993 / 0.998 / 0.996 | 0.998 / 1.000 / 0.999 |

혼동 행렬(행 = 정답 BPD/TED/TSD, 열 = 예측): 새 모델 `[[28,3,1],[3,1164,3],[0,1,559]]`, 동봉 모델 `[[29,3,0],[18,1151,1],[0,0,560]]`. 차이는 주로 동봉 모델이 TED 18개를 BPD로 본 것이다(BPD 정밀도 0.62).

**비교는 참고용이다.** 동봉 모델은 같은 제공자 자료(이 Area들을 포함했을 가능성이 높음)로 학습됐고 학습 split을 이 저장소에서 확인할 수 없다. BPD는 32개뿐이라 BPD 지표와 balanced accuracy의 구간이 넓다. bootstrap은 패치 단위라 같은 영상의 이웃 패치 상관을 무시하므로 구간이 실제보다 좁다.

### 세부 10클래스

| 라벨 | support | precision | recall | F1 |
|---|---|---|---|---|
| BPD | 32 | 0.906 | 0.906 | 0.906 |
| TED_a | 311 | 0.881 | 0.955 | 0.917 |
| TED_b | 464 | 0.986 | 0.940 | 0.962 |
| TED_c | 196 | 0.900 | 0.969 | 0.934 |
| TED_d | 93 | 0.760 | 0.817 | 0.788 |
| TED_e | 50 | 0.343 | 0.240 | 0.282 |
| TED_f | 56 | 0.548 | 0.411 | 0.469 |
| TSD_a | 187 | 0.989 | 0.989 | 0.989 |
| TSD_b | 373 | 0.989 | 0.992 | 0.991 |
| TSD_c | 0 | – | – | – (Area 8에 없음. BPD 1개, TSD_b 1개가 TSD_c로 예측됨) |

- 세부 accuracy 0.918, test에 있는 9개 라벨의 macro F1 0.804, balanced accuracy 0.802.
- **큰 종류가 맞았을 때 세부 정확도 0.922**(1,723개): TED 0.888(1,164개), TSD 0.993(559개).
- TED 세부 안 혼동(행 = 정답 a~f): `a [297,1,0,2,2,8]`, `b [4,436,20,2,0,0]`, `c [0,4,190,2,0,0]`, `d [5,0,0,76,9,0]`, `e [9,0,0,18,12,11]`, `f [22,0,0,0,11,23]`. TED_b→c 20개(생산자가 지적한 b/c 유사), TED_e는 d·f·a로, TED_f는 a·e로 흩어진다. TED_e/f는 train에 68/25개뿐이다.
- 세부 argmax의 큰 종류와 큰 종류 argmax가 다른 패치 1개.

## 6. Area 단위 교차 검증 (leave-one-Area-out, 4 fold)

`crossval` 명령. fold마다 Area 하나를 test로 빼고, 남은 Area 중 패치가 가장 적은 Area를 validation, 나머지를 train으로 쓴다(validation 선택에 제외 Area 정보를 쓰지 않음). 설정은 최종과 같다. 제외 Area에 없는 클래스·세부 라벨은 막지 않고 보고한다(`readiness.mode = cross_validation_fold`). 동봉 모델도 같은 제외 Area 패치로 평가했다(학습 자료가 겹칠 수 있어 참고용). 총 약 30분(RTX 3050).

| 제외(test) Area | train / val | test 패치 | 큰 종류 acc | 큰 종류 bal.acc* | 큰 종류 macro F1* | 세부 acc | 세부 macro F1* | 큰 종류 맞을 때 세부 acc (TED / TSD) | 동봉 모델 acc / bal.acc* / macro F1* |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8·9 / 2 | 9,252 | 0.964 | 0.773 | 0.815 | 0.843 | 0.534 | 0.872 (0.729 / 0.987) | 0.975 / 0.868 / 0.897 |
| 2 | 1·9 / 8 | 1,442 | 0.960 | 0.903 | 0.923 | 0.915 | 0.631 | 0.945 (0.861 / 0.987) | 0.958 / 0.892 / 0.912 |
| 8 | 1·9 / 2 | 1,762 | 0.994 | 0.956 | 0.960 | 0.918 | 0.804 | 0.922 (0.888 / 0.993) | 0.988 / 0.963 / 0.908 |
| 9 | 1·8 / 2 | 4,276 | 0.989 | 0.993 | 0.994 | 0.960 | 0.887 | 0.972 (0.967 / 0.994) | 0.996 / 0.998 / 0.998 |
| **평균 ± 표준편차** | | | **0.977 ± 0.017** | **0.906 ± 0.096** | **0.923 ± 0.078** | **0.909 ± 0.049** | **0.714 ± 0.160** | **0.928 ± 0.042** | 0.979 ± 0.017 / 0.930 ± 0.061 / 0.929 ± 0.047 |

\* 그 fold의 test에 있는 클래스·라벨만 평균한 값. Area 9에는 BPD·TED_e·TSD_c가 없고 Area 8에는 TSD_c가 없다.

- Area 8 fold는 최종 run과 같은 split·설정이라 결과가 소수점까지 같다(결정성 확인).
- **Area 1 제외 fold가 가장 나쁘다.** train이 Area 8·9(6,038 패치, BPD 32개, TSD_c 0개)뿐이라 BPD F1 0.516, TED_b 0.365, TED_e 0.027, TED_d 0.711이다. Area 1은 After 4장 + Before 1장(9,252 패치)으로 다른 Area와 단계(Before/After) 구성도 다르다.
- 세부 종류 성능의 Area 간 편차(세부 macro F1 0.53~0.89)가 크다. 한 Area에서 얻은 수치를 다른 웨이퍼에 그대로 기대하면 안 된다. TSD a/b는 모든 fold에서 0.98 이상, TED는 0.73~0.97.
- 희소 라벨: TED_e(Area 1 68, 2 29, 8 50), TED_f(1 21, 2 20, 8 56, 9 4), TSD_c(1 10, 2 2)는 fold마다 support가 30 미만이거나 없다. 이 라벨의 F1(0.0~0.47)은 잡음이 크다.
- 큰 종류 accuracy 평균은 동봉 모델과 비슷하다(0.977 vs 0.979). 동봉 모델은 이 Area들로 학습됐을 수 있어 실제 일반화 비교가 아니다.

## 7. ONNX 후보 번들

| 폴더 (Git 밖) | 출력 | `bundle_id` | model_sha256 | Analyzer `FrozenClassifier` | Analyzer 모델 업데이트 검증 `validate_in_process` |
|---|---|---|---|---|---|
| `C:\XRT-models\provider-subtype-v1\final\candidate-subtype` | `probabilities` [N,3] + `subtype_probabilities` [N,10] | `provider-subtype-1afb7ce8b904` | `1afb7ce8…cb740` | 받음 | **거부** `ACTUAL_ONNX_CONTRACT`(출력 1개만 허용) |
| `C:\XRT-models\provider-subtype-v1\final\candidate-class-only` | `probabilities` [N,3] | `provider-class-e1106da30881` | `e1106da3…c7fa1` | 받음 | 받음 `ARTIFACT_VERIFIED` |

- 두 번들은 같은 checkpoint에서 나왔고 `probabilities`는 같다. parity(CPU, batch 1/7): 최대 절대 오차 2.4e-7, `probabilities`와 세부 점수 그룹 합 차이 0.
- Analyzer 확인은 Analyzer 코드를 바꾸지 않고 그 가상환경에서 `FrozenClassifier(bundle).predict(...)`와 `model_updates.validation.validate_in_process(bundle)`를 읽기 전용으로 실행했다.
- 결론: 두 출력 번들은 지금 Analyzer의 직접 로드(연구 분석)에서는 동작하지만, 서버 등록·앱 설치 경로(`model_updates/validation.py`, 서버 `model_admin.py`)는 출력 수가 1이 아니면 거부한다. 지금 앱에 배포할 수 있는 형태는 `--coarse-only` 번들이고, 세부 종류를 쓰려면 아래 Analyzer 후속 작업이 먼저 필요하다.

## 8. 한계

- 라벨은 제공자 라벨이며 TED a~f / TSD a~c의 물리적 정의는 확인하지 않았다. 배경 클래스가 없어 후보점이 결함이 아닌 경우를 걸러내지 못한다.
- 라벨이 있는 Area는 4개(웨이퍼 4장)이고 영상 8장이다. test는 Area 8 한 장(1,762 패치)이며 BPD는 32개뿐이다. 패치는 같은 영상 안에서 서로 독립이 아니다.
- 세부 종류는 TIFF 픽셀 격자 기준 방향이다. 제공 JPEG는 TIFF의 좌우 반전 사본(EXIF orientation 2)이므로 EXIF 방향 처리 없이 JPEG를 디코딩하면 TED 세부 예측이 반전된다(4절 표: 반전 시 TED 정답률 0.00~0.07).
- TED_e/TED_f/TSD_c는 학습 표본이 적고(68/25/10) test 성능이 낮거나 측정되지 않았다.
- 점수는 보정되지 않았다. `improvement = NOT_VERIFIED`, `candidate_status = AWAITING_EXPERT_MODEL_ACCEPTANCE`.

## 9. Analyzer 후속 작업 (이 PR 범위 밖)

- `src/sic_xrt_analyzer/model_updates/validation.py`(`validate_in_process`)와 `model_updates/contracts.py`(`frozen_manifest`): manifest에 `subtype_output`이 있으면 두 번째 출력 `subtype_probabilities` [N,10]과 블록 값을 검사하고 출력 2개를 허용. 서버 `model_admin.py`도 같은 검증을 쓴다.
- `src/sic_xrt_analyzer/analysis/research.py`: `FrozenClassifier.predict`가 `subtype_output`이 있을 때 `["probabilities","subtype_probabilities"]`를 함께 받아 `classify_points` 결과에 `subtype`, `subtype_score`, `score_TED_a`… 를 넣고, `subtype_provenance`를 모델(제공자 라벨, 미검증)로 기록. 기존 규칙 기반 TED 방향(`analysis/defect_morphology.py`, `RULE_DIPOLE_ORIENTATION_UNCALIBRATED`)과 이름(`TED-a` vs `TED_a`)·출처가 다르므로 섞지 말고 구분해서 표시.
- `read_rgb`: JPEG를 `Image.open` 원시 픽셀로 읽어 EXIF 방향을 적용하지 않는다. 제공 JPEG(EXIF orientation 2)에서는 TIFF와 좌우가 반대이므로, 세부 종류를 표시할 때는 EXIF 방향을 적용(`ImageOps.exif_transpose`)하거나 방향이 TIFF와 다를 수 있는 입력에서 세부 종류를 숨기고 경고해야 한다. 큰 종류 점수에도 같은 방향 처리를 적용할지 함께 정한다.
- 결과 표·웨이퍼 지도(`wafer_map/defect_map.py`의 `SUBTYPES`, `engine/wafer.py` 필터)와 모델 성능 화면(`engine/model_performance.py`의 `no_direction_subtypes_or_physical_conversion_claim` 문구)은 TED 세부만 가정한다. TSD a~c와 모델 출처 세부 종류를 받도록 확장.
