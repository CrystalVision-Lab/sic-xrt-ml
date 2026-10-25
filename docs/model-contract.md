# 모델 산출물 계약

ML이 만드는 후보 폴더는 `model.onnx`, `manifest.json`, `candidate-registry.json` 세 파일로만 구성한다. 모델 파일·데이터는 Git에 넣지 않는다. 현재 실제 승인 모델은 없다.

## frozen_xrt_patch_classifier

| 항목 | v1 (3클래스) | v2 (4클래스) |
|---|---|---|
| `schema_version` | 1 | 2 |
| `classes` | `["BPD","TED","TSD"]` | `["BPD","TED","TSD","NOT_DEFECT"]` |
| `background_class` | `false` | `true` |
| `output` | `{"name":"probabilities","shape":["N",3],"meaning":"uncalibrated_class_scores"}` | 같은 블록, `shape:["N",4]` |
| 학습 데이터 | `verified_xrt_dataset` v1 | `verified_xrt_dataset` v2 (검수 라벨 합의) |

공통(두 버전 의미 동일):

- `schema="frozen_xrt_patch_classifier"`, `status="validated_export"`, `model_file="model.onnx"`, `research_only=true`, `localizes_defects=false`
- `input`: `{"name":"images","dtype":"float32","shape":["N",3,128,128],"conversion":"uint8_RGB_to_NCHW_div255","local_contrast_inside_model":true}`
- 출력 `probabilities`는 float32이고 각 행의 합이 1(softmax)이다. 보정(calibration)하지 않은 점수이며 정확도가 아니다.
- 클래스 인덱스 0–2는 v1과 같다. v2는 배경 `NOT_DEFECT`를 인덱스 3으로 덧붙인다. v2 출력에서 앞 세 열만 보고 argmax를 하면 배경 점수가 빠지므로, 소비자는 `schema_version`으로 분기한다.
- 그 밖의 필드 `model_sha256`, `dataset_hash`, `config_hash`, `code_revision`, `checkpoint_sha256`, `evaluation_hash`, `architecture`(`xrt_local_contrast_cnn/v1`, 헤드 크기만 클래스 수를 따름), `synthetic_smoke`, `independent_test_passed=false`, `improvement="NOT_VERIFIED"`, `candidate_status`, `parity`는 v1과 의미가 같다. v2 manifest에 새 필드는 없다.
- export는 ONNX checker, CPUExecutionProvider batch 1/7 parity(atol 1e-5, rtol 1e-4), 행 합 1을 검사한다.

3클래스 run은 지금도 v1 manifest를 만든다(필드·값 변화 없음). 4클래스 manifest는 v2 데이터셋으로 학습한 run에서만 나온다.

### v1 + 세부 종류 출력 (`subtype_output`, #51)

`provider_subtype_point_dataset` v1로 학습한 run은 **schema_version 1 그대로**의 manifest를 만들고, ONNX 출력 하나와 manifest 키 몇 개를 더한다. v1 소비자가 비교하는 블록(`schema`, `status`, `classes`, `input`, `output`, `model_file`, `research_only`, `background_class=false`, `localizes_defects=false`, `model_sha256`)은 값이 완전히 같다.

| ONNX 출력 | shape | 의미 |
|---|---|---|
| `probabilities` (첫 번째) | `[N,3]` | BPD/TED/TSD. 각 큰 종류에 속한 세부 점수의 합. 행 합 1 |
| `subtype_probabilities` (두 번째) | `[N,10]` | 세부 라벨 softmax. 행 합 1 |

추가 manifest 키:

```json
"subtype_output": {"name":"subtype_probabilities","shape":["N",10],
  "labels":["BPD","TED_a","TED_b","TED_c","TED_d","TED_e","TED_f","TSD_a","TSD_b","TSD_c"],
  "class_of_label":[0,1,1,1,1,1,1,2,2,2],
  "meaning":"uncalibrated_subtype_scores",
  "relation":"probabilities[:, k] = sum of subtype_probabilities over labels with class_of_label == k",
  "orientation":"tiff_pixel_grid_direction_subtypes_mirror_with_the_image"},
"labels_source":"provider", "expert_verified":false, "limitations":[...],
"bundle_id":"provider-subtype-<model_sha256 앞 12자>", "dataset":{...provider 데이터셋 출처...}
```

- `subtype_output`이 있으면 블록이 위 값과 정확히 같아야 하고 ONNX 출력은 정확히 두 개(`probabilities`, `subtype_probabilities`, 둘 다 float32, `[N, 3]`/`[N, 10]`)여야 한다. 없으면 지금처럼 출력 하나만 허용한다. ML의 `frozen_session`이 이 규칙을 검사한다. `subtype_output`은 schema_version 1에서만 허용한다.
- 세부 종류(TED a~f, TSD a~c)는 **방향** 세부 종류이고 TIFF 픽셀 격자 기준이다. 좌우 반전된 입력(예: 제공 JPEG를 EXIF 방향 처리 없이 디코딩한 영상, 제공 JPEG는 TIFF의 좌우 반전 사본)은 세부 종류 방향도 뒤집힌다. 큰 종류 점수는 영향이 작지만 세부 점수는 해석하면 안 된다.
- 라벨은 제공자 라벨이며 전문가 확정 정답이 아니다(`labels_source=provider`, `expert_verified=false`). 배경 클래스는 없다.
- export는 두 출력 모두 CPUExecutionProvider batch 1/7 parity(atol 1e-5, rtol 1e-4), 두 출력 행 합 1, `probabilities`와 세부 점수 그룹 합의 일치(1e-5)를 검사하고 `parity.results`에 출력별 최대 오차를 남긴다.
- 두 번들 모두 `trained_fine_labels`(10개 세부 라벨)와 `coarse_rule`을 기록한다. `--coarse-only` 번들은 `subtype_output` 없이 출력 하나만 가진다.
- `candidate-registry.json`에는 `dataset`(스키마·버전·`dataset_hash`=manifest.json SHA-256·`samples_sha256`·`output_hashes_sha256`·생산자 코드 리비전·`split_map`)이 들어간다.

## xrt_model_candidate/v1 (candidate-registry.json)

스키마 이름은 그대로다. v1 필드(`candidate_id`, `manifest_sha256`, `dataset_hash`, `evaluation`, `status`, `distribution="NOT_AUTHORIZED"`, `replaces_existing_model=false`)를 유지한다.

v2 데이터셋으로 학습한 후보만 `dataset` 필드를 더해 라벨 데이터셋을 가리킨다.

```json
{"schema":"verified_xrt_dataset","schema_version":2,"dataset_id":"<uuid>","dataset_version":"<이름>",
 "dataset_hash":"<sha256>","content_hash":"<sha256>","label_schema":"review_label/v1",
 "label_event_set_hash":"<sha256>","parameters":{"seed":0,"min_reviewers":1,"consensus":"majority","implicit_weight":1.0,"received_before":"..."}}
```

v2일 때 `evaluation`(evaluation.json 전체)에는 `classes`, `dataset`, `coverage`, `random_sample_candidate`, `random_sample_baseline`이 추가된다.

## 호환성 영향

- Analyzer 소비자는 frozen manifest `schema_version` 1과 2를 모두 받아야 4클래스 후보를 쓸 수 있다. v1만 받는 소비자는 v2를 계약 오류로 거부한다(안전한 실패).
- Analyzer 서버 모델 레지스트리의 현재 `register()`는 v1 `DatasetVersion`의 `dataset_hash`만 조회한다. v2 라벨 데이터셋(`label_dataset_versions`)으로 학습한 후보를 등록하는 일은 Analyzer 쪽 후속 작업이다.
- 세부 종류 번들(출력 2개)은 manifest 검사(`frozen_manifest`)를 통과하고 Analyzer `analysis/research.py`의 `FrozenClassifier`도 받는다(`session.run(["probabilities"], ...)`은 첫 출력만 계산, #51에서 실제 번들로 확인). 그러나 Analyzer 모델 업데이트·서버 등록 검증(`model_updates/validation.py`의 `validate_in_process`)은 ONNX 출력이 정확히 1개여야 해 **`ACTUAL_ONNX_CONTRACT`로 거부**한다. 지금 배포 경로에 올릴 수 있는 것은 같은 모델의 `--coarse-only` 번들(출력 `probabilities` 하나, `subtype_output` 키 없음, `bundle_id` `provider-class-…`)이고, 이것은 `validate_in_process`도 통과한다.
- 세부 종류를 쓰려면 Analyzer가 `subtype_output`을 읽어 출력 2개를 허용하고 `subtype_probabilities`를 함께 요청해야 한다(Analyzer 후속 작업). 그때 JPEG 입력의 좌우 반전(EXIF orientation 2)을 처리하지 않으면 TED 세부 방향이 뒤집힌다.
- `subtype_output` 없이 출력이 두 개인 ONNX, 블록 값이 다른 manifest는 ML `frozen_session`이 거부한다(`WRONG_MODEL_CONTRACT`/`ACTUAL_ONNX_CONTRACT`). 기존 v1/v2 번들 검사는 바뀌지 않았다.
- ML의 `--existing-bundle` 비교는 v1/v2 번들을 모두 받는다. 3클래스 기준 모델을 4클래스 정답과 비교하면 기준 모델은 `NOT_DEFECT`를 예측하지 못하므로 그 열이 0이다.

데이터 소비 규칙은 [VERIFIED-DATASET-V2-TRAINING.md](VERIFIED-DATASET-V2-TRAINING.md)를 본다.
