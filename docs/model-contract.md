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
- ML의 `--existing-bundle` 비교는 v1/v2 번들을 모두 받는다. 3클래스 기준 모델을 4클래스 정답과 비교하면 기준 모델은 `NOT_DEFECT`를 예측하지 못하므로 그 열이 0이다.

데이터 소비 규칙은 [VERIFIED-DATASET-V2-TRAINING.md](VERIFIED-DATASET-V2-TRAINING.md)를 본다.
