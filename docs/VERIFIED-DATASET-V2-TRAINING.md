# verified_xrt_dataset/v2 4클래스 학습 (#48)

Analyzer가 검수 라벨 합의로 만든 `verified_xrt_dataset` `schema_version=2` 반출 폴더를 읽어 `BPD/TED/TSD/NOT_DEFECT` 4클래스 분류기를 학습·평가하고, `frozen_xrt_patch_classifier` v2 후보를 만든다. 생산자 계약은 Analyzer `docs/server/VERIFIED-DATASET-V2-CONTRACT.md`(#189)이며 이 저장소는 그 소스를 복사하지 않고 폴더 형식만 소비한다. 모델 계약은 [model-contract.md](model-contract.md)를 본다.

## 1. 반출 (Analyzer 서버, ADMIN)

Analyzer 문서에 적힌 명령 그대로다.

```
python -m xrt_server.label_dataset_admin --credential-file <private.json> --dataset-id <id> --output <새 private 폴더>
```

- 반출 전 서버에서 해당 버전 `usage_status`가 `AVAILABLE`인지 확인한다. ML은 오프라인이라 철회 여부를 조회하지 않는다(`readiness.usage_status`에 `NOT_CHECKED_OFFLINE`로 표시). `RESTRICTED`가 되면 그 사본으로 학습하지 않고 새 버전을 만든다.
- 폴더 구성: `manifest.json`, `consensus-report.json`, `splits/{train,validation,test}.jsonl`, `images/<PNG-SHA256>.png`. 폴더는 Git 밖 private 위치에 둔다.

## 2. 검증·학습·평가·export

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m sic_xrt_ml validate <export-folder>                       # 기본 gate 10,5,5
& $py -m sic_xrt_ml validate <export-folder> --min-per-class 20,8,8
& $py -m sic_xrt_ml train <export-folder> --config configs/patch-baseline.json --output artifacts/<new-run> [--min-per-class 20,8,8]
& $py -m sic_xrt_ml evaluate <export-folder> --checkpoint artifacts/<new-run>/checkpoint.pt --output artifacts/<new-eval> [--existing-bundle <기존 v1 또는 v2 번들>]
& $py -m sic_xrt_ml export --checkpoint artifacts/<new-run>/checkpoint.pt --evaluation artifacts/<new-eval>/evaluation.json --output artifacts/<new-candidate>
```

명령은 v1과 같다. `manifest.json`의 `schema_version`으로 v1/v2 로더를 고른다. `--min-per-class`는 v2 전용이다(v1에 주면 `MIN_PER_CLASS_V2_ONLY`). evaluate는 학습 때 기록한 gate를 그대로 쓴다.

## 3. 로더 검증 (v1 로더와 같은 수준)

- `schema`/`schema_version=2`/`label_schema="review_label/v1"`, `class_mapping={"BPD":0,"TED":1,"TSD":2,"NOT_DEFECT":3}`, `background_class="NOT_DEFECT"`, `crop_contract`, `split_policy`가 계약과 정확히 같아야 한다.
- `dataset_hash`(dataset_hash 제외 manifest 전체), `content_hash`(id·버전·생성 시각·생성자·두 해시 제외), `file_manifest_hash`를 canonical JSON SHA-256으로 다시 계산한다.
- 파일 inventory = crop이 있는 행의 `image_path`. 비참조 파일, symlink/junction, 경로 탈출, 과대 파일, 중복 JSON key, NaN을 거부한다. `consensus-report.json`과 split JSONL은 manifest 내용과 같아야 한다(바이트가 아니라 파싱 값 비교: 생산자가 JSONL은 ASCII escape, manifest는 UTF-8로 쓴다).
- 행: `sample_id = sha256(canonical([source_sha256, page, candidate_id]))`, `label`↔`class_index`↔`is_background`, `weight`∈(0,1], 원본 좌표, `rect`가 원본 안·한 변 ≤128.
- crop PNG: 파일 SHA, PNG IHDR 8-bit L 또는 RGB(16-bit·팔레트 거부), 크기=`rect`, `pixel_hash = sha256("{mode}:{w}x{h}:" + 픽셀)`, dHash(회색 표준편차 < 2이면 `null`).
- 누수: 같은 `source_sha256`·`split_group_id`·`wafer_id`·`group_mapping_ids`가 split을 넘으면 거부. 같은 픽셀 중복, split 사이 dHash Hamming ≤4 거부(밴드 버킷으로 20,000행도 처리). `evaluation_only` 또는 Wafer `8`은 test 전용(v1 규칙 유지).

## 4. crop이 없거나 모델 입력과 다른 표본

모델 입력은 정확히 128×128 uint8 RGB다. Analyzer 추론도 흑백·128px 미만 패치를 모델에 넣지 않으므로(`require_rgb`, `incomplete_128px_patch`) 학습도 변환하지 않는다.

| 사유 | 의미 |
|---|---|
| `NO_CROP` | 검수자가 조각 전송에 동의하지 않아 crop 없음 |
| `CROP_NOT_128` | 원본이 128px보다 작아 조각이 작음(리사이즈·패딩 안 함) |
| `CROP_GRAY` | L(회색) 조각. RGB로 복제 변환하지 않음 |

건너뛴 표본은 `readiness.coverage`에 split×class별 `samples / with_crop / usable / skipped{사유}`, 전체 `crop_coverage`(생산자 값), `model_input_coverage`로 남긴다. 원본 TIFF에서 `rect`로 다시 잘라 쓰는 경로는 이 저장소 범위가 아니다(원본 변환은 sic-xrt-data-tools 책임).

## 5. 준비도 gate

- 사용 가능한(usable) 표본 기준으로 4클래스 모두 split별 최소 개수를 요구한다. 기본 train 10 / validation 5 / test 5(생산자 `readiness`와 같은 값), `--min-per-class`로 변경 가능하며 값은 checkpoint·training-report에 기록된다.
- 각 split에 독립 그룹이 하나 이상 있어야 한다. `data_origin`에 `synthetic_fixture`가 있는 행이 하나라도 있으면 실제 학습을 거부한다. `--synthetic-smoke`는 모든 행이 `["synthetic_fixture"]`일 때만 허용되고 gate 기본값이 1이다.
- 생산자 manifest의 `readiness` 값은 참고로만 `producer_readiness`에 담는다(crop 누락을 고려하지 않으므로).

## 6. 학습·평가

- 헤드 크기는 데이터셋 클래스 수(4)를 따른다. 구조·전처리·optimizer·resume 규칙은 v1과 같다.
- 손실: 클래스 가중치 `n_train / (4 × n_class)`(usable train 기준)와 표본 합의 가중치 `weight`를 곱한 정규화 Cross Entropy. validation loss(모델 선택)는 클래스 가중치만 적용한다. v1 학습은 바뀌지 않는다(평균 CE).
- 지표: Accuracy, Balanced Accuracy, Macro F1, 클래스별 P/R/F1/Support(`NOT_DEFECT` 포함), 4×4 confusion, BPD→TED/TSD, false BPD, `NOT_DEFECT_as_defect`(배경을 결함으로), `defect_as_NOT_DEFECT`(결함을 배경으로). `random_sample=true` 표본만의 지표를 `random_sample_candidate`로 따로 낸다(편향 없는 추정용).
- 합성 자료 결과는 `scientific_metrics=false`이며 성능 주장이 아니다. 실제 v2 데이터로의 학습은 아직 NOT_RUN, 개선은 NOT_VERIFIED다.

## 7. 호환성

- `verified_xrt_dataset` v1 로더·학습·`frozen_xrt_patch_classifier` v1 export·registry는 그대로다.
- v1 checkpoint(`classes` 키 없음)는 3클래스로 읽는다. v2 checkpoint에는 `classes`, `dataset`(스키마·버전·id·해시), `class_weights`, `minimum_per_class`가 들어간다. resume은 클래스와(v2) gate가 같아야 한다.
- 생산자 계약이 바뀌면(`crop_contract`, `split_policy`, class mapping) 로더가 `MODEL_INPUT_CONTRACT`로 거부한다. 이때 양쪽 문서와 버전을 함께 올린다.
