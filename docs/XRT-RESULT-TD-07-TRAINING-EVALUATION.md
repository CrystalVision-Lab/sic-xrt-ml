# XRT-RESULT-TD-07 — Training & Evaluation

## Overall Status

엔지니어링 검증 진행 중. 실제 정답 DATA_NOT_READY. TRAINING_PIPELINE_VERIFIED는 합성 자료로 학습 코드가 동작한다는 의미다. 실제 학습 NOT_RUN / 독립 평가 NOT_AVAILABLE / 성능 개선 NOT_VERIFIED. Detector·Segmentation 개선을 주장하지 않는다.

## Git / PR

시작 원격 main `cc0cbabb6640ddd2899ecee9b9195ee450f45f4f`를 확인했다. Issue #45, branch feat/45-training-evaluation-pipeline. 로컬의 별도 미병합 feat/43-human-feedback-training@446d7fb는 보존하고 독립 worktree에서 처음부터 구현했다. Analyzer Issue #108의 승인 데이터셋 v1 Artifact만 소비하며 소스를 복사하지 않는다. Draft PR/최종 CI는 검증 완료 후 기록한다.

## Current Ground Truth Availability / Dataset Eligibility

Analyzer localhost API에서 기존 7,474 observations / 16 reviews / Crop 2개, 실제 승인 0개를 재확인했다. 승인·그룹·사용 권한이 확인되지 않은 QA 자료를 실제 정답으로 쓰지 않는다. 후보 수를 정답 수로 집계하지 않는다. 실제 학습은 사전에 정의한 train 10/class, validation 5/class, test 5/class와 각 split 독립 그룹을 모두 요구한다. 이는 최소 실행 gate이며 통계적 표본 충분성 보장이 아니다.

## Adjudication / Versioning / Source & Label Provenance

Analyzer ADMIN의 append-only 승인과 Group Mapping이 선행되어야 한다. version 불변 Manifest, dataset/file SHA, 승인·검수·관리자·시각·사유·원본 source hash·ModelProvenance·원본 crop 좌표를 검사한다. Source hash는 원본 전체를 갖고 있지 않은 서버의 intake identity 증거이며 촬영 조건·전문가 정답의 독립 확인을 대신하지 않는다. 오프라인 Artifact의 철회 여부는 자동 조회하지 않으므로 운영자가 학습 직전 서버의 usage_status=AVAILABLE을 확인해야 한다. RESTRICTED 사본은 사용을 중단한다.

## Image Integrity / Input Contract

`verified_xrt_dataset` schema_version 1. Manifest canonical UTF-8 JSON은 sort_keys=true, ensure_ascii=false, separators=(",",":"), allow_nan=false이며 dataset_hash만 제외한 전체 SHA로 검증한다. file inventory와 split JSONL·eligibility-report가 Manifest와 일치해야 한다. PNG SHA, decoded RGB pixel SHA, dHash, 크기·dtype·원본 좌표, 승인 ID·revision·leaf hash를 검증한다. 중복 JSON key, path escape, symlink/junction, 비참조 파일, 과대 파일, UNKNOWN/BACKGROUND와 잘못된 클래스 순서를 거부한다. uint16 TIFF를 RGB로 변환하거나 학습에 수집하지 않는다.

클래스 0 BPD / 1 TED / 2 TSD. RGB8 정확히 128×128 → float32 NCHW /255. Resize·Center Crop·EXIF 회전·Flip·Rotation·밝기 Augmentation은 하지 않는다. train augmentation=none만 허용한다. 모델 안에 9×9 local mean/variance 정규화(epsilon 1e-4)를 실제 구현했다.

## Dataset Split / Leakage / Wafer Holdout

Producer가 동일 source hash·Wafer·결함 target·Run·촬영·전후·제공자·중복 그룹을 union하여 seed로 결정적 분할한다. ML은 각 관계의 train/validation/test 교차를 다시 거부한다. dHash Hamming ≤4 유사 패치의 교차도 거부한다. 이는 완전한 유사 이미지 검출기가 아니며 검증된 Mapping과 인간 확인이 여전히 필요하다. 정규 십진 wafer_id `8` 또는 evaluation_only 자료는 test 전용이다. 원본 파일명으로 Wafer를 추론하지 않는다. Test는 학습·validation·early stopping에 쓰지 않는다.

## Training Architecture

새 baseline `xrt_local_contrast_cnn/v1`: 내부 local contrast → Conv(3→8,5×5,stride2)+ReLU → Conv(8→16,3×3,stride2)+ReLU → Conv(16→32,3×3,stride2)+ReLU → GlobalAveragePool → Linear 3 logits. CrossEntropyLoss로 학습하고 ONNX는 softmax한 uncalibrated_class_scores를 출력한다. 기존 ONNX의 원본 PyTorch 구조·학습 상태를 복원한 모델이 아니며 기존 ONNX fine-tuning도 아니다.

## Training Configuration / Resources

기본 seed 2026, CPU threads2, batch8, AdamW(lr0.001,weight_decay0.0001), StepLR(step5,gamma0.5), epochs10, patience3, validation loss 최소 모델 선택. Batch 최대256 / threads 최대8 / Dataset 최대4,096. DataLoader workers0로 추가 worker·메모리 복제를 제한한다. 선택적 device=cuda는 실제 CUDA 가용성을 확인하며 자동 GPU 전환하지 않는다. 학습은 Docker API 밖의 별도 프로세스다. 모델·체크포인트·데이터는 ignored artifacts에만 저장한다. CPU 대규모 학습 시간과 타 기기 bitwise 동일 결과를 보장하지 않는다. 디스크·가용 RAM을 확인하고 epoch checkpoint 이후 중단/재개한다.

측정 PC: Windows Python3.13.7, PyTorch2.9.1+cu128, ONNX1.23.1, ORT1.30.0, CPU24 threads, RAM31.7GiB(확인 당시 가용5.1GiB), RTX3050 8GiB, D 여유306.7GiB. 기능 검증은 CPU2 threads를 사용한다. CUDA 실학습 검증은 별도다.

## Checkpoint / Resume

checkpoint.pt는 architecture/dataset/config/code revision, model·best model, optimizer·scheduler, epoch·validation history·bad epoch, Python/NumPy/Torch/CUDA/loader RNG를 저장한다. weights_only=True로 executable pickle을 거부한다. 재개는 같은 Dataset hash·설정(epochs 연장만 허용)으로 새 output directory에서 epoch 경계부터 수행한다. early-stopped run과 최종 test를 사용한 run은 재개하지 못한다. 중간 batch 재개는 지원하지 않는다. Checkpoint 교체는 임시 파일→rename을 사용한다. 정확한 2-epoch CPU 연속 실행과 1+1 재개 결과의 history·모든 weight 동일성을 검사한다.

```powershell
.\.venv\Scripts\python.exe -m sic_xrt_ml train <dataset> --config <extended-config.json> --resume <previous-run>/checkpoint.pt --output artifacts/<new-resumed-run>
```

## Evaluation / Existing Model Comparison

Accuracy, Balanced Accuracy, Macro F1, 클래스별 Precision/Recall/F1/Support, 3×3 confusion matrix, BPD→TED/TSD 오류, false BPD를 계산한다. 없는 클래스의 지표와 전체 Macro/Balanced 지표는 N/A(null)이며 100%로 채우지 않는다. 명시적 evaluate 단계에서만 test를 읽으며 test 접근 이력을 run/final-evaluation.json에 독점 생성한다. 같은 run의 반복 평가·평가 이후 resume를 거부한다. 운영자는 서로 다른 run의 반복 test 튜닝도 금지해야 한다(로컬 파일 guard는 조직 차원의 접근 통제를 대신하지 않는다).

--existing-bundle을 지정하면 기존 frozen_xrt_patch_classifier/v1 Manifest와 ONNX hash·실제 입출력 shape/name/dtype를 검사한다. 동일한 test sample/label/input을 candidate와 기존 CPU ORT에 전달한다. baseline 미지정은 N/A, 합성 비교는 scientific_metrics=false다. 실제 독립 정답이 없으므로 accuracy 개선을 주장하지 않는다. 기존 모델 출력 점수 자체는 정확도가 아니다.

## ONNX Export / Contract Compatibility / Candidate Registry

명시적 export는 선택된 best checkpoint와 연결된 evaluation SHA를 확인하고 새 폴더에 model.onnx, manifest.json, candidate-registry.json을 만든다. ONNX checker와 CPUExecutionProvider에서 batch1/7 dynamic N parity(atol1e-5,rtol1e-4)를 검사한다. 내부 local contrast가 실제 graph에 포함되며 클래스 순서·float32 [N,3,128,128] images와 [N,3] probabilities 계약을 유지한다. 합성 후보는 FUNCTIONAL_TEST_ONLY, 실제 후보도 AWAITING_EXPERT_MODEL_ACCEPTANCE/research_only=true/independent_test_passed=false로 기록한다. 데이터·설정·code·checkpoint·evaluation·model·Manifest SHA를 연결한다. 자동 배포·기존 모델 교체·독립 테스트 합격 선언은 없다.

## ML Tests / CI

작은 generated synthetic_fixture만 사용한다. 로더·해시·PNG·누수·입력·라벨·권한·원본 좌표·동적 batch·forward/backprop·optimizer·체크포인트·정확한 resume·early stop·N/A 지표·ONNX checker·ORT parity·잘못된 모델 계약·같은 평가 자료 비교를 검증한다. CI는 Linux Python3.11와 torch2.9.1 CPU wheel을 사용한다. 합성 Fixture와 모델 파일은 실행 중 생성하며 Git/CI에 실제 연구자료를 올리지 않는다. TorchScript exporter의 deprecation warning 두 건을 기록하며 오류를 숨기지 않는다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Scientific Validation Status / Known Limitations / Next TD

실제 정답 DATA_NOT_READY, 실제 학습 NOT_RUN, 독립 평가 NOT_AVAILABLE, 성능 개선 NOT_VERIFIED. 기능 검증이 실데이터 accuracy를 증명하지 않는다. 실제 사용권·전문가 승인·검증된 독립 그룹·충분한 클래스 표본을 수집한 뒤 새 immutable Dataset을 만든다. 동일 독립 test에서 기존 모델과 비교하고 전문가가 후보를 승인해야 TD08 안전 배포를 진행할 수 있다. 원본 PyTorch checkpoint 호환성은 미확인이다. Augmentation의 물리적 타당성·검사기/Segmentation·3D 재구성은 이번 범위에 없다.
