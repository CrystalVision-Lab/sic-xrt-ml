# 검수 반영 데이터셋 재학습 실행

관련 Issue #11, 입력 생산자 [DataTools #17](https://github.com/CrystalVision-Lab/sic-xrt-data-tools/issues/17).
기존 schema-v1 로더와 SmallPatchCNN을 사용했다. 검수 보고서의 source_image는 기존 image 키,
또는 새 sources.json의 locator 키를 소비하므로 ZIP 내부 원본 경로도 표시할 수 있다.
생산자 모듈이나 원본/패치/모델은 이 저장소에 복사하지 않는다.

2026-10-03 실행 ID `20261003T143900Z_e46b7e2b`.
RTX3050, PyTorch2.9.1+cu128, seed42, batch32, AdamW lr0.001, 15epochs,
128픽셀 원본 dtype 정규화 후 float32 RGB CHW 입력. 리사이즈·증강·좌표 이동은 없다.
모델은 기존 SmallPatchCNN_v1_from_scratch이며 클래스 순서는 BPD/TED/TSD다.

| 자료 | BPD | TED | TSD | 합계 |
|---|---:|---:|---:|---:|
| 전체 학습 후보, 웨이퍼1·9 | 146 | 3,103 | 3,874 | 7,123 |
| 이번 균형 학습 뷰 | 146 | 146 | 146 | 438 |
| 검증, 웨이퍼2 | 33 | 257 | 70 | 360 |
| 예약 시험, 웨이퍼8 | 6 | 780 | 509 | 1,295 |

전체 후보8,778개 중 직접 AI 형태 판독은153개, 나머지8,625개는 자동 검사 통과 제공자 라벨 후보다.
연구용 라벨 상태는 research_pending_provider_and_AI_labels이며 전문가 확정은 아니다.
학습 후보 전체를 사용한 실행이 아니라 클래스 최소 개수에 맞춘438개를 사용한 기존 baseline이다.

최고 검증 epoch13, macro F1 `0.3634262177951499`, accuracy `0.5666666666666667`.
선택된 체크포인트로 검증360개를 다시 예측했고 혼동 행렬과 F1이 history와 일치했다.

| 타입 | 검증 F1 | recall | support |
|---|---:|---:|---:|
| BPD | 0.181818 | 0.272727 | 33 |
| TED | 0.714286 | 0.719844 | 257 |
| TSD | 0.194175 | 0.142857 | 70 |

혼동 행렬(행=제공 라벨, 열=예측, BPD/TED/TSD):

```text
9   19   5
54 185  18
3   57  10
```

후반에 학습 loss는 줄지만 검증 loss가 커지며, BPD와 TSD 성능이 낮다.
현재 모델을 확정 분류기로 승격하지 않는다. 제공 라벨의 정확도를 이 지표로 입증할 수 없다.
기존 실행과 검증 대상이 달라 점수 차이를 데이터 청결도 개선의 인과 효과로 해석하지 않는다.
동일 검증셋에서 최고epoch을 선택했으므로 독립 일반화 성능도 아니다.
웨이퍼8 시험 이미지는 예측하지 않았으며 test_metrics.json도 생성하지 않았다.
웨이퍼3~7 주석, 세부 a~f/a~c 기준, 전후 동일 결함 대응과 3D 깊이 정답은 아직 확보되지 않았다.

재현 실행(환경별 경로 지정):

```powershell
python -m sic_xrt_ml.training.patch_classifier --dataset DATASET --outputs NEW_RUNS --epochs 15 --batch-size 32 --device cuda
```

기본 lr0.001, size128과 균형 학습을 유지한다. 완료 후 별도 validation_review 보고서를 생성한다.
자료가 동일한지는 config.json의 manifest_sha256/train_csv_sha256과 생산자 provenance를 비교한다.
원본/라벨/기존모델을 보존하고 매 실행을 새로운 run 디렉터리에 저장한다.

이번 samples.csv SHA-256:
`bcc39e24d5d46650525b08a1d046c4ea26d8f146c50799c9047ae30117ddedb0`.
