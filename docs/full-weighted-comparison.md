# 전체 후보 가중 학습과 동일 검증 비교

관련 Issue #13. 원본 오류 진단은 [DataTools #19](https://github.com/CrystalVision-Lab/sic-xrt-data-tools/issues/19),
입력은 기존 schema-v1 연구 후보 dataset으로 고정했다. 생산자 코드를 import/복사하지 않는다.

`train(..., balanced=False, class_weighting='inverse_frequency')`는 학습7,123개 전부를
매epoch 한 번씩 사용한다. class_weights는 선택된 학습 행의 빈도에서만 N/(3*n_c)로 계산한다.
검증·시험 빈도나 예측으로 가중치를 정하지 않는다. 균형 downsampling과 가중 손실을 함께 지정하면 거절한다.
기본값 balanced=True/class_weighting=none은 기존438개 균형 baseline의 동작을 유지한다.

이번 학습 빈도 BPD146/TED3,103/TSD3,874에 따른 가중치는
`[16.26255707762557, 0.765173488022344, 0.6128893477886767]`이다.
미니배치 최적화는 가중 cross entropy 평균을 사용하며 epoch 손실은
모든 loss 합/모든 target weight 합으로 기록한다. 배치 평균들을 단순 표본 수로 평균하지 않는다.
검증 loss와 F1/정확도는 가중하지 않는다. 가중 학습 loss와 검증 loss는 서로 다른 집계라는 점을 기록한다.

config에 class_weighting/class_weights/class_weight_basis, validation_loss_weighted=false,
training_loss_reduction, optimizer_steps_per_epoch를 추가한다. 모델과 입력 계약은 바꾸지 않아
기존 평가·검수 로더에서 새 체크포인트를 사용할 수 있다. centre crop 비교도 가중 정책을 전달한다.

`full_training_comparison`은 이전 완료 balanced 모델·checkpoint·history·validation CSV를
입력 dataset과 검증한 후, 같은360개 검증 행의 원래 라벨/경로/해시를 고정한다.
실행 전에 plan.json에 가중치, 입력·검증 행 해시, 보호할 파일 해시와 비교 조건을 기록한다.
전체 후보 학습을 새 run에 실행하고 최고 체크포인트의 val 예측을 다시 생성해 저장된 혼동 행렬과 비교한다.
모든 epoch에서 학습7,123개/검증360개를 사용했는지 확인하고 이전 결과 및 입력 기록의 해시 보존을 검사한다.
시험 이미지는 예측하지 않는다. 모델을 자동으로 교체하거나 정답 라벨을 수정하지 않는다.

2026-10-04 실제 RTX3050 실행 `20261004T120428Z_2c372b4e`:
PyTorch2.9.1+cu128, seed42, batch32, lr0.001, 15epoch, 원본128픽셀, SmallPatchCNN from scratch.
학습 웨이퍼1·9, 검증2, 예약 시험8. 증강/리사이즈/좌표 이동/자료 재선별을 추가하지 않았다.

| 지표 | 기존 균형438 | 전체 후보7,123+가중 손실 |
|---|---:|---:|
| 최고epoch | 13 | 14 |
| 검증 Macro F1 | 0.363426 | 0.437805 |
| 검증 accuracy | 0.566667 | 0.691667 |
| BPD F1 | 0.181818 | 0.294118 |
| TED F1 | 0.714286 | 0.817052 |
| TSD F1 | 0.194175 | 0.202247 |
| TSD recall | 0.142857 | 0.128571 |
| TSD→TED | 57/70 | 56/70 |
| optimizer steps/epoch | 14 | 223 |

검증 라벨 순서 BPD/TED/TSD, 새 혼동 행렬(행=제공 라벨, 열=예측):

```text
10  20   3
20 230   7
5   56   9
```

동일 사례에서 이전 오류→새 정답50개, 이전 정답→새 오류5개였다.
전체 Macro F1은0.074379 올랐으나 TSD recall은 감소했고 TSD/TED 혼동은 거의 해결되지 않았다.
검증 loss가 높고 회차별 지표 변동이 있어 확정 분류기로 승격하지 않는다.
57개 오류 진단은 지정 좌표 형상/인접 대상 관계가 모호해 보류했으며 검증 라벨에는 반영하지 않았다.
제공 라벨에 대한 탐색 검증이며 전문가 확인 정답에 대한 성능은 아니다.
학습량·손실 가중치·전체optimizer업데이트 수가 함께 바뀌어 각각의 효과를 분리해 입증하지 않는다.
같은 검증 웨이퍼에서 최고epoch을 선택했으므로 독립 일반화 성능으로 해석하지 않는다.

입력 samples.csv SHA-256:
`bcc39e24d5d46650525b08a1d046c4ea26d8f146c50799c9047ae30117ddedb0`.
검증360개 canonical 행 SHA-256:
`a57419c5f1118d109db80b6de303fe0dd92e492e558933288280fb34ad4194b7`.

출력: plan.json, 독립runs의config/history/checkpoint/result,
comparison.json/CSV/PNG, paired_validation_predictions.csv와 검증 예측 보고서.
실제 이미지·모델·좌표별 예측은 Git에 저장하지 않는다.

```powershell
python -m sic_xrt_ml.training.full_training_comparison --dataset DATASET --baseline OLD_RUN --baseline-review OLD_VAL_REVIEW --outputs NEW_OUTPUTS --device cuda
```

CLI 단일 학습도 `--unbalanced --class-weighting inverse_frequency`로 전체 행과 가중 손실을 지정할 수 있다.
기존 체크포인트의 class_weighting 필드 누락은 none으로 해석한다. CUDA 완전 결정성은 보장하지 않는다.
