# TSD 혼동의 자료 분포와 중심 입력 진단

Issue #15, PR #14의 전체 후보 가중 학습을 기준으로 한다.
`training.tsd_diagnostics`는 ML이 소비하는 schema-v1 입력을 읽으며 생산자 코드를 import/복사하지 않는다.
데이터·주석·좌표·원본 모델을 변경하지 않는다. 실제 이미지·진단 CSV·모델은 Git에 넣지 않는다.

입력 검증: 완료한 full/inverse-frequency baseline의 manifest, train CSV, 클래스 빈도,
웨이퍼 분리, PyTorch/device, train-only 가중치, checkpoint/config/최고회차 지표를 확인한다.
baseline val 예측 CSV의 모든 canonical 필드와 checkpoint/manifest 해시를 확인하고 혼동 행렬을 재계산한다.
변경·누락·중복 행이나 맞지 않는 checkpoint를 거절한다.

실행 전에 두 crop 크기(64·32), 고정 검증 행 해시, baseline 파일 해시, 제한을 plan.json에 기록한다.
train/val 7,483개 패치만 해시·shape·dtype를 확인하고 unit-range 픽셀 통계를 추출한다.
시험 TIFF는 디코딩/예측하지 않는다. 진단 feature는 모델에 추가하지 않는다.

## 픽셀 기술 통계

원래 학습 정규화(uint8/255, uint16/65535)로 RGB 평균을 만든다. 중앙16픽셀 영역과 나머지 배경에서
배경 median/std, 중앙 q99−배경 q99, 중앙 mean−배경 median을 기록한다.
3×3 평균 밝기가 가장 높은 위치와 패치 중심 사이의 픽셀 거리를 기록한다.
작은 테스트 입력에서는 중앙 폭을 size//2로 줄인다. 밝은 위치는 지정 결함 위치를 의미하지 않고,
통계로 TSD/TED 물리 타입이나 제공자 라벨 오류를 확정하지 않는다.
split/제공 기본·세부 라벨/촬영 시점별 q25·median·q75를 저장한다.

## 고정 비교

중앙64·32픽셀을 리사이즈 없이 잘라 별도 모델을 각각 처음부터 학습한다.
학습 7,123개, 검증360개, 클래스 가중치, 15회, seed42, batch32, lr0.001, 동일 CNN을 고정한다.
crop은 문맥과 CNN pooling의 상대적 공간 해상도를 함께 바꾸므로 배경 효과만을 분리한 인과 실험은 아니다.
각 후보의 완료 checkpoint와 모든 회차의 train/val 개수를 재검사한다.
최고회차의 val 예측을 다시 생성하고 원래360개 행/라벨/좌표/해시와 지표 일치를 확인한다.
기존 baseline과 데이터셋 기록의 해시 보존을 확인한다.

출력은 plan.json, input_diagnostics.json, pixel_features.csv, pixel_descriptors.png,
고정 중심 실험 모델·검증 보고서, paired_validation_predictions.csv, diagnosis.json이다.
제공 세부 라벨별 혼동 행렬과 재현율도 기록한다. subtype만 포함한 그룹의 Macro F1은 빈 다른 클래스를
포함하므로 그룹 비교에는 support/해당 클래스 recall을 사용한다.
의미 검수 상태는 연구 후보로 유지하고 모델을 자동 승격하지 않는다.

## 이미 확인한 자료 분포

학습 웨이퍼1·9에는 전후 원본7개, 검증 웨이퍼2에는 애널링 전 원본1개가 있다.
학습 TSD_a 1,857 / TSD_b 2,011 / TSD_c 6, 검증 TSD_a 70이다.
학습 TED_e 22 / TED_f 7, 검증 TED_e 26 / TED_f 20이다.
기본3종 학습에서 세부 라벨은 새 정답으로 예측하지 않고 분포·오류 그룹을 설명하는 데만 사용한다.
검증 조건이 제한되어 있고 희귀 세부 타입이 부족하다. 이 사실만으로 촬영 조건 차이가 오류 원인이라고
입증하거나 웨이퍼별 효과·촬영 시점별 효과를 분리할 수 없다.

## 2026-10-04 실제 GPU 결과

RTX3050 / PyTorch2.9.1+cu128에서 사전 계획한 두 입력을 각각15회 완료했다.
baseline 20261004T120428Z_2c372b4e, 64픽셀 20261004T141424Z_b2cda206,
32픽셀 20261004T141829Z_b21ecf5e. 동일360개 행 해시:
`a57419c5f1118d109db80b6de303fe0dd92e492e558933288280fb34ad4194b7`.

| 입력 | 최고회차 | Macro F1 | BPD F1 | TED F1 | TSD F1 | TSD 일치 | TSD→TED |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 14 | 0.437805 | 0.294118 | 0.817052 | 0.202247 | 9/70 | 56/70 |
| 64 | 11 | 0.506727 | 0.500000 | 0.847341 | 0.172840 | 7/70 | 61/70 |
| 32 | 12 | 0.523760 | 0.509804 | 0.863946 | 0.197531 | 8/70 | 58/70 |

중심 입력의 전체 지표 개선은 BPD/TED에서 나왔다. TSD 재현율과 F1은 두 후보 모두 개선되지 않았다.
전체 점수가 가장 높은32픽셀 모델을 TSD 문제 해결 모델로 승격하지 않는다.
중심 문맥만 줄이는 개선으로 TSD 혼동을 해결했다는 가설을 뒷받침하지 못했다.

제공 TSD_a 중 train-before812개의 중앙 q99−배경 q99 중앙값0.467255,
train-after1,045개는0.313595였다. baseline이 TED로 예측한 val TSD_a56개는−0.001699,
TSD로 예측한9개는0.398915였다. 가장 밝은3×3의 중심 거리 중앙값은 각각
4.30/4.74/53.00/9.82픽셀이다. 배경 median은 train-before0.338562,
val→TED0.364052로 중심 신호 차이에 비해 가깝다.
이 수치는 제공 라벨별 입력 모양 차이에 대한 기술 통계이며 잘못된 라벨/좌표라는 증명은 아니다.

기존 생산자 산출물을 읽어보면 웨이퍼2 제공 TSD_a844개 중 현재 검증에는70개만 포함된다.
제외 기록의 중심 대비 경고755개/대비 위치 경고479개는 중복되는 사유다.
TSD_b7개/TSD_c2개는 전부 제외됐다. 따라서 현재 지표는 선별된 후보 집단의 결과다.
기존 주석의 image mapping 근거는 같은 폴더·촬영 시점의 TIFF 대응이며
`candidate_pending_visual_review` 상태다. 파일/해시/좌표 검증과 의미상 정합성은 별개다.
다음 데이터 정합성 점검의 우선순위는 웨이퍼2 TSD(a) ROI와 연결 TIFF의 실제 원본 대응 확인이다.
현재 단계에서는 제공 라벨·좌표·고정 검증 집단을 보존한다.

```powershell
python -m sic_xrt_ml.training.tsd_diagnostics --dataset DATASET --baseline FULL_WEIGHTED_RUN --review BASELINE_VAL_REVIEW --outputs NEW_OUTPUTS --device cuda
```

제공 라벨은 전문가 확정 정답이 아니다. 이미 재사용한 검증 자료에 대한 탐색 실험이며
시험 웨이퍼8의 독립 성능이나 subtype 정확도·물리 정답을 주장하지 않는다.
CUDA 완전 결정성은 보장하지 않는다.
