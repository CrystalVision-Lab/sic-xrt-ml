# 약지도 배경 및 보류 실험

`python -m sic_xrt_ml.training.background_abstention --cohort ... --background .../weak_background.json --baseline ..._oof.json --output 새폴더`

DataTools의 xrt_weak_background_v1와 해시가 같은 개발 코호트를 읽는다. 패치 파일의 해시·형상·dtype·경로를 확인하며, 1/2/9 외 웨이퍼는 제외 후보에도 허용하지 않는다. 기존 라벨을 수정하지 않는다. AI 배경 후보만 사용하며 uncertain/artifact_candidate는 제외한다. 배경은 결함 부재를 입증한 정답이 아니고 expert/human_verified=false를 강제한다.

각 웨이퍼를 통째로 제외해 12epoch 고정 학습한다. 제외한 웨이퍼로 epoch/threshold를 고르지 않는다. 64px 중심과 128px 문맥에 국소 대비 정규화를 적용한 별도 4클래스 CNN이다. 데이터 증강은 거친 3종 분류용으로만 적용한다. 방향 세부타입의 정답을 학습하지 않는다. 이전 OOF는 동일 패치 ID/라벨/웨이퍼인지 확인한다. 이전 모델과 아키텍처가 다르므로 배경 추가만의 인과 효과를 주장하지 않는다.

실험 전 plan.json에 임계값과 통과 조건을 저장한다. 배경 점수 ≥0.95일 때 제거하고, 종류 점수 ≥0.8 및 배경 점수 ≤0.2인 경우만 종류 확정 후보로 본다. 조건부 종류 점수와 배경 점수를 혼동하지 않는다. 배경 제거율, 제공 결함의 잘못된 제거율, 종류 혼동, 보류 비율을 별도로 기록한다. BPD 표본이 없는 웨이퍼의 BPD 재현율은 null이다.

각 개발 웨이퍼에서 전체 결함 99% 이상·각 타입 98% 이상 보존, 약지도 배경 70% 이상 제거, 기존 OOF 대비 macro F1/BPD TP 비감소·BPD FP 비증가를 요구한다. 한 항목이라도 실패하면 교체하지 않는다. 통과해도 전체 이미지 검출 및 전문가 평가 전에는 배포하지 않는다. 최종 시험 웨이퍼 8은 읽지 않는다.

체크포인트는 PyTorch state_dict이며 계약 `background_context_cnn_v1_rgb128_uint8_gray_local_box15_center64_context128`, 출력 순서 BPD/TED/TSD/BACKGROUND다. 기존 Analyzer의 3클래스 ONNX 계약과 호환되지 않으며 이 runner는 export/교체를 수행하지 않는다. 데이터·체크포인트는 Git 밖의 지정 결과 폴더에만 저장한다.
