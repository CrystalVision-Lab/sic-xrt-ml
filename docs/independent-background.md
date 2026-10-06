# 기존 종류 분류를 유지하는 독립 배경 판별

DataTools [#29](https://github.com/CrystalVision-Lab/sic-xrt-data-tools/issues/29)의 `xrt_detection_observations_v1`을 소비한다. 세 종류 분류기의 학습/가중치/예측을 수정하지 않고, 별도 이진 모델이 결함 가능성 점수 하나를 출력한다. 입력은 uint8 RGB128, 모델 계약은 `independent_background_v1_rgb128_uint8_defect_probability`다. 회색조·국소 정규화·절대 대비를 함께 사용해 중심64px와 문맥128px를 본다.

실행: `python -m sic_xrt_ml.training.independent_background --cohort ... --background .../weak_background.json --baseline ..._oof.json --observations .../observations.json --output 새폴더`

입력 패치 해시와 AI 출처를 검증한다. weak_background만 배경 학습에 추가한다. possible_defect/uncertain은 정답으로 훈련하지 않고, 해당 웨이퍼 제외 모델의 별도 관찰 표본으로 평가한다. 원래 종류 라벨과 모델 출력은 보존한다. 기본 16epoch·seed42·웨이퍼 1/2/9 제외 학습이며 계층(BPD/TED/TSD/무작위배경/실제후보배경)을 균등 샘플링한다. 데이터 증강은 이진 분류에만 적용한다.

학습 전 plan.json에 기준을 기록한다. 배경 점수≥0.95는 background_suspect, 결함 점수≥0.8은 defect_candidate, 나머지는 uncertain이다. uncertain을 자동 제거하지 않는다. 각 웨이퍼의 제공 결함 보존 99% 이상, 각 타입 보존 98% 이상, 관찰상 결함 가능성 보존 98% 이상, 지원되는 배경 출처마다 70% 이상 제거를 요구한다. 실제 후보 배경이 2개 이상 웨이퍼에 존재하고 전체 이미지 검증까지 통과해야 배포를 검토할 수 있다.

결과의 보존율은 제공 라벨과 AI 관찰에 대한 수치이지 전문가 정확도가 아니다. 실제 후보 200개는 높은 기존 점수로 선택한 편향 표본이며 중복 물체를 포함할 수 있다. 최종 시험 웨이퍼 8은 읽지 않는다. runner는 ONNX export나 Analyzer 교체를 수행하지 않는다. 기존 3클래스 모델 자리에 이 체크포인트를 넣으면 안 된다. 기준 실패 시 not_promoted를 유지한다.
