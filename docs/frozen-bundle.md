# 고정 연구 모델 묶음 v1

`export.frozen_bundle.freeze_bundle`은 기존 중심64 연구 후보를 학습 없이 ONNX로 고정합니다.
이미 있는 출력 폴더를 덮어쓰지 않으며 검증 실패 시 완료 manifest를 쓰지 않습니다.
성공한 `manifest.json`과 `model.onnx`만 있으면 소비자는 ML 저장소 소스 없이 실행할 수 있습니다.

계약 이름은 `frozen_xrt_patch_classifier`, schema_version 1입니다.
입력 `images`는 float32 N×3×128×128 RGB/255이며 명암 보정은 모델 안에 있습니다.
출력 `probabilities`는 N×3, 순서는 BPD/TED/TSD입니다. 점수는 확률 보정되지 않았습니다.
배경 클래스와 위치 검출 기능이 없으므로 모든 패치를 확정 결함으로 세면 안 됩니다.
종류별 개수는 제공된 점 또는 별도 후보 검출기의 결과에 의존합니다.

소비자는 manifest의 버전·상태·입출력 계약·클래스·모델 SHA256을 확인해야 합니다.
`research_only=true`, `independent_test_passed=false`를 결과에 유지합니다. 여기서 검증된 것은
내보내기 계산의 일치이며 전문가 승인/성능 인증이 아닙니다. ONNX는 이후 고정됩니다.
manifest는 원래 후보 포인터/구성 모델 해시, 데이터 manifest, 코드 리비전과 계산 오차를 기록합니다.

회전은 transpose/flip으로 표현합니다. RGB128과 중심64로 한정하므로 두 특징 분기의
8×8→4×4 adaptive pool을 동등한 2×2 평균 풀링으로 바꾸어 내보냅니다.
변환 전후 PyTorch 확률 일치, ONNX 구조, CPU 런타임 배치1/7/전체 실제 개발 probe의
최대 오차(1e-4 이하)와 클래스 완전 일치를 확인한 후에만 완료 상태를 기록합니다.

실행 의존성은 `[frozen-export]`입니다. 소비자 Analyzer는 별도 Issue/PR에서 이 파일 계약만
구현하며 저장소 간 소스 경로 import를 사용하지 않습니다. 실제 모델·검증 패치는 Git 제외입니다.

내보내기 API 참고: https://docs.pytorch.org/docs/stable/onnx_export.html
