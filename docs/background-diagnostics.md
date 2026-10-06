# 배경 판별 실패 원인 진단

`python -m sic_xrt_ml.evaluation.background_diagnostics --cohort ... --background ... --baseline ... --observations ... --experiment 기존실험 --output 새폴더`

기존 독립 판별기 체크포인트를 고정하고 같은 해시의 입력으로 학습 자료와 제외 웨이퍼 자료의 점수 분포를 비교한다. 학습을 새로 하지 않는다. 기존 OOF 점수가 1e-5 이내에서 재현돼야 결과를 저장한다. actual candidate possible_defect/uncertain은 학습 자료로 계산하지 않는다. 최종 시험 웨이퍼 8은 읽지 않는다.

진단 항목은 출처·타입별 점수 분위수, 기존 0.95 배경 기준의 제거 수, 제외 자료를 알고 있다고 가정한 낙관적 구분 한계다. 마지막 항목은 배포용 임계값 선택이 아니다. 제공 결함 전체 99%·각 타입 98%·결함 가능성 98%를 보존하는 제약에서 최대 배경 제거가 얼마인지 계산해, 단일 임계값 조정만으로 해결 가능한지 반증하는 데 사용한다. 정렬 점수의 동률과 >= 경계는 nextafter로 처리한다.

한계 수치는 평가 라벨을 보고 계산한 진단 전용 값이며, 검증 성능·새 임계값·운영 정책으로 내보내지 않는다. 기존 프로그램 모델과 임계값을 바꾸지 않는다. 자료에 대한 원본 ROI/좌표 대응 검사는 [DataTools #31](https://github.com/CrystalVision-Lab/sic-xrt-data-tools/issues/31)의 책임이다.
