# 개발 웨이퍼 교차검증

소비 계약은 Data Tools `wafer_grouped_development_cohort` v1이다. manifest와 전체
파일 해시를 확인하고 W1/2/9 이외의 row를 거부한다. source/point/patch 파일 해시가
다른 웨이퍼에서 중복되면 중단한다. W8 로딩·추론·재평가 경로는 없다.

세 웨이퍼를 차례로 보류한다. 같은 웨이퍼의 열처리 전후 영상·패치는 항상 같은
그룹이다. 학습 전 고정하는 설정은 seed 42, 12 epochs, AdamW 0.001, batch 32이며
마지막 epoch를 사용한다. 보류 웨이퍼로 epoch나 임계값을 선택하지 않는다.

비교할 세 recipe는 center32/역빈도 sampler, center32/제곱근 역빈도 sampler,
center32_context128/제곱근 역빈도 sampler이다. sampler는 그 fold의 학습 클래스
개수만 사용한다. 증강은 학습 입력에만 적용한다. 같은 코호트와 seed를 사용한다.

각 점은 해당 웨이퍼가 보류된 fold에서 한 번만 평가된다. 세 fold의 confusion을
합한 pooled OOF Macro F1로 설정을 고른다. **선택에 사용한 개발 점수이며 독립 시험
점수가 아니다.** W9에는 BPD가 없어 해당 fold의 BPD 재현율은 해석할 수 없고,
기존 3종 macro 계산은 support=0 클래스를 0으로 처리한다. 따라서 비교에는 통합
OOF 지표와 support를 사용하고, 웨이퍼별 표에는 정확도와 종류별 support를 표시한다.

첫 비교의 웨이퍼/열처리 조건 간 실패를 확인한 뒤 국소 명암 보정 두 설정을 별도 계획에
기록하고 추가했다. RGB 평균 회색조에서 reflect padding의 15픽셀 box 배경을 빼고,
잔차 RMS(최소 1/255)로 나눈 뒤 `clip(0.5+0.15*z,0,1)`을 세 채널로 반복한다.
원본 이미지를 변경하지 않는다. 중심 32와 중심+주변 두 모델 모두 같은 seed/epochs/
제곱근 sampler/그룹 분할을 사용한다. 이는 첫 결과를 본 이후의 적응적 개발 실험이며
그 사실을 `contrast_development` 계획·결과에 표시한다.

선택 설정은 전체 개발 자료로 다시 12 epochs 학습해 `development_model`에 저장한다.
로더는 `grouped_development.load_development_model`이다. 기존 raw 모델은 기존 로더로
위임하며 명암 보정 모델은 `NormalizedBPDContextCNN_v1`과 별도 전처리 계약을 요구한다.
호환성을 위해 `best_model.pt`라는
파일명을 쓰지만 실제로는 사전 고정된 마지막 epoch이다. 저장/재로드 예측 일치도
확인한다. 새로운 독립 시험을 통과한 모델로 표시하거나 자동으로 운영 모델을 바꾸지 않는다.

`plan.json`, fold별 config/history/heldout_predictions, recipe별 OOF predictions,
`comparison.json`, `result.json`은 로컬에 저장한다. 제공자 라벨과 AI 검수 이력을
전문가 정답으로 승격하지 않는다. BPD는 두 웨이퍼에만 있으며 seed도 하나이므로
개선은 제한된 개발 자료에서 얻은 증거로 해석한다.

이번 동일 코호트 OOF 비교에서 기존 중심/역빈도 기준은 정확도 69.68%, Macro F1
0.5358, BPD 정밀도 6.17%/재현율 89.34%였다. 선택된 명암 보정+문맥/제곱근 모델은
93.52%, 0.7796, 34.91%/59.90%였다. BPD 오탐은 2,676→220개로 줄었지만 놓친
BPD는 21→79개로 늘었다. 오류 감소와 누락 증가를 함께 보고해야 한다. 독립 시험
개선을 입증한 수치가 아니며 이전 W8 시험 95.83%와 직접 비교하지 않는다.
