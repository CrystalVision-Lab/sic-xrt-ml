# 전체 영상 후보의 위치·종류 분리 평가

`python -m sic_xrt_ml.evaluation.localization_audit input.json report.json --radius 20`

입력 계약 `xrt_point_audit_inputs_v1`은 `split: development`, `reference`와 `prediction` 객체를 포함합니다. 두 객체 모두 동일한 `image_sha256`(소문자 64자리), `coordinate_space: raw_pixel_xy`, `rows`를 명시해야 합니다. 각 행은 유일한 문자열 `id`, 유한한 `x`, `y`, 선택적 `type`을 가집니다. `provenance`에 좌표 변환 근거·라벨 출처·모델 해시를 넣으면 보고서에 그대로 기록됩니다. 이미지 해시는 사전에 실제 파일과 검증해야 하며, 평가기는 서로 다른 영상/좌표계 선언의 결합을 거부합니다. 원본·주석·모델 파일은 Git에 넣지 않습니다.

거리 제한 이내에서 **최대 개수의 1대1 대응**을 찾습니다. ID 순서, 거리순 이웃 탐색을 사용하는 증대 경로 방식이며 최소 총거리 최적화는 아닙니다. 종류 라벨을 대응 결정에 사용하지 않습니다. 복수 후보/복수 주석이 엮인 모호한 대응도 기록하므로 해당 쌍의 종류 비교는 별도 검토해야 합니다. 반경은 픽셀 기준이며 보정되지 않은 µm 값으로 해석하지 않습니다. 10·20·40픽셀 등의 민감도도 함께 보고합니다.

- `reference_point_coverage`: 제공 주석 중 1대1 대응된 점의 비율. 모든 물리적 결함에 대한 재현율이 아닙니다.
- `unmatched_without_reference_neighbor`: 반경 내 제공 주석이 없는 후보. 배경/오탐 정답이 아닙니다.
- `unmatched_competing_for_reference`: 가까운 주석은 있으나 다른 후보와 경쟁해 미대응된 후보. 중복 의심 대상이며 실제 중복 판정은 아닙니다.
- `coverage_by_reference_type`: 제공 라벨별 위치 대응. 종류 예측 점수와 독립적으로 계산합니다.
- `matched_type_agreement`: 대응 쌍 중 BPD/TED/TSD 라벨이 있는 쌍의 종류 일치. 검수 미확정 제공 라벨과의 비교이며 독립 시험 정확도가 아닙니다.

출력 `xrt_point_audit_v1`에는 대응 ID·거리·모호성, 미대응 ID, 혼동행렬, 입력 SHA256을 포함합니다. 기존 보고서를 덮어쓰지 않습니다. 제공 주석의 완전성/전문가 검수가 확인되지 않은 상태에서 precision, F1, 배경 정답을 생성하지 않습니다. CLI는 final_test 입력을 거부합니다. 최종 시험 자료로 탐색 설정을 선택하지 마세요.

Analyzer의 외부 좌표/예측 산출물만 소비합니다. Analyzer 내부 모듈을 import하거나 데이터를 변환하는 기능은 없습니다. 탐색 파라미터 비교는 연구 실험이며, 검증되지 않은 필터를 배포하거나 고정 모델을 자동 교체하지 않습니다.
