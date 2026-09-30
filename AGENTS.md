# sic-xrt-ml Agent 규칙

작업 전 [중앙 handbook](https://github.com/CrystalVision-Lab/engineering-handbook)의 AI_AGENT_RULES.md와 REPOSITORY_BOUNDARIES.md를 읽는다. 독립 체크아웃에서도 아래 규칙은 적용된다.

- 이 저장소는 데이터 로더, 전처리, PyTorch 학습, 평가, ONNX export, 모델 계약만 소유한다. Analyzer UI와 Data Tools 변환 코드는 각 저장소의 책임이다.
- 다른 저장소의 코드·설정·문서를 이 Issue/PR에서 수정하거나 복사하지 않는다. 필요하면 상대 저장소에 별도 Issue/PR을 만든다.
- 모든 기능, 결함, 문제는 이 저장소 GitHub Issue에 기록한다. Issue → 이 저장소 브랜치 → 기능 단위 커밋 → PR → CI/리뷰를 따른다.
- 커밋과 PR 제목은 `feat: 한글 설명`처럼 영어 접두사 + 한글 본문을 쓴다. PR 본문에 `Closes #번호`를 적는다.
- 실제 XRT 데이터, 가공 데이터, 모델 파일, 비밀값을 Git에 넣지 않는다.
- 저장소 간 계약의 버전·스키마·해시·호환성 영향을 문서화한다. 빈 모듈을 구현 완료로 보고하지 않는다.
