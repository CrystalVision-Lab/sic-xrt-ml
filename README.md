# sic-xrt-ml

SiC XRT 결함 탐지·분류 모델의 연구, 학습, 평가, ONNX 내보내기를 담당합니다. BPD/TED/TSD/Artifact 등의 라벨 체계는 데이터 계약과 Issue에서 확정한 뒤 구현합니다.

## 시작

- Python 3.11 이상
- 개발 도구: `python -m pip install -e ".[dev]"`
- 학습 환경: `python -m pip install -e ".[train]"`
- 검사: `ruff check .`, `pytest`

`data/`, `preprocessing/`, `training/`, `evaluation/`, `export/` 모듈 경계를 둡니다.
점 패치 분류 기준 모델의 학습·평가는 아래 노트북으로 실행합니다.
그 밖의 학습과 ONNX 내보내기는 별도 기능 Issue에서 추가합니다.

schema-v1 점 패치의 3클래스 기준 CNN 학습·검증은
[JupyterLab 학습 노트북 안내](docs/notebook-training.md)를 따르세요.
학습 산출물은 연구용이며 승인 모델/위치 검출/ONNX export는 별도 범위입니다.

명암 보정 모델의 누락 개선은 [BPD 오탐 제한 개발 실험](docs/recall-budget.md)에서
같은 웨이퍼 분할로 비교합니다. 새 후보가 제한을 충족하지 못하면 기존 모델을 유지합니다.

## Windows GPU 연구 서버

로컬 JupyterLab 실행/종료, CUDA 실제 연산·역전파 점검, 외부 자료 폴더의 읽기 전용
ZIP 목차·TIFF 헤더 점검을 지원합니다. [설치와 사용 방법](docs/windows-server.md)을 따르세요.
자료 경로는 Git 제외 `local.json`에 지정하고 작업 노트북·로그·보고서는 Git 제외
`work/`와 `outputs/`에 저장합니다. 서버는 이 PC에서만 접속할 수 있습니다.

원본 XRT 데이터, 체크포인트, 실험 산출물은 Git에 넣지 않습니다. Analyzer에는 버전·해시·라벨·입출력 스키마가 기록된 모델 산출물만 전달합니다. [AGENTS.md](AGENTS.md)와 [공통 handbook](https://github.com/CrystalVision-Lab/engineering-handbook)을 읽으세요.
