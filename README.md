# sic-xrt-ml

SiC XRT 결함 탐지·분류 모델의 연구, 학습, 평가, ONNX 내보내기를 담당합니다. BPD/TED/TSD/Artifact 등의 라벨 체계는 데이터 계약과 Issue에서 확정한 뒤 구현합니다.

## 시작

- Python 3.11 이상
- 개발 도구: `python -m pip install -e ".[dev]"`
- 학습 환경: `python -m pip install -e ".[train]"`
- 검사: `ruff check .`, `pytest`

현재는 `data/`, `preprocessing/`, `training/`, `evaluation/`, `export/` 모듈 경계만 마련했습니다. 학습·평가 코드는 별도 기능 Issue에서 추가합니다.

원본 XRT 데이터, 체크포인트, 실험 산출물은 Git에 넣지 않습니다. Analyzer에는 버전·해시·라벨·입출력 스키마가 기록된 모델 산출물만 전달합니다. [AGENTS.md](AGENTS.md)와 [공통 handbook](https://github.com/CrystalVision-Lab/engineering-handbook)을 읽으세요.
