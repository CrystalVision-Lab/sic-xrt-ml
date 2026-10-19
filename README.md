# SiC XRT ML

승인된 `verified_xrt_dataset/v1` RGB8 128×128 패치의 BPD/TED/TSD 분류, 그리고 검수 라벨 합의 `verified_xrt_dataset/v2`의 BPD/TED/TSD/NOT_DEFECT 4클래스 분류 연구 파이프라인. 학습·평가·ONNX 후보 산출을 수행하며 Analyzer 모델을 자동 교체하지 않는다. 입력 데이터 변환·UI 코드는 이 저장소에 포함하지 않는다.

## 설치 및 실행

Python 3.11 이상. CPU 설치 예:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train]"
.\.venv\Scripts\python.exe -m sic_xrt_ml validate <private-dataset-directory>
.\.venv\Scripts\python.exe -m sic_xrt_ml train <private-dataset-directory> --config configs/patch-baseline.json --output artifacts/<new-run>
.\.venv\Scripts\python.exe -m sic_xrt_ml evaluate <private-dataset-directory> --checkpoint artifacts/<new-run>/checkpoint.pt --output artifacts/<new-evaluation> --existing-bundle <existing-approved-model-directory>
.\.venv\Scripts\python.exe -m sic_xrt_ml export --checkpoint artifacts/<new-run>/checkpoint.pt --evaluation artifacts/<new-evaluation>/evaluation.json --output artifacts/<new-candidate>
```

Linux에서는 `.venv/bin/python`을 사용한다. 현재 실제 승인 데이터가 0개라 실데이터 학습은 NOT_RUN이며 실제 성능 향상은 NOT_VERIFIED다. 준비되지 않은 데이터는 학습·평가를 거부한다. 자동 데이터 다운로드나 업로드는 없다. 서버에서 현재 Dataset usage_status=AVAILABLE을 확인한 관리자가 반출한 자료만 사용한다.

[TD-07 결과·사용 정책](docs/XRT-RESULT-TD-07-TRAINING-EVALUATION.md)에 학습 설정·재개·독립 평가·제약을 기록한다.

[v2 4클래스 학습](docs/VERIFIED-DATASET-V2-TRAINING.md): Analyzer `python -m xrt_server.label_dataset_admin`으로 반출한 폴더를 같은 명령으로 학습한다(`--min-per-class train,validation,test`). [모델 산출물 계약](docs/model-contract.md)에 frozen manifest v1/v2를 기록한다.
