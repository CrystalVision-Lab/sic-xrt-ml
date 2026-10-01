# Windows 로컬 GPU 연구 서버

JupyterLab에서 노트북을 실행하는 단일 사용자 연구 환경입니다. 결함 추론 API나
학습된 결함 모델은 포함하지 않습니다. 실행은 수동이며 재부팅 후 시작 파일을 다시 실행합니다.

## 설치

Windows 11, NVIDIA GPU/드라이버와 Python 3.11 이상이 필요합니다.
코드는 OneDrive 원본 자료 폴더 밖의 별도 폴더에 체크아웃합니다.
다른 프로그램의 가상환경을 재사용하지 않습니다. 저장소 루트에서 PowerShell로 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[server,dev]"
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name sic-xrt-ml --display-name "SiC XRT GPU"
Copy-Item local.example.json local.json
```

GPU wheel은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라
공식 인덱스에서 설치합니다. 이 환경은 PyTorch 2.9.1/CUDA 12.8 wheel로 고정합니다.
전체 CUDA 개발 도구나 WSL 설치는 이 구성에 필요하지 않습니다.
`local.json`의 `data_root`를 실제 자료 경로로 설정합니다. 포트 기본값은 8890이며
다른 프로그램이 쓰고 있으면 `port`를 변경합니다. `SIC_XRT_DATA_ROOT` 환경변수도 지원합니다.

## 사용

- `scripts/start-server.cmd`: 서버 시작 및 인증된 브라우저 열기. 이미 실행 중이면 같은 서버를 엽니다.
- `scripts/stop-server.cmd`: 이 체크아웃의 서버와 실행 중인 커널을 종료합니다. 작업을 저장한 뒤 사용합니다.
- `scripts/check-environment.cmd`: GPU 연산·역전파와 자료 메타데이터를 점검합니다.

JupyterLab 주소는 `http://127.0.0.1:8890/lab`입니다. 토큰이 필요한 새 브라우저에서는
시작 파일을 다시 실행하면 인증 URL이 열립니다. 인증 토큰은 로그나 문서에 복사하지 않습니다.
서버는 `127.0.0.1`에만 연결하고 기본 토큰 인증을 유지합니다. 방화벽 개방이나 외부 터널은 없습니다.
Windows가 꺼지거나 절전 상태에 들어가면 작업을 계속할 수 없습니다. 이 설치는 전원 설정을 변경하지 않습니다.

명령줄 상태 확인/재시작도 가능합니다.

```powershell
.\.venv\Scripts\python.exe -m sic_xrt_ml.workstation.server status
.\.venv\Scripts\python.exe -m sic_xrt_ml.workstation.server restart
.\.venv\Scripts\python.exe -m sic_xrt_ml.workstation.doctor --require-cuda
.\.venv\Scripts\python.exe -m sic_xrt_ml.workstation.inventory
```

## 작업과 자료

최초 실행에 `notebooks/00_environment.ipynb`를 Git 제외 `work/`로 복사합니다.
**SiC XRT GPU** 커널을 선택하면 이 가상환경에서 실행됩니다. 사용자가 수정한 노트북은
다음 실행에 덮어쓰지 않습니다. 실험 노트북을 Git에 공유할 때는 원본 파일명/경로와
셀 출력·주석 자료·모델이 포함되지 않도록 별도 검토해야 합니다.

원본은 `data_root`에 두고 메타데이터 보고서는 `outputs/inventory.json`에 저장합니다.
검사 코드는 읽기 전용이며 ZIP을 해제하거나 이미지 픽셀을 디코딩하지 않습니다.
ZIP 목차/확장자·ROI 파일명과 TIFF 페이지 수·dtype·축을 기록합니다.
압축 파일의 CRC, 모든 이미지의 디코딩 무결성, ROI 좌표와 이미지 매칭은 검증하지 않습니다.
복사 중인 파일은 다시 점검하세요. 중첩 ZIP 내부는 이 점검 범위에 포함되지 않습니다.

노트북은 사용자 권한으로 실행되므로 원본 경로에 쓰기가 OS 차원에서 차단되는 것은 아닙니다.
원본을 덮어쓰지 말고 작업 결과를 `work/`, `outputs/`, `runs/`에 저장합니다.
설정·토큰/로그·실행 결과·원본·모델은 Git에서 제외합니다.
실제 설치 버전은 로컬 `outputs/environment-freeze.txt`에 기록할 수 있습니다.

## 학습을 이어가기 전

ImageJ ROI가 있더라도 클래스 정의와 각 원본 TIFF에 대한 좌표/짝을 먼저 확정해야 합니다.
원본 변환과 주석/데이터셋 검증은 `sic-xrt-data-tools`의 별도 Issue 범위입니다.
ImageJ 헤더 `TYX`는 공간 `ZYX` 계약으로 바꾸지 않습니다.
8GB GPU에서 원본 전체 스택을 한 번에 올리는 방식 대신 라벨·모델 계약을 확정한 후
타일/패치 및 작은 배치 학습을 설계합니다. 이 서버 구축은 계약 스키마나 모델을 변경하지 않습니다.

환경 노트북의 Conv2d 역전파는 GPU 동작 점검이며 결함 학습·평가 결과가 아닙니다.
승인 모델/실제 학습·ONNX export·Analyzer 연결은 후속 기능입니다.
