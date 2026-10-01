# JupyterLab에서 분류 기준 모델 학습

데이터셋을 업로드하거나 Jupyter 작업 폴더로 복사할 필요가 없습니다.
`notebooks/01_train_classifier.ipynb`를 Git 제외 `work/`에 복사하고
**SiC XRT GPU** 커널로 열어 외부 데이터셋 경로를 설정합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[baseline]"
Copy-Item notebooks/01_train_classifier.ipynb work/01_train_classifier.ipynb
```

노트북을 위에서 아래로 Shift+Enter로 실행하거나 **Run → Run All Cells**를 선택합니다.
기본 128 입력·batch=32·15 epoch이며, 학습은 균형 CSV(각 클래스 같은 수), 검증은
val 전체를 사용합니다. Windows 노트북에서는 num_workers=0으로 실행합니다.
새 실행마다 runs 안의 별도 디렉터리를 사용하며 이전 모델/데이터셋을 덮어쓰지 않습니다.

명령줄에서도 동일한 학습을 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m sic_xrt_ml.training.patch_classifier --dataset DATASET_V1_ROOT --outputs runs --epochs 15
```

## 입력과 결과

Data Tools 출력 schema_version=1을 CSV/JSON으로 읽으며 내부 코드를 import하지 않습니다.
현재 클래스 순서는 **BPD, TED, TSD**입니다. 검증 보고서 통과와 manifest 해시,
웨이퍼별 단일 split, balanced CSV가 변경되지 않은 train 행만 포함하는지 확인합니다.
각 패치를 읽을 때 파일 SHA-256과 shape/dtype을 확인합니다.

- uint8은 255, uint16은 65535로 나눈 float32 [0,1] 입력입니다.
- RGB YXS를 CHW로 바꾸고 grayscale은 동일 값 3채널로 반복합니다.
- 크기를 바꾸거나 중심을 이동시키는 crop/augmentation은 적용하지 않습니다.
- SmallPatchCNN_v1은 사전 학습 가중치 없이 초기화하여 학습하는 작은 기준 CNN입니다.
- 출력은 [N,3] logits, softmax를 적용하면 위 클래스 순서의 점수입니다.
- 학습·검증 손실, 클래스별 precision/recall/F1과 혼동행렬을 history.json에 기록합니다.
- val macro F1이 가장 높은 epoch를 best_model.pt로 저장합니다. test는 선택에 사용하지 않습니다.
- config.json은 데이터/학습 CSV 해시, 클래스·전처리·분할·seed/torch·설정을 기록합니다.
- 중단하면 result.json에 interrupted를 남기고 완료한 epoch의 기록과 best 체크포인트를 보존합니다.

GPU 메모리가 부족하면 batch를 16 또는 8로 낮춥니다. epoch는 학습 자료를 한 번씩
읽는 횟수입니다. 반복 실행은 새 실험이며 이전 결과를 자동으로 이어받지 않습니다.

## 평가와 사용 범위

데이터의 시각적 검수 상태가 pending이면 결과에도 그대로 남깁니다.
이 기준 모델은 주석 중심의 패치 종류를 분류하며 정상 클래스·전체 영상 위치 검출·
3D·승인 모델·ONNX export·Analyzer 연결을 포함하지 않습니다.

노트북의 RUN_TEST=False가 기본값입니다. 설정을 확정한 후 True로 바꾸어 test를
한 번 평가합니다. 반복 test 확인으로 모델을 고르면 시험셋의 독립성을 잃습니다.
결과는 test_metrics.json에 따로 저장합니다. 기준 모델의 점수는 제품 성능 주장이나
독립적인 새 웨이퍼 전체 성능으로 확대 해석하지 않습니다.
