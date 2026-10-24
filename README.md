# Flow Fold

짧은 단량체 단백질의 서열 조건부 backbone 예측을 연구하는 실험 저장소다.
현재 모델은 frozen ESM-2 150M과 작은 single/pair folding trunk, triangle multiplication,
point attention, 잔기 프레임 decoder를 사용하는 **직접 구조 예측 모델**이다.
초기 Flow Matching 실험과 이후 직접 예측 실험을 함께 보존한다.

## 현재 결과

고정 train 1,989 / validation 249 / test 249 split을 사용했다.
첫 A100 MIG 20GB 실행은 8,972 step에서 수동 중단했고, validation 기준 최고 모델은
35 epoch / 8,715 step이었다. Test는 이 실행과 후속 분석에서 읽지 않았다.

| 지표 | Train 전체 | Validation 전체 |
|---|---:|---:|
| 평균 Cα lDDT | 0.727 | 0.480 |
| 중앙값 정렬 Cα RMSD | 1.75Å | 5.89Å |

일반화 격차와 큰 backbone 결합/충돌 위반이 확인됐다. 고정밀 또는 물리적으로 검증된
구조 예측기로 취급하지 않는다. 이 수치는 자체 Cα 평가 구현에 따른 단일 seed의 결과이며,
validation으로 checkpoint를 선택했으므로 독립적인 test 성능이 아니다.

- [결과 분석과 학습 곡선](docs/results/gpu_v1/report.md)
- [현재 모델과 회전 불변 손실](docs/folding_v2.md)
- [다음 버전 설계안: 기하 제약·distogram·recycling](docs/folding_v3_plan.md) — 아직 구현되지 않은 계획
- [이전 저장소 실험과 재현 명령](docs/legacy_experiments.md)

## 설치와 검증

Python 3.14와 uv를 사용한다.

```bash
# Mac MPS
uv sync --locked
uv run --no-sync python -m unittest discover -s tests -v

# Linux CPU
uv sync --locked --extra cpu

# Linux NVIDIA GPU: PyTorch 2.11 / CUDA 12.6 build
uv sync --locked --python 3.14 --extra cu126
```

CUDA build는 시스템 toolkit과 별개다. GPU 드라이버와 실제 forward/backward 호환성을
preflight로 확인한다. `CUDA_EXTRA=cu130`으로 실행 스크립트의 CUDA 13 build를 선택할 수도 있다.
그 경우 해당 build를 지원하는 드라이버가 필요하다.

## 데이터와 학습

- [데이터 품질 기준](docs/data_criteria_v1.md)
- [로컬 PDB 인덱싱](docs/local_archive.md), [전체 병렬 검사](docs/full_screen.md)
- [중복 제거와 고정 분할](docs/dataset_v1.md)
- [GPU 준비·캐시 검증·학습·별도 test 평가](docs/gpu_training.md)

전체 캐시가 준비된 GPU 서버에서:

```bash
python3 scripts/verify_gpu_ready.py --cache data/cache150 --ready data/gpu_ready
nohup bash scripts/run_gpu_training.sh > gpu_training.log 2>&1 < /dev/null &
```

학습기는 train/validation만 사용하며, validation lDDT로 best checkpoint를 선택한다.
SIGINT/SIGTERM 시 마지막 update 이후 optimizer/RNG 상태까지 저장한다.
Test 준비와 평가는 별도 opt-in 명령이다.

PDB 원본, ESM 캐시/가중치, 데이터셋 좌표, checkpoint, 로컬 백업과 가상환경은 Git에 포함하지 않는다.
따라서 새 clone만으로 위 결과를 즉시 재평가할 수는 없으며 데이터 준비 절차가 필요하다.

## 코드 구성

- `foldflow/`: 모델, 기하, 손실, shard dataset, 실행 유틸리티.
- `scripts/`: PDB 수집/검사/분할, ESM 캐시, 과적합 검사, 본 학습과 결과 분석.
- `tests/`: 기하 불변성, mask, gradient, 캐시 무결성, resume 및 분할 격리 검사.
- `docs/`: 실험 방법, 결과, 한계, 다음 설계.
- `outputs/*.py`, 기존 `data/*.py`: 이전 실험 코드. 현재 패키지와 별도로 보존한다.

학습 곡선 작성은 선택 의존성 matplotlib를 사용한다. 예:

```bash
uv run --no-project --python 3.14 --with matplotlib python scripts/plot_gpu_analysis.py \
  --run PATH_TO_EXPORTED_RUN --cache data/cache150 --out outputs/analysis
```
