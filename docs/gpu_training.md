# 단일 CUDA GPU 본 학습 파이프라인

## 준비된 기능

- frozen ESM-2 150M, 고정 revision a695f6045e2e32885fa60af20c13cb35398ce30c.
- split별 재시작 가능한 CPU shard 캐시. 기본 train/validation만 읽음. test는 명시적 opt-in.
- 고정 train1,989 / validation249 / test249를 그대로 사용. 코드에서 수량을 강제하지 않음.
- CPU shard LRU, 길이별 묶음 배치, padding/관측 mask, persistent worker/prefetch와 CUDA pinned memory 전송.
- 기존 FoldingModel(hidden128,pair64,pair3,geometry3,recycling3), 새로 초기화. ESM은 학습하지 않음.
- invariant loss와 전체 target rigid augmentation. 원자/회전/기하 손실은 FP32.
- precision=bf16은 CUDA의 서열/Pair trunk만 autocast하며 기하 디코더와 손실은 FP32.
- AdamW, warmup+cosine LR, gradient clipping, validation mean Cα lDDT 기반 best.pt와 early stopping.
- last.pt에 optimizer/scheduler/RNG/epoch/다음 batch/부분 epoch 누계/데이터 및 코드 fingerprint 저장.
- SIGTERM/Ctrl+C를 받으면 현재 update 후 checkpoint 저장. 강제 SIGKILL이면 마지막 저장 이후만 재실행.
- test는 학습기에서 열지 않음. 완료 후 validation-selected best.pt를 별도 명령으로 한 번 평가.

## 1. 코드와 환경

GPU 서버 작업 폴더에 foldflow-gpu-pipeline.tar.gz를 올려서 푼다.

```bash
mkdir -p ~/foldflow
cd ~/foldflow
tar -xzf /파일/경로/foldflow-gpu-pipeline.tar.gz
uv sync --locked --python 3.14 --extra cu126
uv run --locked --extra cu126 python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

기본 extra cu126은 PyTorch 2.11.0 / CUDA 12.6 build다. Elice R535 드라이버에서는
CUDA 12.x minor-version compatibility를 사용하며 실제 forward/backward 검증 후 학습한다.
시스템 CUDA toolkit을 별도로 설치한다는 의미는 아니다. CUDA 13 드라이버에서는
CUDA_EXTRA=cu130으로 실행 스크립트의 build를 선택할 수 있다.
CUDA unavailable이면 CPU로 조용히 전환하지 않고 종료한다.
다른 driver/build가 필요하면 해당 PyTorch index를 추가해 lock을 다시 생성해야 한다.
CPU 환경은 uv sync --extra cpu, Mac MPS는 기존 기본 uv sync를 사용한다.
Python3.15는 현재 고정 torch wheel과 맞지 않아3.14를 쓴다.

## 2. 데이터 전송

92GB PDB 원본은 GPU 서버에 필요 없다. CPU 서버의 /var/projects/foldflow에서
고정 train/validation JSON과 체크섬/완료표시만 전송한다. USER@GPU를 실제 주소로 바꾼다.

```bash
ssh USER@GPU 'mkdir -p ~/foldflow/data/dataset_v1_final'
rsync -av --partial \
  --include='train.jsonl' --include='validation.jsonl' \
  --include='checksums.json' --include='COMPLETE' --exclude='*' \
  data/dataset_v1_final/ USER@GPU:~/foldflow/data/dataset_v1_final/
```

test.jsonl은 이 단계에서 옮기지 않는다. 별도로 전체 cache를 이미 만들어둔 경우,
아래 cache 디렉터리의 config/train/validation을 폴더 구조째 옮기면 원본 JSON도 필요 없다.

## 3. GPU 대여 전 캐시 만들기 (선택)

Mac 또는 CPU에서 아래를 실행한 뒤 data/cache150를 GPU에 전송하면 GPU 대여 중
ESM 다운로드/임베딩 준비 시간을 줄일 수 있다. 같은 source checksum과 model revision을 쓴다.

```bash
uv run python scripts/prepare_dataset.py \
  --dataset data/dataset_v1_final --out data/cache150 --device mps --batch-size 4
```

CPU는 --device cpu. --batch-size는 ESM cache 생성용 batch다.
GPU에서 직접 만들려면:

```bash
uv run --locked --extra cu126 python -u scripts/prepare_dataset.py \
  --dataset data/dataset_v1_final --out data/cache150 --device cuda --batch-size 8
```

중단 후 동일 명령을 실행하면 완성된 shard를 검증해 재사용한다. source/model/shard_size가
변하면 새 --out이 필요하다. 최종 index.json이 있어야 학습에 쓸 수 있다.

## 4. 실측 후 본 학습

```bash
uv run --locked --extra cu126 python scripts/preflight_gpu.py \
  --cache data/cache150 --device cuda --batch-size 8 --precision fp32 \
  --out outputs/preflight_fp32.json
```

실제 모델의 forward/backward/optimizer를 실행해 처리량과 peak GPU 메모리를 기록한다.
부족하면 batch4/2로 낮추고, BF16-capable GPU이면 --precision bf16도 별도 결과로 비교한다.
첫 실제 epoch의 로딩/validation 시간을 더해서 전체 예산을 계산한다.

일단100update까지만 훈련 경로를 실행하는 방법:

```bash
uv run --locked --extra cu126 python -u scripts/train_dataset.py \
  --cache data/cache150 --out outputs/fold150_full_v1 \
  --device cuda --precision fp32 --epochs 100 --batch-size 8 --workers 2 \
  --max-updates 100
```

이 cap 실행은 TRAINING_COMPLETE를 만들지 않는다. 이어서 같은 설정으로 --resume을
붙이고 --max-updates를 제거하면 이어간다. epoch/lr/model/precision/data/code 변경은 거부한다.
GPU 메모리·실행 경로 확인용으로만 사용하고 test를 열지 않는다.

한 번에 실행하려면 프로젝트 루트에서:

```bash
nohup bash scripts/run_gpu_training.sh > gpu_training.log 2>&1 < /dev/null &
```

fish에서 PID를 저장/분리하려면 위 명령 직후:

```fish
set training_pid $last_pid
echo $training_pid > gpu_training.pid
disown $training_pid
```

기본 cache=data/cache150, run=outputs/fold150_full_v1, batch8, FP32다.
BATCH_SIZE, PRECISION, DATASET_DIR, CACHE_DIR, RUN_DIR 환경변수로 바꿀 수 있다.
예: env BATCH_SIZE=4 PRECISION=bf16 bash scripts/run_gpu_training.sh
runner는 last.pt가 있으면 동일 설정으로 resume하고, 완료된 run은 덮어쓰지 않는다.
다른 batch/precision을 비교할 때는 새 RUN_DIR를 사용한다.

## 5. 학습 기록과 선택

- config.json: 모델·optimizer·cache/code fingerprints·runtime 정보.
- history.jsonl: step loss/LR/처리량/VRAM 및 epoch별 validation 기록.
- validation_*.json: validation249개 전부의 단백질별 Cα RMSD/lDDT/정렬 후 회전오차.
- best.pt: validation mean Cα lDDT가 min_delta이상 개선될 때 저장.
- last.pt: 재시작용 최신 상태.
- status.json: 현재 cursor와 중단 이유.
- TRAINING_COMPLETE.json: epoch 상한/validation early stopping으로 끝난 경우에만 기록.

warmup250updates, 최대100epochs, patience15epochs가 초기 기준이다. 실제 validation을 보고
새 실험의 설정을 조정할 수 있지만 test를 모델 선택에 쓰지 않는다.
초기15개 과적합 checkpoint를 이어 쓰지 않고 새 모델을 초기화한다.
throughput은 validation 시간을 제외하되 checkpoint/I/O/worker startup을 포함한다.
단일 GPU용이며 DDP나 여러 GPU 자동 사용은 구현하지 않았다.

## 6. 최종 test (학습·모델 선택 완료 후에만)

완료된 run을 확정한 뒤 CPU 서버에서 test.jsonl을 같은 dataset 폴더로 옮긴다.

```bash
uv run --locked --extra cu126 python scripts/prepare_dataset.py \
  --dataset data/dataset_v1_final --out data/cache150 --device cuda --batch-size 8 --splits test
uv run --locked --extra cu126 python scripts/evaluate_test.py \
  --run outputs/fold150_full_v1 --cache data/cache150 \
  --device cuda --out outputs/fold150_full_v1_test
```

학습 완료/validation 선택 여부를 확인한 뒤 best.pt만 사용한다. train/validation의
index fingerprint와 test의 ID·전체서열 hash·cluster·원 PDB entry 겹침을 다시 검사한다.
TEST_EVALUATED.json으로 노출을 기록하고 동일 run 반복 실행은 거부한다.
이는 test로 다시 튜닝하는 것을 막기 위한 구분이며, 별도 실험 간 test 사용 정책까지 자동 통제하지는 않는다.

## 검증 범위와 남은 한계

- 전체22개 테스트 통과: padding 불변 예측, split leakage 거부, shard checksum,
  중간 batch resume와 연속 실행의 동일 결과, test 사전 노출 차단 및 별도 최종 평가 포함.
- 실제150M 임베딩의 train4/validation2 실행용 부분집합에서 캐시→MPS 배치학습2epochs→validation/best/last 저장 실행.
- CPU에서 persistent worker를 포함한 동일 경로 재실행 확인.
- 전체 데이터 학습과 CUDA/BF16 실제 실행은 현재 GPU가 없어 미검증. GPU에서 preflight부터 실행한다.
- Cα lDDT는 이 프로젝트 구현이며 공식 all-atom lDDT와 같다고 주장하지 않는다.
- 구조 native supervision, 원자 마스크, monomer 필터는 기존 초기 기준이다. 상세 wwPDB validation,
  원거리 구조 상동성, PLM 사전학습 중복과 side-chain 정확도는 아직 검증하지 않았다.

## 맥북 전처리 완료 상태 (2026-10-24)

MPS 과적합 학습을 종료했고 마지막 정상 checkpoint22,500step을 보존했다.
종료 순간 저장을 지원하지 않는 기존 학습기였으므로 저장되지 않은 이후 update는 포함되지 않는다.

전체 train1,989개/validation249개의150M 캐시가 data/cache150에 준비됐다.
총39파일,601,233,579bytes(약601MB/573MiB)다.
샤드 checksum, source sequence/cluster 대응, NaN/Inf, proper SO(3) frame/축/평면,
frame/atom mask, 최소95% backbone 관측률, peptide link consistency와 split overlap 검사를 통과했다.

배치8 계획에서 train249batches, validation32batches이며 매 epoch 모든 record가 정확히1회 포함된다.
이 계획의 padding 잔기 비율은 train약1.71%, validation약1.36%다.
리간드 종류/접촉 잔기와 결측 backbone residue 목록은
 data/gpu_ready/preprocessing_manifest.jsonl에 보존했다.

GPU 서버에는 코드 + data/cache150 + data/gpu_ready만 옮기면 된다.
원본 PDB, ESM 가중치와 원본 train/validation JSON은 본 학습에 필요 없다.

```bash
rsync -av --partial data/cache150/ USER@GPU:~/foldflow/data/cache150/
rsync -av data/gpu_ready/ USER@GPU:~/foldflow/data/gpu_ready/
python3 scripts/verify_gpu_ready.py --cache data/cache150 --ready data/gpu_ready
```

대상 디렉터리는 먼저 만든다. run_gpu_training.sh도 READY.json이 있으면 같은 체크를 수행한다.
이 검증기는 Python표준 라이브러리만 사용하므로 PyTorch 설치 전에 실행할 수 있다.

본 학습에는15개 과적합 checkpoint를 사용하지 않고 새로 초기화한다.
Test249개는 원본/임베딩을 읽지 않았고 이번 캐시/전송 manifest에 포함하지 않았다.
