# CPU 서버 비교 실행

USER@CPU_SERVER:/var/projects/foldflow에 코드, tests, pyproject/lock과
150M frozen embedding cache를 rsync로 전송했다. ESM 가중치는 실행에 필요하지 않다.

서버의 기존Python3.15 .venv는 데이터 처리용으로 보존한다.
현재 고정 PyTorch2.14.1의 cp315 wheel이 없어 별도Python3.14 .venv-cpu를 uv로 만든다.
Linux는 공식 pytorch-cpu index, macOS는 기존 PyPI/MPS wheel을 사용하도록 marker를 설정했다.

```fish
cd /var/projects/foldflow
env UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --python 3.14 --extra cpu

env UV_PROJECT_ENVIRONMENT=.venv-cpu uv run --python 3.14 --extra cpu python -u scripts/train_direct_overfit.py \
  --cache data/mps_pilot/cache_150M_15.pt \
  --architecture fold --loss-mode invariant --augment-rigid \
  --hidden 128 --pair-dim 64 --pair-layers 3 --geometry-layers 3 --recycles 3 \
  --device cpu --steps 1500 --eval-every 1500 \
  --out outputs/cpu_fold150m15_test
```

현재 학습기는 torch.set_num_threads(4)를 사용한다.
속도 비교는 초기 import/평가를 제외하고 history.jsonl의 step100 이후 구간의
elapsed_s 차이/step 차이를 사용한다. CPU와 MPS의 모델/데이터/목표/평가 주기를 동일하게 둔다.
같은 seed여도 backend float32 연산 차이로 학습 수치가 완전히 같을 필요는 없다.
1,500step 뒤 필요하면 같은 명령에서 --steps30000 --resume으로 이어간다.
학습 출력은 MPS 폴더와 분리한다. 원격 학습이 SSH 종료 후에도 필요하면 nohup과
fish $last_pid/disown을 사용한다.
