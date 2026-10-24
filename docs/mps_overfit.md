# MPS에서 첫 train 과적합 검증

## 준비 상태

프로젝트에 pyproject.toml과 uv.lock을 추가했다. 검증 환경은 arm64 macOS,
Python3.14.5, PyTorch2.14.1, MPS였다. Python3.15 실행은 이번에 검증하지 않았다.
원격 고정 데이터셋의 train.jsonl만 data/mps_pilot/train.jsonl로 복사했다.
validation/test 파일은 가져오거나 읽지 않았다.
고정 seed20261024로 선택한16개 서열의 ESM 임베딩 캐시도 만들어두었다.

```fish
uv sync --python 3.14
uv run python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
uv run python -m unittest discover -s tests -v
```

MPS가 False면 프로그램은 오류로 종료한다. 자동 CPU fallback을 켜지 않는다.

## 첫 학습

이미 data/mps_pilot/cache.pt가 있으므로 바로 실행할 수 있다.

```fish
uv run python -u scripts/train_mps_overfit.py \
  --device mps --cache data/mps_pilot/cache.pt \
  --steps 2000 --batch-size 1 --eval-every 200 --ode-steps 32 \
  --out outputs/mps_overfit16
```

M1 메모리와 처리량을 고려해 float32, batch1, hidden96, pair32,
서열 pair block2개, 기하 attention block2개로 시작한다. 학습 본체401,678파라미터다.
긴 실행 중 절전/노트북 덮개 닫힘은 작업을 정지시킬 수 있다. 터미널을 닫지 않고
필요하면 caffeinate -i uv run python ... 형태로 실행한다.

checkpoint는 평가 구간 끝마다 last.pt로 저장한다. 중단 후 같은 명령에 --resume을 붙인다.
--steps는 추가 횟수가 아니라 최종 누적 목표다. 출력 폴더를 재사용할 때 --resume이 필수다.
checkpoint에는 모델, optimizer, cache hash, 모델 설정, CPU torch RNG, NumPy RNG가 들어 있다.
기존 checkpoint 이후 저장되지 않은 step은 재실행한다. MPS의 완전한 비트 재현성을 보증하지 않는다.

## ESM 캐시 재생성

```fish
uv run python scripts/prepare_mps_pilot.py --count 16
```

공개 facebook/esm2_t6_8M_UR50D 모델과 commit c731040fcd8d73dceaa04b0a8e6329b345b0f5df를
사용한다. 인증 토큰 없이 공개 가중치를 받고 CPU에서 frozen embedding을 만든다.
리간드는 입력에 사용하지 않는다. 원본 train의 전체 SHA256과 선택 ID를 cache에 기록한다.
캐시는 자체 생성 파일만 읽는다. 외부 임의 pickle 파일을 사용하지 않는다.

## 모델과 단위

- 서열 → frozen ESM320차원 + 아미노산/절대 잔기 위치 → Single/Pair trunk.
- 현재 Cα 위치/국소 회전/시간 → pair geometry + multi-head point attention → body-frame 위치/회전 속도.
- 위치 prior는 중심을 뺀 표준 Gaussian(nm), 회전 prior는 Haar SO(3).
- CPU에서 정답을 noise에 proper Kabsch 정렬하고, 선형 위치 및 SO(3) geodesic 경로를 만든다.
- 모델·역전파·optimizer·생성 적분은 MPS에서 수행한다. SVD/SO(3) log는 CPU 정답 구성에만 사용한다.
- 속도 MSE + Cα FAPE + pair distance + 원자 위치 + 펩타이드 결합 길이 보조 손실.
- 원본 label_seq_id 위치를 유지한다. 결측 frame/atom은 손실에서 제외하고 gap에 결합 손실을 걸지 않는다.
- 길이 padding과 실제 frame/atom 관측 마스크는 구분한다. 관측 여부는 모델 입력 조건으로 주지 않는다.
- 내부 좌표 nm, 최종 PDB/평가 RMSD는 Å. 회전 행렬은 column-vector local-to-global이다.
- N/CA/C는 ideal local backbone으로 복원한다. O는 다음 N을 이용한 평면 이등분 방향 근사다.
  말단/끊긴 결합의 O는 원자 손실에서 제외한다. export는 서열상 연속 골격을 가정하고 마지막 O를 생략한다.
- 이는 backbone 모델이다. 곁사슬/리간드 좌표나 신뢰도 head는 아직 구현하지 않았다.

## 결과 읽기

- history.jsonl: stochastic train flow/geometry 손실.
- eval_000000.json: 학습 전 고정 noise 생성 기준선.
- eval_000200.json 등: 같은16개 train 단백질에서 한 고정 seed, 동일 ODE step의 생성 평가.
- mean_ca_rmsd_A: proper Kabsch 정렬 후 관측 Cα RMSD 평균.
- mean_ca_lddt: 관측 Cα의15 Å 이내 잔기쌍에서 .5/1/2/4 Å 거리오차 기준 점수 평균.
  공식 all-atom lDDT와 동일 지표라고 해석하지 않는다.
- 마지막 step에는 각 서열의 생성 PDB와 last.pt를 저장한다. PDB B-factor0은 신뢰도가 아니다.

학습 손실 감소만으로 성공을 판정하지 않는다. 고정 seed에서 구조 RMSD 감소와 Cα lDDT
상승을 함께 확인한다. 여러 seed 중 정답에 가장 가까운 샘플을 고르지 않는다.
이 검증은 train 과적합 실험이며 일반화 성능 보고가 아니다.

## 실제 검증 범위

CPU 단위 테스트12개 통과: SE(3) equivariance, padding, SO(3) zero-angle gradient,
손실 역전파와 기존 데이터 파이프라인 검사를 포함한다.
MPS에서 synthetic forward/backward/optimizer3 step, SO(3) 생성 실행을 확인했다.
실제16개 단백질에서 짧은 MPS 학습·생성·PDB·checkpoint 저장 및 resume를 확인했다.
이 짧은 실행은 코드 연결 검증으로, 과적합 성공이나 실제 예측 성능 검증이 아니다.

## 과적합 실패 진단 (추가 학습 없음)

```fish
uv run python scripts/diagnose_flow.py \
  --cache data/mps_pilot/cache1.pt \
  --checkpoint outputs/mps_overfit1/last.pt \
  --device mps --ode-steps 32 128 256 --trials 4 \
  --out outputs/diagnostics_overfit1
```

report.json의 roundtrip은 프레임 직교성/방향/평면과 원본 local좌표 왕복을 검사한다.
ideal_backbone_rms_error_A는 고정 결합 길이·각도와 O 근사의 오차로 별도 보고한다.
oracle은 정답 속도/회전 경로를 적분해 종점에 도착하는지 검사한다.
time_profile은 동일 prior 경로를 t=0..1에서 탐색하는 teacher-path 진단이다.
ode_comparison은 학습 평가와 같은 초기 noise seed를 사용해32/128/256step 생성을 비교한다.
서로 다른 noise 중 정답에 가까운 샘플을 고르지 않는다.

Sanity 항목만 자동 pass/fail로 판단하며, 학습된 모델의 RMSD/lDDT는 진단 수치로 보고한다.
프로파일의 t=0/1은 학습 시간 범위(.02,.98) 밖이다. t=1의 endpoint 좌표 오차0은
정의상 당연하므로 속도 예측이 정확하다는 증거로 해석하지 않는다.
