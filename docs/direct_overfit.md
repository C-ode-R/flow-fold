# 단일 서열 직접 프레임 과적합 대조 실험

```fish
uv run python -u scripts/train_direct_overfit.py \
  --cache data/mps_pilot/cache1.pt --device mps \
  --steps 2000 --eval-every 200 --out outputs/mps_direct1
```

같은 cache1.pt를 사용한다. forward에는 ESM 임베딩·아미노산·서열 mask만 전달하며,
정답 좌표/회전, 관측 mask, 잡음, 시간은 전달하지 않는다.

기존 FrameFlow의 encode trunk와 GeometryBlock을 재사용한다. 시간/속도 head 대신
서열 특징에서 initial 위치+6D회전을 만들고, 두 기하 block에서 결정론적으로 위치/회전을
보정한다. 잔기별 학습 테이블은 없다. 원자 복원과 결측 mask는 기존 코드와 같다.

손실은 native 고정 좌표계에서 위치 MSE·회전 chordal MSE와 FAPE·거리·원자·결합 손실이다.
이는 Flow와 같은 최적화 문제는 아니다. 직접 모델 성공은 표현과 디코더가 이 서열을
외울 수 있다는 증거이며, Flow 구현 버그를 확정하는 실험은 아니다.

기본 출력:
- history.jsonl: 직접 좌표/회전/기하 손실.
- eval_*.json: Kabsch 정렬 후 Cα RMSD/lDDT, 정렬 전 좌표 RMSD, 최대 frame 회전 오차.
- prediction.pdb: 최신 생성 백본.
- last.pt: 모델·optimizer·cache hash·모델 설정·step.

operational_overfit_target_met는 Cα RMSD<2 Å 및 Cα lDDT>0.8 여부만 나타낸다.
이것만으로 회전 프레임 전체가 정확하다고 해석하지 않는다. 회전 오차도 함께 본다.
성공 여부는 train 1개 과적합으로, 일반화 성능이 아니다.
새 실험은 새 --out, 재시작은 같은 명령에 --resume을 붙인다.
--steps는 총 누적 목표이며 checkpoint는 eval 구간 끝에 저장한다.

## 실제 실행 결과

동일 cache1.pt(SHA256을 기존 Flow5,000step checkpoint와 대조)에서 MPS2,000step 완료.
출력 outputs/mps_direct1_check/.
Cα RMSD0.10815 Å, Cα lDDT1.0, 정렬 전 Cα RMSD0.11006 Å,
최대 native-frame 회전오차1.73835°. 과적합1개의 결과이며 일반화 성능이 아니다.
직접 모델383,549파라미터(Flow401,678)이고, 학습 목표/좌표계 조건도 달라진 대조 실험이다.
전체 단위 테스트15개가 통과했다.

## 15개 서열 과적합

train 15개를 고정 seed로 선정한 data/mps_pilot/cache15.pt를 준비했다.
기존1개 과적합 서열도 포함한다. 새로운 모델을 초기화해서15개를 함께 학습한다.
1개에서 학습한 checkpoint를 초기값으로 사용하지 않는다.
매15step마다 각 서열을 정확히 한 번씩 방문하고 순서는 epoch별로 섞는다.
sequence_id와 선택 순서는 resume 이후에도 유지된다.

```fish
uv run python -u scripts/train_direct_overfit.py \
  --cache data/mps_pilot/cache15.pt --device mps \
  --steps 30000 --eval-every 1500 --out outputs/mps_direct15
```

30,000step은 각 서열2,000회 방문에 해당하며 동일 모델을 공유한다.
매1,500step마다15개 전부를 평가한다. 평균뿐 아니라 worst RMSD, worst lDDT,
Cα 기준 통과 개수와 단백질별 지표를 root eval_*.json에 기록한다.
각 sequence_id 폴더에는 해당 단백질의 평가와 prediction.pdb를 저장한다.
모든 단백질에 RMSD<2 Å, Cα lDDT>0.8을 요구하는 것은 운영상 과적합 목표이며
프레임 회전오차는 별도로 확인한다. 평균만 낮아졌다고 전체 과적합으로 판정하지 않는다.
아직 validation/test를 읽지 않으며15개 과적합 성공도 일반화를 증명하지 않는다.

중간 결과로 빠르게 점검하려면 --steps3000 --eval-every300으로 별도 출력에 실행할 수 있다.
30,000step 본 실행은 자동으로 시작하지 않았다. 15개 모두1번씩 학습하는 짧은 MPS 실행으로
다중 서열 학습/전체 평가/checkpoint 저장을 검증했다.
