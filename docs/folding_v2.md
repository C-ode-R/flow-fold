# ESM-2 150M + 작은 folding trunk

## 추천 구조와 구현

ESM-2 150M frozen (640차원) → Single/Pair trunk → outgoing/incoming 삼각 곱셈 갱신
→ multi-head point attention을 이용한 잔기 frame 갱신 → recycling → N/CA/C/O backbone.
foldflow/folding.py의 FoldingModel은 기존 point attention과 frame update를 재사용한다.
ESMFold에서 아이디어를 가져온 작은 모델이며 full ESMFold/Evoformer/정식 IPA 복제는 아니다.
삼각 attention, side-chain torsion, confidence head는 현재 포함하지 않는다.

## 캐시

```fish
uv run python scripts/prepare_mps_pilot.py \
  --esm-model 150M --count 15 --embedding-device mps
```

기본 모델은150M으로 변경했다. 모델명을 지정할 수 있고 hidden_size를 자동 감지한다.
결과는 data/mps_pilot/cache_150M_15.pt. 기존8M cache15.pt 등은 보존한다.
기존 파일을 덮어쓰지 않으며 다른 --out을 지정해야 한다.
Hugging Face model commit SHA를 해석한 후 그 revision으로 가중치/토크나이저를 고정하고
cache provenance에 기록한다. 공개 가중치는 token=False로 접근한다.
CPU 서버에서 생성할 경우 --embedding-device cpu를 사용한다. 첫 다운로드/캐시 생성 후
학습에는 ESM을 실행하지 않으므로 ESM 가중치를 학습 메모리에 유지하지 않는다.

## 새 모델 학습

```fish
uv run python -u scripts/train_direct_overfit.py \
  --cache data/mps_pilot/cache_150M_15.pt \
  --architecture fold --loss-mode invariant --augment-rigid \
  --hidden 128 --pair-dim 64 --pair-layers 3 --geometry-layers 3 --recycles 3 \
  --device mps --steps 30000 --eval-every 1500 \
  --out outputs/mps_fold150m15_v2
```

먼저 --steps15 --eval-every15와 별도 --out으로 한 바퀴만 확인해도 된다.
기존 고정 좌표계 학습은 --architecture direct --loss-mode fixed로 재현 가능하다.
기존 checkpoint와 새 모델/임베딩/손실의 resume 혼합은 거부한다.

## 회전·이동 처리

--loss-mode invariant는 기존 절대 위치/회전행렬/원자위치 MSE를 사용하지 않는다.
Cα FAPE, 상대 회전 Ri^T Rj, 잔기쌍 거리, 모든 관측 원자의 frame-aligned FAPE,
펩타이드 C–N 길이를 사용한다. 다른 정답을 전체적으로 회전·이동시켜도 동일한 손실이다.
--augment-rigid는 정답의 모든 위치·회전·원자에 하나의 공통 proper 회전/이동을 적용한다.
반사와 잔기별 독립 회전은 사용하지 않는다. 순수 서열 입력은 그대로 유지한다.
이 손실에서는 rigid augmentation이 새 접힘을 제공하지 않으며 좌표계 일관성을 보장하는 역할이다.

max_frame_rotation_error_deg는 과거 비교를 위한 raw gauge 값이다.
새 max_aligned_frame_rotation_error_deg는 Cα의 proper Kabsch 정렬을 frame에도 적용한 뒤
계산하므로 전역 방향 차이를 제거한다. 새 모델에서는 정렬 후 회전 오차를 읽는다.

## 검증·비교

- 예측/정답에 독립적인 전체 회전·이동을 적용했을 때 손실과 위치 gradient 불변 검사.
- 삼각 갱신 padding mask 검사.
- 640차원, recycling과 역전파 검사 및 MPS optimizer 짧은 실행.
- 기존 테스트 포함19개 통과.

성능 개선 여부는 아직 검증하지 않았다. 모델 크기/목표/ESM을 동시에 바꾸므로
150M만의 효과를 판단하려면 다음 대조가 필요하다:
1. 8M + 기존 direct + invariant loss.
2. 150M + 기존 direct + 같은 invariant loss.
3. 150M + 새 folding + 같은 invariant loss.
동일15개/같은 seed로 과적합한 뒤 고정 train/validation으로 확장한다. test는 건드리지 않는다.

## 이번 실행의 실제 확인 사항

- facebook/esm2_t30_150M_UR50D revision a695f6045e2e32885fa60af20c13cb35398ce30c 가중치를 내려받았다.
- 이전15개와 같은 sequence_id의640차원 frozen 임베딩을 MPS에서 생성했고 cache_150M_15.pt로 저장했다.
- hidden128/pair64/pair3/geometry3/recycles3 모델의 학습 파라미터는1,075,319개다.
- 실제150M캐시15개로 MPS5step 학습·전부 평가·PDB/checkpoint 저장을 확인했다.
- 전체19개 테스트 통과. 실제 긴 과적합/validation 성능은 아직 확인하지 않았다.
- frame-aligned norm은0.01 Å smoothing을 사용해 정확히 일치하는 원자 근처의 수치 기울기를 안정화했다.

공식 참고:
- https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/config.json
- https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/trunk.py
- https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/tri_self_attn_block.py
