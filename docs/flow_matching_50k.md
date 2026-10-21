# 40–128잔기, 5만 구조 SE(3) Flow Matching 실행 안내

상태: 수집·클러스터 제외·ESM 샤드·학습·최종 평가 **코드 제공**. 이 환경에는 `mmseqs`와 GPU가 확인되지 않아 5만 개 수집이나 대규모 학습은 실행하지 않았다. 확보 수, 수렴, 구조 성능 수치는 아직 없다.

## 데이터 출처와 제한

40–128잔기의 실험 구조만으로 5만 학습 예제를 얻는다고 가정하지 않는다. `collect_afdb_short.py`는 UniProtKB의 해당 길이 서열을 조회하고 AlphaFold DB PDB 좌표를 받아서 `평균 pLDDT ≥85`, `최소 잔기 pLDDT ≥70`, `pLDDT 표준편차 ≤15`, 완전한 N/Cα/C 및 끊어지지 않은 backbone을 확인한다. 예측 구조는 **pseudo-label**이며, 실험 정답과 동등하지 않다. API 버전·필터에 따라 7만 후보를 채우지 못할 수 있고 이 경우 스크립트가 명시적으로 실패한다.

기존 validation/test 실험 구조 중 **40–128잔기**만 사용하고, 이들과 30% sequence-identity 클러스터가 겹치는 학습 후보는 `mmseqs`로 제외한다. 따라서 validation/test 수는 각각 100개보다 적을 수 있다. 이 test는 이미 연구 과정에서 여러 번 확인했으므로 최종 논문 수준의 독립 평가에는 **새로운 untouched experimental holdout**이 필요하다. 30% 서열 클러스터링만으로 구조적 유사도 누수가 사라지는 것도 아니다.

## 실행

아래 명령은 저장소 루트 기준이다. `outputs/requirements.txt` 설치와 `mmseqs` 실행 파일이 필요하다. GPU에서 ESM 임베딩과 학습을 실행할 때는 CUDA PyTorch 설치가 필요하다.

### GPU 대여 전에 CPU에서 준비하기

별도 CPU 머신의 저장소 루트에서 아래를 실행한다. 스크립트는 7만 후보 수집, 5만 학습 구조 확정, 실험 구조 다운로드, 파일 감사를 수행한다. 7만 후보가 확보되지 않으면 실패하며, 이때 5만 학습 구조가 준비되었다고 간주하면 안 된다.

```bash
bash scripts/prepare_flow50k_cpu.sh
.venv/bin/python data/flow50k_filelist.py \
  --manifest work/flow50k/manifest.json \
  --out work/flow50k/upload_files.txt
```

H100 서버에서 저장소를 클론한 다음, CPU 머신에서 최종 manifest가 참조하는 **선택된 5만 구조와 validation/test 구조만** 전송한다. 아래 `USER@HOST`는 실제 SSH 주소로 바꾼다.

```bash
rsync -a --partial --files-from=work/flow50k/upload_files.txt ./ USER@HOST:~/flow-fold/
```

전송 후 H100의 저장소에서 `bash scripts/run_h100_flow50k.sh`를 실행하면 이미 전송된 manifest를 찾아 수집·클러스터링을 건너뛰고, 감사 후 ESM 임베딩과 학습을 진행한다. 터미널을 닫을 계획이라면 앞서 안내한 `nohup` 명령을 사용한다. CPU와 H100 양쪽에서 저장소의 `work/` 및 `data/cif/` 상대 위치를 유지해야 한다. 최종 manifest는 자체 위치를 기준으로 파일 경로를 저장하므로 저장소 전체를 다른 머신에 옮겨도 유효하다.

```bash
.venv/bin/python data/collect_afdb_short.py --out work/afdb_short \
  --target 70000 --max-candidates 500000 --workers 8

.venv/bin/python data/finalize_flow_manifest.py \
  --candidates work/afdb_short/candidates.json \
  --holdout-manifest data/manifest_1000train_200holdout.json \
  --out work/flow50k/manifest.json --target 50000 --threads 16

.venv/bin/python data/audit_manifest.py \
  --manifest work/flow50k/manifest.json \
  --report work/flow50k/audit.json

.venv/bin/python data/download_structures.py

.venv/bin/python data/embed_flow_shards.py \
  --manifest work/flow50k/manifest.json \
  --out work/flow50k/cache --device cuda \
  --model facebook/esm2_t6_8M_UR50D --batch-size 32 --shard-size 256

cd outputs
../.venv/bin/python train_flow_large.py \
  --cache ../work/flow50k/cache --out run_flow_50k_batched \
  --epochs 10 --batch-size 8 --accumulate 4 --ode-steps 25 --device cuda

../.venv/bin/python evaluate_flow_large.py \
  --cache ../work/flow50k/cache \
  --checkpoint run_flow_50k_batched/best.pt \
  --out run_flow_50k_batched/test.json --device cuda
```

`work/`와 checkpoints는 Git에서 제외된다. 후보 수집은 페이지 단위, 임베딩은 샤드 단위로 재시작할 수 있다. 학습은 **완료된 epoch 경계**에서만 재시작한다:

```bash
../.venv/bin/python train_flow_large.py \
  --cache ../work/flow50k/cache --out run_flow_50k_batched \
  --epochs 10 --batch-size 8 --accumulate 4 --ode-steps 25 --device cuda \
  --resume run_flow_50k_batched/last.pt
```

실제 확보 가능한 데이터 수가 5만 미만이면 `finalize_flow_manifest.py`는 요청한 수를 조용히 줄이지 않고 실패한다. 확보 수와 필터 탈락 사유를 보고한 뒤 목표를 다시 결정해야 한다. 임베딩 모델을 바꾸면 별도 캐시·실험으로 취급한다. 첫 구현은 작은 ESM-2 8M을 고정하며, 큰 PLM 효과와 Flow Matching 효과를 혼동하지 않기 위함이다.

H100 학습 경로는 샤드별로 다음 샤드를 CPU에서 미리 읽고, 샤드 안의 서열을 길이별로 묶어 최대 8개씩 padding한다. 모든 attention, 삼각 갱신, 속도·거리·FAPE 손실에서 padding을 제외한다. CPU batch는 pinned memory를 사용해 비동기 전송하고, PyTorch scaled-dot-product attention과 BF16 autocast를 사용한다. `--batch-size 8 --accumulate 4`는 시작 설정이며 H100 메모리와 처리량을 보고 조정할 수 있다. 기존 단일 단백질 checkpoint와 새 batched 모델의 가중치는 호환되지 않아 실행 디렉터리를 `run_flow_50k_batched`로 분리했다.

## 알고리즘

각 잔기는 Cα 위치 `xᵢ ∈ ℝ³`와 N–Cα–C로 만든 고유 회전 `Rᵢ ∈ SO(3)`으로 표현한다. 좌표는 10 Å 단위로 스케일한다.

1. **조건 입력:** 고정 ESM-2 residue 임베딩과 아미노산 ID를 single representation으로, 두 residue와 절대 서열 offset을 pair representation으로 만든다. 현재 noisy frame에서 계산한 국소 변위·상대 회전·거리를 pair에 주입하고 한 번의 삼각 pair 갱신을 한다. IPA 방식의 point-distance, pair bias, pair value를 통해 residue 상태를 갱신한다.
2. **확률 경로:** 평균을 제거한 Gaussian 위치 `x₀`와 무작위 proper SO(3) 회전 `R₀`을 시작점으로 뽑는다. 정답 구조는 반사를 허용하지 않는 Kabsch 회전으로 시작점에 정렬한다. `t∼U(0.02,0.98)`에서 `xₜ=(1−t)x₀+t x₁`, `Rₜ=R₀ Exp(t Log(R₀ᵀR₁))`를 만든다.
3. **목표 속도:** 위치 목표는 `Rₜᵀ(x₁−x₀)`인 residue local velocity, 회전 목표는 `Log(R₀ᵀR₁)`인 body-frame angular velocity다. 모델은 서열·`xₜ,Rₜ,t`를 받아 두 값을 예측한다.
4. **손실:** 위치/회전 속도 MSE가 주목표다. symmetric distogram CE와 예측한 종료 frame의 Cα frame-aligned point error를 보조 손실로 준다. 학습 중에는 유효한 같은 chain segment의 pair만 구조 손실에 포함한다. 회전 적분은 SO(3) exponential map을 써서 determinant +1을 유지한다.
5. **추론:** 서열과 무작위 시작 frame에서 25단계 Euler ODE 적분으로 구조를 만든다. Validation은 매 epoch 고정 seed로 같은 방식으로 생성하며, `median RMSD/10 + 1−median lDDT`로 best checkpoint를 고른다. Test 데이터는 `evaluate_flow_large.py`에서 선택 후에만 읽는다.

이것은 FoldFlow/AlphaFold 전체를 복제한 모델이 아니다. 특히 큰 PLM, 다층 삼각 trunk, 재활용, 강한 원자 수준 보정은 이후 ablation으로 추가한다. 먼저 이 작은 조건부 Flow가 5만 pseudo-label에서 학습 가능한지 확인한다.

## 산출물과 성공 판단

- `work/afdb_short/candidates.json`: 후보, 품질 수치, 탈락 사유
- `work/flow50k/manifest.json` 및 `audit.json`: 확정 분할과 중복 감사
- `work/flow50k/cache/index.json` 및 `shard_*.pt`: 임베딩·frame 정답과 checksum
- `outputs/run_flow_50k_batched/history.jsonl`, `last.pt`, `best.pt`, `best_validation.json`, `test.json`

생성 손실이 내려가는 것만으로 성공으로 보지 않는다. 동일 데이터·평가 조건의 기존 Pair-Aware와 Cα RMSD, lDDT-Cα, TM 지표, 장거리 접촉을 비교한다. 현재 TM 지표는 고정 잔기 대응 근사치이므로 최종 평가에는 정식 TM-align 구현을 별도로 고정해야 한다. 새 untouched experimental test에서 모델을 선택하지 않는다.

참고: [FoldFlow-2](https://arxiv.org/html/2405.20313), [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2), [AlphaFold DB API](https://alphafold.ebi.ac.uk/api/prediction/Q92793), [UniProt REST](https://www.uniprot.org/help/api).
