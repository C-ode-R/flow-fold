# FlowFold 대규모 학습 파이프라인

상태: 실행 계획과 데이터 분할 감사 도구까지 준비. 샤드 임베딩·GPU 학습기·v3 모델은 아직 구현 전이며, 이 문서는 실행되지 않은 성능이나 학습 시간을 주장하지 않는다. 모델 구조는 [FlowFold v3 설계](flowfold_v3_design.md)를 따른다.

## 목표와 단계

| 단계 | 데이터 | 모델 | 진입 조건 |
| --- | --- | --- | --- |
| P0: 재현 기준선 | 기존 1,000/100/100, 길이 25–100 | Pair-Aware / GeometryFrame | 현재 결과와 평가 코드 고정 |
| P1: 파이프라인 프로파일 | 1만 개 수준의 실험 구조, 길이 ≤384 | 작은 v3 D/E 모델 | 분할 감사 통과, 샤드·배치·재시작 동작 확인 |
| P2: 구조 학습 | 5만–10만 개 수준을 목표로 수집·필터 | v3 D/E/F | validation에서 전역 구조 지표 개선 |
| P3: 생성 경로 | P2와 동일한 고정 split | v3 G: SE(3) flow | F가 독립 구조 지표에서 기준선 개선 |
| P4: pseudo-label | 검증된 고신뢰 예측 구조만 추가 | 가장 좋은 구조 모델 | P2/P3의 독립 검증 결과 확보 |

개수는 데이터 품질 감사 후의 **목표 범위**다. 실제 확보 가능한 유효 chain 수가 적으면 단계와 예산을 다시 잡는다. 구조가 많아도 homolog·동일 fold 누수가 있으면 평가가 과대 추정될 수 있다.

## 데이터 계약

한 chain 레코드는 `id`, `chain`, `sequence`, `path`, `split`, `cluster`, `residues`, `source`, `release_date`, `quality`, `checksum`을 가진다. `cluster`는 split 결정에 사용된 버전 고정 클러스터 ID다. structure는 residue index, atom mask, chain break를 보존해야 한다. 누락 잔기를 조용히 이어 붙이지 않는다.

1. 실험 구조를 먼저 수집하고 해상도·길이·비표준 잔기·원자 누락·chain break를 기록한다. 한 PDB에 여러 chain이 있어도 같은 구조나 유사 서열은 split을 넘지 않도록 묶는다.
2. sequence clustering과 구조 유사도 감사를 끝낸 **후** cluster 단위로 train/val/test를 할당한다. 기존 100개 test와 그 중복·유사 cluster는 새 train에서 제외한다. 새 test도 고정된 잠금 목록으로 보관한다.
3. `data/audit_manifest.py`로 필수 필드, 정확히 같은 서열의 split 누수, 선언된 cluster 누수, 길이·split 분포를 확인한다. 이 도구는 구조적 homology를 검출하지 않으므로 별도 구조 클러스터링이 필요하다.
4. manifest, 클러스터링 버전, 원본 구조 checksum, PLM 모델 ID/버전, 전처리 설정을 불변 스냅샷으로 기록한다. 검증 후 manifest를 수정하지 않고 새 버전을 만든다.
5. 예측 구조를 나중에 넣을 때는 출처와 신뢰도를 별도 필드로 유지한다. 저신뢰 잔기와 글로벌 형상이 이상한 예측은 제외한다. 실험 구조와 pseudo-label의 혼합 비율을 기록한다.

현재 기준선 manifest 감사 실행:

```bash
work/.venv/bin/python data/audit_manifest.py \
  --manifest data/manifest_1000train_200holdout.json \
  --report work/pdb_scale/manifest_audit.json
```

파일 경로 존재도 확인하려면 `--check-files`를 추가한다. 현재 1,200개 manifest에서는 train/val/test가 1,000/100/100이고, 정확한 서열·선언된 클러스터의 split 누수가 검출되지 않았다. 이것은 구조적 homology가 없다는 증명은 아니다.

## 저장·임베딩·로딩

- 기존 `data/embed_esm.py`는 전체 임베딩을 한 `.pt`에 모으므로 수만 chain에 적용하기 전에 바꾼다. Frozen PLM을 GPU에서 inference-only로 실행하고, 길이별 토큰 배치 후 128–512 chain 단위의 별도 샤드에 저장한다. 샤드는 원자적으로 쓰고 인덱스에 모델 버전·manifest SHA256·dtype·차원·샤드 checksum을 기록한다. 재시작 시 유효한 샤드는 건너뛴다.
- 구조는 mmCIF 파일을 학습 중 매 step 파싱하지 않는다. 한 번 정규화해 서열, N/Cα/C 좌표, frame, residue/atom/chain-break mask를 구조 샤드에 넣는다. 임베딩과 구조는 `id:chain` 키와 서열 checksum으로 결합한다. 불일치는 학습 전에 실패시킨다.
- GPU 학습기는 전체 코퍼스를 RAM/GPU에 올리지 않는다. `Dataset`이 인덱스를 읽고 필요한 샤드만 열며, 샤드 단위 캐시와 prefetch를 사용한다. train/val/test 샤드를 별도로 두고 학습 프로세스는 test 인덱스를 받지 않는다.
- 버킷은 실제 길이로 만들고, 배치당 `ΣL²`가 예산 이내가 되도록 묶는다. 긴 chain은 batch를 줄이고, 짧은 chain은 여러 개를 묶는다. crop을 쓰더라도 일정 비율의 전체 chain을 유지해 장거리 관계를 학습시킨다. crop은 chain break와 원자 mask를 함께 자른다.

## GPU 학습기

초기 프로파일 대상: H100 80GB 1장, BF16, 길이 128/256/384/512, pair width 64/128. 기록할 값은 `peak_allocated_GB`, `step_seconds`, `residues_per_second`, `L²_per_second`, 데이터 대기 시간, validation 시간이다. 이 측정 전에는 전체 학습 시간을 약속하지 않는다.

학습기 요구사항:

1. 단일 GPU와 `torchrun` DDP가 동일한 configuration을 사용한다. 각 rank는 train 샘플을 중복 없이 배분하며, validation 집계는 전체 val에 대해 한 번 수행한다.
2. BF16 autocast, gradient accumulation, clip, 길이별 배치, 필요 시 activation checkpointing을 쓴다. GPU 메모리가 불안정하면 먼저 batch의 `ΣL²` 한도를 낮춘다.
3. 구조 loss는 최종 recycle의 FAPE와 장거리 지표 대리 loss를 중심으로 두고, 보조 distogram·contact·orientation 및 중간 recycle loss를 분리 기록한다. 각 항의 mask와 사용된 residue/pair 수를 함께 로그에 남긴다.
4. `last` checkpoint에는 모델, optimizer, scheduler, GradScaler 사용 시 scaler, step/epoch, RNG 상태, sampler 위치, 설정, git commit, 데이터/PLM checksum을 저장한다. `best`는 validation 기준으로만 갱신하며 원자적으로 쓴다. 중단 후 resume은 같은 데이터 순서·설정으로 이어져야 한다.
5. 실험 설정은 한 파일에서 관리한다. 각 run은 설정 원본, 환경 버전, JSONL 학습 기록, validation per-chain 결과, best/last checkpoint를 보존한다. 다른 run의 기록을 덮어쓰지 않는다.

## 검증과 모델 선택

- 학습 도중에는 train/val만 로드한다. 일정 step마다 validation의 **독립 구조 지표**를 계산하고 사전에 고정한 선택식으로 best를 고른다. 기존 `RMSD/10 + 1−lDDT`를 연속성 기준으로 쓰되, v3에서는 lDDT/TM/장거리 접촉을 함께 보고 과적합·지표 충돌을 확인한다.
- 모델 구성, PLM 크기, loss 가중치, crop 정책, 샘플 수를 고르는 데 test를 쓰지 않는다. 최종 후보를 고정한 후 `geometry_compare.py`와 호환 가능한 공통 평가기로 test를 평가한다.
- TM 지표는 현재 저장소의 고정 대응 근사치와 별개로, 버전이 기록된 TM-align 계열 도구의 정식 점수를 도입한다. 평가 구현이 바뀌면 모든 기준선을 다시 평가한다.
- 최종 D/E/F/G는 최소 3개의 **독립적인 전체 학습 시드**로 비교한다. 단백질별 짝지은 변화, bootstrap 구간, 길이·접촉 밀도·2차 구조별 결과를 기록한다.

## 비용을 쓰기 전의 통과 기준

1. 새 manifest가 split 누수 감사와 구조 유사도 감사를 통과한다.
2. 1만 chain의 임베딩·구조 샤드를 중단 후 재시작해도 checksum과 샘플 수가 동일하다.
3. 1 GPU에서 길이별 메모리·처리량을 측정한다. 같은 checkpoint로 resume했을 때 데이터 순서와 다음 loss가 허용 오차 내에서 재현된다.
4. 작은 subset에서 overfit, frame의 SE(3) equivariance, SO(3) 유효성, mask, gradient를 확인한다.
5. 그다음 1만 chain 규모 D/E/F를 학습하고 validation에서 실제 전역 지표 개선이 있는 경우에만 5만–10만 규모와 다중 GPU로 확장한다.

## 현재 저장소에서 해야 할 구현 작업

1. `data/embed_esm.py`를 샤드형 frozen-PLM 임베더로 대체하고, 전처리 구조 샤드·인덱스 작성기를 추가한다.
2. `outputs/geometry_train.py`의 전체 RAM 적재·단백질 1개/step 학습을 Dataset/길이 버킷/DDP/BF16 기반 학습기로 교체한다. 기존 코드는 기준선 재현용으로 남긴다.
3. [v3 설계](flowfold_v3_design.md)의 D/E/F 모델을 별도 파일로 구현하고, 먼저 단일 GPU 측정 결과를 저장한다.

이 문서의 파이프라인은 FoldFlow-2의 frozen ESM·구조/서열 결합과 AlphaFold의 pair·IPA·recycling을 **우리의 sequence-only folding 평가**에 맞게 축소한 것이다. 참고: https://arxiv.org/html/2405.20313 및 https://www.nature.com/articles/s41586-021-03819-2
