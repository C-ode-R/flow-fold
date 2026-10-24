# 데이터 채택 기준 v1 — 독립 단일 체인 백본 예측

2026-10-23 결정안. 아래 숫자는 프로젝트의 초기 운영 기준이며 PDB의 공식 합격 기준이 아니다.
모델의 첫 과제는 40–128잔기 서열에서 N·Cα·C·O 백본을 예측하는 것이다.
원본은 보존하고 core / review / out_of_scope 목록을 분리한다.

## 1. 입력과 감독

- 전체 deposited sequence 길이 40–128. 관측된 잔기만 이어 붙이거나 길이를 맞추려고 crop하지 않는다.
- entity_poly_seq의 전체 서열과 label_seq_id를 기준으로 원자 좌표를 대응한다. auth 번호의 간격을 결측으로 오인하지 않는다.
- 초기 core는 표준 아미노산 20종만 포함한다. 변형 잔기, 서열 미상, microheterogeneity는 review로 보낸다. MSE 등을 자동 치환하지 않는다.
- 양의 occupancy를 가진 N·Cα·C가 모두 있는 잔기가 전체의 95% 이상이어야 core 후보가 된다.
- 결측 잔기는 입력 서열에 남긴다. frame·atom·pair 손실에는 각각 유효 마스크를 사용한다. 산소 O 결측만으로 전체를 버리지 않는다.
- altloc는 잔기별 backbone occupancy 합이 가장 큰 conformer를 택하고 공통 원자와 결합한다. 다른 conformer를 원자별로 섞지 않는다. 동률이면 ID 순으로 택하고 기록한다.
- 연속 label_seq_id의 C–N 거리가 1.1–1.8 Å 밖이면 review한다. 이 검사는 전체 입체화학 검증을 대신하지 않는다.
- 누락 구간을 가로질러 펩타이드 결합 손실을 만들지 않는다.

## 2. 실험 품질

- 첫 core 수집은 X-ray, 해상도 ≤2.5 Å, R_free ≤0.30을 사용한다.
- R_free/해상도 누락은 실패로 단정하지 않고 review에 보관한다.
- cryo-EM은 별도 review pool로 유지한다. global resolution만으로 작은 체인의 국소 정확성을 판단하지 않는다. 추후 map-model/국소 품질 검토 절차를 정한 뒤 편입한다.
- NMR은 ensemble pool로 별도 보관한다. 첫 모델을 대표 정답으로 임의 확정하지 않는다.
- core 확정 전에 wwPDB validation의 clash, backbone conformation, density-fit 이상을 확인한다. 해상도·R_free만 통과한 상태는 최종 채택이 아니다.
- 실험 구조의 B-factor를 AFDB의 pLDDT로 해석하지 않는다.

## 3. 구조 맥락

- asymmetric unit의 chain 개수로 단량체를 판정하지 않는다.
- 대상 chain을 포함하는 biological assembly 주석이 일관되게 단량체인 경우만 core 후보로 둔다. 다중체·assembly 누락·주석 불일치는 review한다.
- 다중체에서 얻은 chain이라고 스스로 접히지 못한다고 단정하지 않는다. 첫 독립 체인 과제에서 사용을 보류하는 것이다.
- ligand/metal/nucleic acid의 entry 단위 존재 여부와 대상 chain의 실제 접촉/의존성을 구분한다. 물·완충액·이온이 존재한다는 이유만으로 제외하지 않는다.
- 결합 맥락, 막 환경, fusion/tag/fragment 여부는 최종 core 확정 전 확인한다. deposited sequence는 자연 단백질 전장과 같다는 보장이 없다.
- 따라서 현재 audit의 eligible_pending_validation_and_context는 최종 학습 채택 상태가 아니다.

## 4. 중복과 분할

- 같은 서열의 여러 chain/entry를 독립적인 학습 사례 수로 세지 않는다. 우선 한 대표 구조를 선택하고 나머지는 대체 관측으로 연결한다.
- 전체 후보를 확보한 뒤 sequence clustering을 수행하고 클러스터 단위로 split한다. 초기 제안은 identity 30%, 양쪽 coverage 80%이며 실제 실행 옵션과 추가 split 간 검색을 별도로 검증한다.
- 이 기준만으로 원거리 상동성/구조 유사성이 제거됐다고 주장하지 않는다.
- train/validation/test는 중복 제거 후 고정한다. 500개 등 목표 수량을 먼저 강제하지 않는다. 날짜·출처·선정 사유·checksum·정제 버전을 남긴다.
- PLM의 서열 사전학습 포함 여부와 supervised structure split 누수는 따로 기록한다.

## 5. 현재 원본 조사 결과

data/catalog/20261023T032824Z/audit.json 기준:

- 원본 20 entry, 후보 chain 84개. 동일 entity의 복제 chain을 포함한다.
- 54 chain이 backbone 관측률 95% 이상, 30 chain이 미달.
- 84 chain 모두 대상 assembly가 다중체로 주석돼 review로 분류됐다.
- 실험 방법은 X-ray 1 entry, cryo-EM 19 entry.
- 예: 10CY C/D chain은 각각 95잔기이며 backbone 관측률은 90/95, 89/95다.
- 이 표본은 ID 정렬의 첫 20개이므로 전체 통과율·실험 방법 분포를 추정하는 데 쓰지 않는다.
- 파싱 오류 0개. 현재 최종 채택된 학습 구조는 0개다.

다음 실행: core 검색을 위한 X-ray/품질/assembly 메타데이터 필터를 적용하고,
그 후보에서 고정 seed로 다양한 entity를 추출해 수백 개 규모의 두 번째 파일럿을 구성한다.
필터별 잔존 수와 별도 review 사유를 보고한 뒤 대량 다운로드를 시작한다.

## 재현

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-data.txt
.venv/bin/python scripts/audit_pdb.py --catalog data/catalog/20261023T032824Z
```

## 근거 자료

- https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/biological-assemblies
- https://pdb101-east.rcsb.org/learn/guide-to-understanding-pdb-data/primary-sequences
- https://www.rcsb.org/docs/general-help/assessing-the-quality-of-3d-structures
- https://mmcif.k8s.wwpdb.org/docs/user-guide/guide.html
