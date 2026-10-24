# 데이터 중복 제거와 고정 분할 — 2026-10-24

서버 최종 데이터셋: /var/projects/foldflow/data/dataset_v1_final

## 실제 결과

- 입력: 통과 chain 관측6,938개, entry+entity4,577개.
- 전체 서열 완전 일치로 중복 제거: 고유 서열2,487개.
- 제거한 중복 관측4,451개는 observations.jsonl과 각 대표 구조의 observations 링크로 보존.
- train1,989 / validation249 / test249.
- 연결 성분801개. train644 / validation78 / test79개 성분. 최대 성분125서열.

## 대표 구조 선정

관측 backbone 비율 내림차순 → 해상도 오름차순 → R_free 오름차순 → 관측 산소 수
내림차순 → PDB entry/chain ID 순으로 한 구조를 고른다. 같은 서열의 다른 결합/리간드
상태가 동일 구조라는 가정은 하지 않는다. 모든 관측의 ligand_metadata는 observations.jsonl에
보존하며 대표 구조의 리간드 정보와 섞지 않는다.

## 그룹과 분할

MMseqs2 fcf516aba34be4569b3eb5bda2b9db9e26a8f278의 CPU 검색을 사용했다.
--prefilter-mode 2로 kmer prefilter를 비활성화하고 서열 전체 상호 검색을 수행했다.
--alignment-mode3, --seq-id-mode0, identity≥0.30, query/target coverage 모두≥0.80,
E-value 상한1e6. 저복잡도 masking과 composition bias correction은 이 감사에서 껐다.
실제 실행 인수와 버전은 provenance.json에 저장했다.

검출된 유사 서열 edge와 동일 PDB entry 관계를 합쳐 연결 성분을 만들고 성분 전체를
하나의 split으로 보낸다. seed20261024, 큰 성분 우선,80/10/10 목표로 할당했다.
원거리 상동성·구조적 유사성의 부재를 보증하는 기준은 아니다.
이전 개발 실행 data/dataset_v1은 SUPERSEDED로 표시했다. 학습에는 final 경로만 사용한다.

## 산출물

- train.jsonl, validation.jsonl, test.jsonl: 대표 구조의 서열/백본좌표/마스크 관련 원본 정보와 리간드 메타데이터.
- manifest.jsonl: 전체 분할, sequence_id, cluster_id, 대표 및 중복 관측 링크.
- observations.jsonl: 모든 원 관측의 ligand metadata와 원본 경로/line. 중복 backbone 좌표는 원 candidates.jsonl에서 참조한다.
- sequences.fasta: 고유 서열.
- all_pairs.tsv: 검사한 조건에서 검출된 MMseqs2 alignment 결과.
- provenance.json, mmseqs.log: 원본 SHA256·대표 선정 규칙·tool 버전·명령·로그.
- summary.json, checksums.json, COMPLETE: 집계·산출물 checksum·완료표시.

파일 checksum, 대표 서열의 유일성, 분할별 수량, 동일 entry 분리 여부,
검출된 유사 서열의 split 간 연결0건과6,938개 관측 보존을 재검증했다.
기존 실험 데이터이므로 wwPDB validation 상세 검사·PLM 사전학습 중복은 아직 확인하지 않았다.

재생성 시 scripts/split_dataset.py를 사용하고 새 --out을 지정한다. 기존 고정 분할은 덮어쓰지 않는다.
