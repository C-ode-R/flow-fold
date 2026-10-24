# 무작위 PDB 1,000 entry 조사

## 실행

저장소 루트에서:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-data.txt
.venv/bin/python -u scripts/pilot_pdb.py --sample-size 1000 --seed 20261023
```

기본 결과 폴더는 data/catalog/pilot1000이다. 다른 표본은 seed와 --out을 함께 바꾼다.
중단/실패 후에는 같은 명령을 실행한다. 고정 sample.json을 재사용하고 받은 mmCIF는
재다운로드하지 않으며 현재 감사 코드로 다시 검사한다. 손상 캐시는 재다운로드한다.
파일별 실패를 기록하고 계속 진행하며, 끝에 실패가 남으면 exit code 1을 반환한다.
동시에 같은 결과 폴더를 사용하는 프로세스를 실행하지 않는다.
원본 PDB entry 전체를 받으므로 큰 복합체 파일이 포함될 수 있다. 디스크/시간 사용량은
작은 체인 1,000개만 추출해 받는 경우와 다르다. 진행 로그를 보며 필요하면 중단할 수 있다.

## 표본과 분모

- 현재 RCSB에서 40–128잔기 실험 protein entity 전체를 검색한다.
- 그 entity를 포함하는 고유 PDB entry 목록에서 1,000개를 균등 무작위 비복원 추출한다.
- X-ray/단량체/품질 필터를 표본 선정 전에 적용하지 않는다. 따라서 전체 후보 entry 중
  초기 기준이 얼마나 통과하는지 조사할 수 있다.
- 각 entry의 길이 조건을 만족하는 모든 후보 entity와 복제 chain을 검사한다.
- 주 지표는 성공적으로 검사한 entry 중 적어도 한 후보 chain이 모든 자동 기준을 통과한 비율이다.
- entity 지표는 entry+entity ID로 복제 chain을 묶는다. 서로 다른 chain의 서로 다른 기준
  통과 결과를 합쳐 통과로 만들지 않는다. 한 chain이 모든 기준을 만족해야 한다.
- entity/chain 비율은 표본 내 기술 통계다. entity 균등 표본이 아니므로 전체 entity 모집단
  비율로 해석하지 않는다. 서열/상동성 중복 제거 전 결과이며 독립 단백질 수가 아니다.
- 네트워크/파싱 실패는 생물학적 탈락과 구분한다. 전체 1,000개 대비 확인된 통과율은
  미확인 항목을 남긴 하한이고, 성공 entry 분모의 비율도 함께 제시한다.

## 파일

- sample.json: 검색 조건, snapshot 시간, seed, 모집단 수, 고정 표본과 entity ID.
- search_response.json: 원 검색 응답.
- results.json: entry/chain별 관측률, 실험 품질, assembly, 탈락 사유와 오류.
- summary.json: 통과 수/분모/백분율, 방법별 entry 수, 기준별 entity 탈락 비율.
- data/raw/pdb/*.cif.gz: 원본 좌표. checksum은 results.json에 기록한다.

summary의 entity_failures_all_copies_fail_each_criterion은 기준별 탈락률이다.
같은 entity가 여러 사유에 해당할 수 있으므로 합계는 100%가 아니다.
not_yet_automated에 있는 wwPDB validation, ligand/핵산 의존성, 막·tag·fragment 맥락은
아직 미판정이다. 현재 통과율은 **자동 기준의 잠정 통과율**이며 최종 학습 채택률이 아니다.
실험 기준은 data_criteria_v1.md를 따른다. NMR은 첫 모델만 관측 통계를 기록하고
method 기준에서 보류한다. NMR ensemble 품질을 판정한 결과가 아니다.

## 로컬 검증

```bash
.venv/bin/python -m unittest discover -s tests -v
```

집계의 복제/실패 처리와 로컬 원본을 이용한 재실행을 검사한다. 새 네트워크 수집은 하지 않는다.
