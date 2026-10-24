# 데이터 수집 1단계

목적: 실제 학습용 데이터셋을 확정하기 전에 출처와 원본을 확보한다.

- 출처: RCSB PDB 실험 구조. AFDB는 현재 수집하지 않는다.
- 검색 단위: protein polymer entity. 전체 deposited sequence 길이 40–128.
- 다운로드 단위: PDB entry의 원본 mmCIF.gz. 파일에는 다른 체인·리간드가 포함될 수 있다.
- 검색 JSON, 원 응답, 후보 ID, 다운로드 URL, 파일 SHA256을 보관한다.
- 처음 20개 entry는 ID 정렬 순으로 고른 연결 확인용 표본이다. 대표 표본이 아니다.
- 후보 개수는 독립 단백질 수나 학습 가능 구조 수가 아니다.
- 현 단계에서는 실험 방법·해상도 필터, 단량체 판정, 잔기/원자 마스크, 중복 제거, split을 적용하지 않는다.

## 다음 단계에서 결정할 사항

1. 실험 방법별 품질 기준과 backbone 관측률.
2. biological assembly를 이용한 단량체/복합체 의존성 분류.
3. 전체 서열과 관측 잔기의 대응, altloc, 변형 잔기, chain break 처리.
4. 서열 클러스터 단위 중복 제거와 train/validation/test 고정.

검색 길이를 맞추기 위해 결측 잔기를 제거하거나 긴 체인을 임의로 자르지 않는다.
전처리 시 비표준 잔기, NMR 다중 모델, 리간드 의존성 등은 명시적으로 기록한다.

## 실행

```bash
python3 scripts/collect_pdb.py --pilot-entries 20
```

재실행하면 새 검색 snapshot을 만들고 기존 원본 파일은 재사용한다.
수집 결과는 data/catalog/, 좌표 원본은 data/raw/pdb/에 보관하며 Git에서 제외한다.

공식 문서:
- https://search.rcsb.org/
- https://data.rcsb.org/
- https://www.rcsb.org/docs/programmatic-access/file-download-services

## 1차 감사 완료

20 entry의 후보 chain 84개를 감사했다. 채택 정책과 실측 결과는
[data_criteria_v1.md](data_criteria_v1.md)에 기록했다.
자동 감사 결과는 각 catalog의 audit.json에 저장한다.
