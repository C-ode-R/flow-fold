# 전체 단량체 후보 병렬 검사

## 실행 (fish, CPU 서버 프로젝트 폴더)

기존 전체 색인 data/catalog/local1000/index.sqlite를 읽기 전용으로 사용한다.
원본 아카이브와 색인은 갱신하지 않는다. worker 수는 기본 2, 이 서버에서는 4부터 측정한다.

```fish
uv run python -m unittest discover -s tests -v
nohup uv run python -u scripts/screen_archive.py \
  --index data/catalog/local1000/index.sqlite \
  --out data/catalog/full_screen_v1 \
  --workers 4 --contact-cutoff 4.5 \
  > full_screen.log 2>&1 < /dev/null &
set screen_pid $last_pid
echo $screen_pid > full_screen.pid
disown $screen_pid
```

```fish
tail -f full_screen.log
cat data/catalog/full_screen_v1/summary.json
```

## 대상과 조건

- 약26만 등록 구조의 기존 색인에서 전체 길이40–128잔기 polypeptide(L) entity 전부를 선정한다.
- X-ray, 기록된 모든 정제 블록의 해상도 ≤2.5 Å, 모든 기록된 R_free ≤0.30.
- 대상 체인이 포함된 모든 assembly의 oligomeric_count=1. 누락/불일치는 보류한다.
- 표준20잔기, N/CA/C 관측률 ≥95%, 연속 관측 잔기의 C–N 거리1.1–1.8 Å.
- 메타데이터 조건을 먼저 검사한다. 보류된 파일은 원자 좌표를 저장하지 않는다.
- 이후 후보 체인·비수성 리간드 원자만 유지하는 스트리밍 파서로 기하를 검사한다.
- 거대 복합체라도 원자 테이블 전체를 Python 객체로 보존하지 않는다. 파일 자체는 끝까지 순차 스캔한다.

## 저장 항목과 리간드 정의

- candidates.jsonl: 통과 chain별 전체 서열, 관측 백본 N/CA/C/O 좌표(Å), 원본 경로,
  결측 label_seq_id, auth 번호와 insertion code, 품질 지표, assembly 주석, 리간드 메타데이터.
- review.jsonl: 보류 사유. screen_stage=metadata인 경우 geometry_checked=false이며
  관측률·원자 수 등의 미검사 항목은 기록하지 않는다.
- errors.jsonl: 파싱/좌표 변환 등 처리 오류. 생물학적 탈락과 분리한다.
- screen.sqlite: 파일별 checkpoint. 같은 코드·설정·아카이브에서 재실행하면 완료 파일은 건너뛴다.
- source_summary.json, config.json, summary.json: 전체 모수·색인상태·코드 hash·설정·최종 집계.
- preliminary_accepted_entities: 최소 한 chain이 모든 조건을 만족한 entry+entity 수.
  같은 entity 복제 chain은 수를 늘리지 않지만 동일/상동 서열은 아직 중복 제거하지 않았다.

리간드: _entity.type이 non-polymer 또는 branched인 원자의 성분. HOH/DOD와 수소/중수소,
occupancy≤0을 제외한다. 단원자 이온·금속과 당류도 포함한다. DNA/RNA 등 polymer는
리간드 목록으로 치환하지 않는다. 성분 ID, CCD 이름·type·formula(있을 때), asym/auth ID를 보존한다.

entry_has_nonwater_ligand는 원본 파일 내 존재 여부다. biological_assembly_ligands는
해당 biological assembly에 포함된 리간드 인스턴스다. has_contacting_ligand는 그 중
대상 단백질의 중원자와 4.5 Å 이하 거리인 리간드가 있는지다. 접촉 잔기는 label_seq_id로
저장하고 백본 auth 번호 대응도 함께 남긴다. 임계값은 운영상 기하 정의이며 결합 의존성 판정이 아니다.

assembly 연산의 범위/곱을 전개하고 표적 체인과 리간드 좌표 양쪽에 변환을 적용한다.
서로 다른 공간의 좌표를 직접 비교하지 않는다. 지원하지 않는 연산/필수 변환 누락은 error로
기록하며 '리간드 없음'으로 치환하지 않는다. 리간드 altloc는 양의 occupancy인 모든 conformer의
접촉 합집합이다. 단백질은 잔기별 일관된 altloc를 선택한다. 백본 좌표는 원본 파일 기준 좌표다.

## 운영

원본/색인 경로가 다르거나 파일이 바뀌었으면 오류를 보고하고 중단한다.
전체 색인이 완성된 상태에서 실행한다. 실행 중 원본을 rsync로 갱신하지 않는다.
같은 --out으로 여러 프로세스를 동시에 실행하지 않는다. 코드/설정/원본이 바뀌면 새 --out을 쓴다.
파일 단위로 결과를 저장하며 오류 파일은 재실행 시 다시 검사한다.
작업자 강제 종료는 명확한 오류로 종료하므로 로그를 확인하고 RAM/worker 수를 조절한다.
출력은 원본을 참조하는 예비 학습 후보다. validation 추가 검사, 서열 클러스터링/중복 제거와
train/validation/test 고정은 후속 단계다. 리간드가 있다는 이유로 후보에서 제외하지 않는다.
