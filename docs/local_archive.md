# CPU 서버 로컬 아카이브 처리

네트워크 호출 없이 전체 polypeptide(L) entity 서열을 추출하고, 길이 40–128 후보에서
고정 seed로 1,000 entity를 균등 무작위 비복원 추출한다. 같은 entry의 복제 chain은 표본을
늘리지 않는다. 같은 서열의 서로 다른 entity/entry는 아직 중복 제거하지 않았다.
기존 API pilot은 entry 균등 표본이므로 두 통과율을 그대로 비교하지 않는다.

서버에 scripts/ 디렉터리 전체와 requirements-data.txt를 복사한다.
프로젝트 폴더에서 아래를 실행한다. --archive는 실제 다운로드 경로에 맞춘다.

```fish
python3 -m venv .venv
.venv/bin/pip install -r requirements-data.txt

nohup .venv/bin/python -u scripts/local_pdb_pipeline.py \
  --archive /실제/경로/pdb_archive/mmCIF \
  --out data/catalog/local1000 \
  --sample-size 1000 --seed 20261023 --workers 2 \
  > local_pipeline.log 2>&1 < /dev/null &
set pipeline_pid $last_pid
echo $pipeline_pid > local_pipeline.pid
disown $pipeline_pid
```

```fish
tail -f local_pipeline.log
ps -p (cat local_pipeline.pid) -o pid,etime,args
cat data/catalog/local1000/summary.json
```

## 동작과 재시작

1. divided/mmCIF의 *.cif.gz 파일을 나열한다. index.sqlite에 파일 경로·크기·mtime과
   추출 결과를 기록한다. 전체 원본은 압축 상태로 둔다.
2. deposited sequence를 entity_poly_seq에서 추출한다. 좌표 결측 때문에 서열을 줄이지 않는다.
   표준 외 잔기/서열 이질성은 FASTA에서 X로 보존하며 standard=false로 표시한다.
3. 현재 아카이브 파일 전체를 성공적으로 색인한 뒤 FASTA와 후보 목록을 만든다.
   파싱 실패가 있으면 보고하고 표본 생성 전에 중단한다. 실패 파일을 몰래 빼고 비율을 내지 않는다.
4. entity 단위 무작위 표본을 고정하고 같은 entry는 한 번 읽어 chain들을 감사한다.
5. entity의 적어도 한 chain이 모든 자동 기준을 동시에 통과해야 잠정 통과로 계산한다.

처음에는 압축 원본 전체를 읽고 파싱하므로 상당한 시간이 걸린다. 작업자마다 파일 하나를
메모리로 읽는다. 큰 복합체에서 메모리가 부족하면 --workers 1로 낮춘다. 여유가 있으면
4 등으로 늘릴 수 있으나 무조건 CPU 코어 수만큼 늘리지 않는다.
파일 목록은 mirror 완료 후 고정하고 처리 중 rsync 갱신을 돌리지 않는다.

중단되면 같은 명령을 다시 실행한다. 성공한 색인은 재사용하고, 실패/변경 파일은 재검사한다.
표본 감사는 다시 수행한다. 표본, seed, 파일 inventory가 달라지면 기존 결과와 섞지 않고
오류로 중단한다. 아카이브 갱신이나 다른 표본은 새 --out을 사용한다.
동시에 같은 --out에 두 프로세스를 실행하지 않는다.

## 결과

- index.sqlite: 재시작 가능한 파일별 서열 색인.
- protein_sequences.fasta: 전체 단백질 entity 서열. 자연 단백질의 전장 여부는 아직 미판정.
- candidates.jsonl: 40–128잔기 후보와 원본 경로.
- index_summary.json: 파일/서열/후보 수, 파싱 오류, 경로·크기·mtime inventory hash.
- sample.json: 고정 1,000 entity 목록, seed, inventory hash.
- audit.json: 표본별 chain 감사 상세와 실패 사유.
- summary.json: 잠정 통과율·기준별 탈락률·미판정 수.

inventory hash는 파일 내용 checksum이 아니다. 아카이브 원본 내용 전체 무결성을 검증하는
단계는 아니다. 통과율은 wwPDB validation, ligand/핵산 의존성, 막·fusion/tag 맥락과
상동성 중복 제거 전의 자동 기준 통과율이다. 최종 학습 채택률로 해석하지 않는다.

서열 추출·마스크·assembly 품질 기준은 data_criteria_v1.md를 따른다.
