# Elice 첫 GPU 학습 실행

서버: `USER@GPU`, 작업 폴더: `~/foldflow`.
A100 MIG 2g.20gb / 드라이버 535.183.06 / Python 3.14.9 /
PyTorch 2.11.0+cu126. 시스템 toolkit 12.8과 독립적인 wheel runtime 12.6을 사용한다.
GPU 행렬 연산과 backward 및 full-model preflight가 통과했다.
서버 단위 테스트는 23개 중 22개 통과, 원본 pilot fixture가 없는 1개 skip.

캐시: train 1,989 + validation 249, 39개 파일 checksum 검증 완료.
Test 원본/캐시는 전송하지 않았다.

동일 train 입력의 3 warmup + 10 update 측정:

| precision | 초/update | samples/s | peak allocated GPU GB |
|---|---:|---:|---:|
| FP32 | 0.1660 | 48.20 | 3.459 |
| BF16 trunk | 0.1676 | 47.72 | 2.884 |

배치 8, FP32, 최대 100 epoch, validation patience 15로 새 모델 학습을 실행했다.
과적합 checkpoint는 재사용하지 않는다. Test는 별도 최종 평가까지 열지 않는다.

```bash
ssh USER@GPU
cd ~/foldflow
tail -f gpu_training.log
cat outputs/fold150_full_v1/status.json
```

`status.json`은 학습 종료/중단 시 생성된다. 학습 중에는 `history.jsonl`과 로그를 본다.
체크포인트는 `outputs/fold150_full_v1/last.pt`, validation-selected `best.pt`.
런처 PID는 `gpu_training.pid`이며 실제 Python trainer PID와 다를 수 있다.
설정은 `outputs/fold150_full_v1/config.json`, throughput 실측은 `outputs/preflight_fp32.json`,
`outputs/preflight_bf16.json` 및 실행 스크립트의 `outputs/fold150_full_v1_preflight.json`.
