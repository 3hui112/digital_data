# XAI639 Project Mamba — Team 6 (T4 Frequency-Band Reliance)

Phase 1 공통 실험과 T4 H2 파일럿 실험의 코드와 결과입니다. 보고서(PDF/HTML)와 학습된 체크포인트(`best.pt`, `.joblib`)는 이 저장소에 넣지 않았습니다.

**모든 수치는 validation(subject 5명: 17, 21, 25, 26, 30)입니다. 공식 test set은 한 번도 평가하지 않았습니다.**

## 구조

| 경로 | 내용 |
|---|---|
| `mamba_starter/` | 공식 starter v1.1-student, **수정 없음** (MANIFEST 해시 일치, pytest 47개 통과). Apache 2.0, `NOTICE` 참고 |
| `run_phase1.sh`, `run_remaining.sh` | (모델, seed)마다 별도 프로세스로 starter `train.py`/`train_ema.py` 실행 → `runs/<model>/seed<S>/` |
| `phase1/` | Phase 1 분석 코드 (starter를 import만 함): `signal_tasks.py`(Task 1.1–1.3), `summarize.py`(1.4 표·confusion), `dissect.py`(1.5) |
| `runs/` | Phase 1 학습 결과: mamba, lti, cnn, lstm, ema × seed 0–2 (`summary.json`, `history.json`, `validation.json`, `config.json`, `split.json`) |
| `results/`, `figs/`, `logs/` | Phase 1 수치(JSON), 그림, 학습·분석 로그 |
| `h2_check/` | T4 H2 파일럿 실험 (아래 참고) |

## 환경과 데이터

- 실행 환경: Python 3.11, PyTorch 2.6.0 (CUDA 12.4), RTX 3090. starter 검증 환경(3.12 / 2.5.1 CPU)과 다릅니다.
- venv: `uv venv ~/.venvs/mamba-dd --python 3.11` 후 torch(cu124)와 `mamba_starter/requirements.txt` 설치. 스크립트는 `VENV`(기본 `~/.venvs/mamba-dd`)를 씁니다.
- 데이터: UCI HAR은 저장소에 없습니다. `run_phase1.sh`가 `./data`에 자동으로 받거나, 이미 있으면 `DATA=<UCI HAR Dataset 폴더의 상위 폴더>`로 지정합니다.

## 재현

커밋된 `runs/`와 `h2_check/runs/`에는 결과 JSON만 있고 체크포인트는 없습니다. 그대로 두면 `run_phase1.sh`는 `summary.json`이 있는 run을 건너뛰고, `h2_check/run_all.sh`는 출력 폴더가 이미 있어서 멈춥니다. 다시 학습하려면 커밋된 결과를 옆으로 옮긴 뒤 실행하세요.

```bash
# Phase 1: 학습(모델·seed당 1 프로세스) + 분석 + 표/그림
mv runs runs_committed
bash run_remaining.sh <GPU index>
# H2 파일럿: 파일럿 E1은 Phase 1 체크포인트가 필요하고, Check A/B는 12 run을 GPU 두 장에서 학습
python h2_check/pilot_e1/e1_quick.py
mv h2_check/runs h2_check/runs_committed
bash h2_check/run_all.sh <GPU for A> <GPU for B> && python h2_check/evaluate_checks.py
```

`CUDA_DEVICE_ORDER=PCI_BUS_ID`가 걸려 있어 GPU 번호는 `nvidia-smi`와 같습니다.

## Phase 1 결과 요약 (3 seed mean ± sample std)

| 모델 | val acc (%) | val macro-F1 (%) |
|---|---|---|
| Mamba (selective) | 95.07 ± 0.41 | 95.48 ± 0.41 |
| Mamba-LTI | 95.71 ± 0.22 | 96.09 ± 0.21 |
| 1D CNN | 92.48 ± 1.22 | 92.95 ± 1.22 |
| LSTM | 92.06 ± 1.37 | 92.49 ± 1.28 |
| EMA bank + LogReg | 89.85 | 89.54 |

자세한 수치는 `results/phase1_models.json`, `results/phase1_signal.json`, `results/phase1_dissect.json`.

## H2 파일럿 (`h2_check/`)

제안서 초안의 H2는 "selective Mamba가 LTI보다 3–8 Hz에 더 의존한다"입니다. 판정은 seed별 동적 클래스(WALK/UP/DOWN) F1 하락 차이 q<sub>s</sub>로 하고, δ = 1%p입니다. 결과를 보기 전에 정한 판정 기준은 `h2_check/PREREGISTRATION.md`에 있습니다. 필터는 Butterworth band-stop N=4, `sosfiltfilt`, 정규화 전 9채널 전체에 적용합니다.

1. **파일럿 E1** (`pilot_e1/`): Phase 1 체크포인트에 7개 band를 하나씩 적용했습니다. 세 모델 모두 하락이 1–2 Hz에서 가장 크고(selective 6.2, LTI 4.1, CNN 11.9%p), 3–8 Hz와 8 Hz 이상은 ≈ 0입니다. 그래서 원래 H2는 판단 불가입니다(q<sub>s</sub> = −0.36, −0.49, +0.44).
2. **Check A**: 파일럿에서 나온 "1–2 Hz에서 selective > LTI"를 새 seed 3–5로 확인했습니다. q<sub>s</sub> = −2.48, −3.15, +3.47로 **재현되지 않았습니다(판단 불가)**.
3. **Check B**: 0.3–3 Hz를 지운 데이터로 재학습한 뒤 3–8 Hz를 추가로 지웠습니다. q<sub>s</sub> = +2.70, +0.75, +0.88로 방향은 가설대로지만 δ 미만이 있어 **판단 불가**입니다.
4. **대역 간 중복:** 3–5 Hz를 지웠을 때 동적 F1 하락이 원래 모델에서는 ≈ 0이고, 저주파 없이 학습한 모델에서는 13–19%p입니다. 3–8 Hz에도 정보는 있지만, 저주파가 있으면 모델이 쓰지 않습니다.

5. **3–8 Hz 동시 제거 (탐색적, 사전 등록 아님):** `exploratory_joint_3_8.py`로 위 모든 checkpoint에서 3–8 Hz를 한 번에 지웠습니다. 하락이 따로 지운 하락의 합보다 커서(Check B macro-F1 18–28%p) 3–5 Hz와 5–8 Hz 사이에도 중복이 있습니다. q<sub>s</sub>는 파일럿 −5.76, −4.48, +0.77, Check A −3.92, −0.62, +4.43, Check B +15.61, +0.70, +0.49로 모두 판단 불가입니다.

전체 수치는 `h2_check/exploratory_joint_3_8.txt`, `h2_check/results.txt`, `h2_check/results.json`, `h2_check/pilot_e1/e1_quick.json`에 있습니다.
