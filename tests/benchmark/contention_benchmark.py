"""OCR이 게임에 주는 CPU 부하를 어림하는 대리 측정 (Phase 2 3단계).

실제 게임을 돌릴 수 없으므로 "가짜 게임 루프"로 대신한다. 60fps 게임처럼 프레임마다 정해진 양의
CPU 작업(여러 작업 스레드가 나눠 하는 행렬 곱)을 하고 남은 시간은 쉰다. 프레임 작업 시간은 모든
작업 스레드가 끝날 때까지의 시간이다. OCR 없이 잰 프레임 작업 시간과, 다른
프로세스에서 OCR을 쉬지 않고 돌릴 때의 프레임 작업 시간을 비교한다.

- 늘어난 작업 시간은 CPU 코어·캐시·전력 한도를 OCR과 나눠 쓴 영향이다.
- 부하가 이어지면 CPU 부스트 클럭이 떨어지므로(전력 한도), OCR을 멈춘 뒤에도 한 번 더 재서
  (OCR 없음 → OCR 있음 → OCR 없음) 앞뒤 평균과 비교한다. 순서 때문에 생기는 차이를 빼기 위함이다.
- GPU 경합(게임 렌더링과 OCR이 GPU를 나눠 씀)은 이 측정으로 알 수 없다.
- 실제 게임은 여러 스레드를 쓰고 GPU에 의존하므로, 이 값은 경향을 보는 참고값이다.

실행 예:

    python tests/benchmark/contention_benchmark.py --ocr-python .venv/Scripts/python.exe -- --device cpu --scenes box_ja
"""

import argparse
import json
import os
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from pathlib import Path

# 가짜 게임은 단일 스레드로 계산한다 (numpy가 여러 코어를 쓰면 측정이 흐려짐). numpy를 가져오기 전에 정한다.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

# 60fps 한 프레임 시간 (초)
FRAME_SECONDS = 1 / 60
# OCR 없이 프레임 작업이 차지할 목표 시간 (초). 한 프레임의 약 절반.
TARGET_WORK_SECONDS = 0.008
# 보정할 때 반복해서 재는 횟수 (중앙값을 쓴다)
CALIBRATION_TRIALS = 15
BENCHMARK = Path(__file__).with_name("ocr_benchmark.py")


def _work(matrix: np.ndarray, repeats: int) -> None:
    """작업 스레드 하나의 한 프레임 분량 계산 (numpy 행렬 곱은 GIL을 풀어 스레드끼리 함께 돈다)."""
    for _ in range(repeats):
        matrix @ matrix


def _frame(pool: ThreadPoolExecutor, matrices: list[np.ndarray], repeats: int) -> float:
    """모든 작업 스레드가 한 프레임 분량을 끝낼 때까지의 시간(초)."""
    started = time.perf_counter()
    for future in [pool.submit(_work, matrix, repeats) for matrix in matrices]:
        future.result()
    return time.perf_counter() - started


def calibrate(pool: ThreadPoolExecutor, matrices: list[np.ndarray]) -> int:
    """OCR 없이 한 프레임 작업이 목표 시간 정도 걸리도록 스레드별 반복 횟수를 정한다 (비례 보정)."""
    repeats = 4
    for _ in range(3):
        measured = statistics.median(_frame(pool, matrices, repeats) for _ in range(CALIBRATION_TRIALS))
        repeats = max(1, round(repeats * TARGET_WORK_SECONDS / measured))
    return repeats


def game_loop(pool: ThreadPoolExecutor, matrices: list[np.ndarray], seconds: float, repeats: int) -> dict:
    """가짜 게임 루프를 돌려 프레임 작업 시간(ms)의 통계를 돌려준다."""
    works: list[float] = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        work = _frame(pool, matrices, repeats)
        works.append(work)
        rest = FRAME_SECONDS - work
        if rest > 0:
            time.sleep(rest)
    works.sort()
    return {
        "frames": len(works),
        "work_p50_ms": round(statistics.median(works) * 1000, 2),
        "work_p95_ms": round(works[int(len(works) * 0.95)] * 1000, 2),
        # 작업만으로 한 프레임(16.7ms)을 넘긴 비율: 60fps를 지키지 못한 프레임
        "over_budget_ratio": round(sum(w > FRAME_SECONDS for w in works) / len(works), 3),
    }


def main() -> int:
    """OCR 없이, 그리고 OCR을 계속 돌리는 동안 가짜 게임 루프를 재고 JSON을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ocr-python", required=True, help="OCR을 돌릴 가상환경의 python")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--label", default="")
    # 가짜 게임의 작업 스레드 수 (요즘 게임은 보통 여러 코어를 쓴다)
    parser.add_argument("--game-threads", type=int, default=6)
    # OCR 프로세스의 OpenMP 스레드 수 (0이면 정하지 않음). oneDNN은 cpu_threads 대신 이 값을 따른다.
    parser.add_argument("--ocr-omp-threads", type=int, default=0)
    parser.add_argument("ocr_args", nargs=argparse.REMAINDER, help="-- 뒤에 ocr_benchmark.py 인자")
    args = parser.parse_args()
    ocr_args = [a for a in args.ocr_args if a != "--"]

    rng = np.random.default_rng(1)
    # 작업 스레드마다 따로 쓰는 행렬
    matrices = [rng.random((160, 160)) for _ in range(args.game_threads)]
    pool = ThreadPoolExecutor(max_workers=args.game_threads)
    repeats = calibrate(pool, matrices)
    results: dict = {
        "label": args.label,
        "ocr_args": ocr_args,
        "game_threads": args.game_threads,
        "ocr_omp_threads": args.ocr_omp_threads,
        "repeats": repeats,
    }
    results["idle"] = game_loop(pool, matrices, args.seconds, repeats)

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    # OCR 프로세스는 단일 스레드 제한을 물려받지 않게 한다.
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        env.pop(name, None)
    if args.ocr_omp_threads:
        env["OMP_NUM_THREADS"] = str(args.ocr_omp_threads)
    process = subprocess.Popen(
        [args.ocr_python, str(BENCHMARK), *ocr_args, "--loop-seconds", str(args.seconds + 30)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env=env,
    )
    try:
        # 모델을 불러오고 예열이 끝날 때까지 기다린다.
        for line in process.stdout:
            if line.strip() == "READY":
                break
        time.sleep(2.0)
        results["with_ocr"] = game_loop(pool, matrices, args.seconds, repeats)
    finally:
        process.kill()
        process.wait(30)
    try:
        # OCR을 멈춘 뒤 다시 잰다 (순서 효과 확인용).
        time.sleep(1.0)
        results["idle_after"] = game_loop(pool, matrices, args.seconds, repeats)
    finally:
        pool.shutdown()
    # OCR 앞뒤 OCR 없는 구간의 중앙값 평균
    idle = (results["idle"]["work_p50_ms"] + results["idle_after"]["work_p50_ms"]) / 2
    busy = results["with_ocr"]["work_p50_ms"]
    results["work_p50_increase_percent"] = round((busy / idle - 1) * 100)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
