"""번역 모델(llama.cpp 서버 + GGUF) 비교 측정 (Phase 2 4단계).

`translation_eval_en_ko.json`의 직접 작성한 영어 문장을 앱의 번역 어댑터(`ChatCompletionTranslator`)로
번역하고 다음을 잰다.

- 첫 번역까지 걸린 시간 (서버 시작·모델 로딩 포함)
- 한 줄씩 번역할 때의 지연 (박스에 대사 한 줄이 있는 경우)
- 전체를 묶어 번역할 때의 지연 (전체 화면처럼 텍스트가 많은 경우)
- 참고 번역과의 chrF (문자 n-그램 F 점수, 0~100). 참고 번역은 가능한 정답 중 하나라 보조 지표로만 쓴다.
- 영어가 그대로 남은 번역 수, 출력 형식 오류(다시 시도) 횟수, 번역 실패 수
- 모델을 불러온 뒤 늘어난 VRAM (nvidia-smi 전체 사용량 차이)

실행 예:

    python tests/benchmark/translation_benchmark.py --model Qwen3.5-9B-Q4_K_M.gguf

결과 JSON 한 줄(번역문 포함)을 표준 출력 마지막 줄에 쓴다.
"""

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from translator.adapters.translation.chat_completion import (  # noqa: E402
    ChatCompletionTranslator,
    ChatServerSettings,
)
from translator.adapters.translation.llama_server import LlamaServerProcess  # noqa: E402
from translator.core.cancellation import CancellationToken  # noqa: E402
from translator.core.errors import TranslationError  # noqa: E402
from translator.core.models import TranslationRequest  # noqa: E402

EVAL_FILE = Path(__file__).with_name("translation_eval_en_ko.json")
# 기본 llama.cpp·모델 위치 (앱의 기본 위치와 같음)
LLM_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "MultimodalTranslator" / "llm"
# chrF의 최대 문자 n-그램 길이와 재현율 가중치 (chrF 정의에서 정한 표준 값)
CHRF_MAX_N = 6
CHRF_BETA = 2
# 번역문에 영어가 남았다고 볼 연속 로마자 길이
LEFTOVER_LATIN = re.compile(r"[A-Za-z]{3,}")


def chrf(hypothesis: str, reference: str) -> float:
    """문장 하나의 chrF 점수 (0~100, 공백 무시)."""
    hyp, ref = "".join(hypothesis.split()), "".join(reference.split())
    precisions, recalls = [], []
    for n in range(1, CHRF_MAX_N + 1):
        hyp_grams = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        ref_grams = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        if not hyp_grams or not ref_grams:
            continue
        overlap = sum((hyp_grams & ref_grams).values())
        precisions.append(overlap / sum(hyp_grams.values()))
        recalls.append(overlap / sum(ref_grams.values()))
    if not precisions:
        return 0.0
    precision, recall = statistics.fmean(precisions), statistics.fmean(recalls)
    if precision + recall == 0:
        return 0.0
    beta2 = CHRF_BETA ** 2
    return 100 * (1 + beta2) * precision * recall / (beta2 * precision + recall)


def gpu_memory_used_mb() -> int | None:
    """GPU 전체 사용 메모리 (MB). 확인하지 못하면 None."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return int(output.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class _RetryCounter(logging.Handler):
    """어댑터가 남기는 '출력 형식 오류' 경고 수를 센다."""

    def __init__(self) -> None:
        """0부터 센다."""
        super().__init__(logging.WARNING)
        # 출력 형식 오류로 다시 시도한 횟수
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """출력 형식 오류 경고면 센다."""
        if "번역 출력이 올바르지 않습니다" in record.getMessage():
            self.count += 1


def translate_all(translator, sources: list[str], single: bool) -> tuple[list[str | None], list[float], int]:
    """원문을 한 줄씩 또는 한꺼번에 번역한다. (번역문 목록, 요청별 지연(ms), 실패 수)를 돌려준다."""
    outputs: list[str | None] = []
    latencies: list[float] = []
    failures = 0
    chunks = [[s] for s in sources] if single else [sources]
    for chunk in chunks:
        started = time.perf_counter()
        try:
            results = translator.translate_batch(
                [TranslationRequest(s, source_language="en") for s in chunk], CancellationToken()
            )
            outputs.extend(r.translated_text for r in results)
        except TranslationError:
            failures += 1
            outputs.extend([None] * len(chunk))
        latencies.append((time.perf_counter() - started) * 1000)
    return outputs, latencies, failures


def summarize(items: list[dict], outputs: list[str | None]) -> dict:
    """번역문 목록의 chrF 평균, 영어 남음 수를 계산한다."""
    scores = [chrf(o, i["reference"]) if o else 0.0 for i, o in zip(items, outputs)]
    leftovers = sum(1 for o in outputs if o and LEFTOVER_LATIN.search(o))
    return {"chrf": round(statistics.fmean(scores), 1), "english_left": leftovers}


def main() -> int:
    """모델 하나를 측정하고 결과 JSON을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="GGUF 파일 이름(기본 모델 폴더) 또는 경로")
    parser.add_argument("--server", default=str(LLM_DIR / "llama.cpp" / "b11557" / "llama-server.exe"))
    parser.add_argument("--label", default="")
    parser.add_argument("--repeats", type=int, default=2, help="한꺼번에 번역하기 반복 횟수")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    retry_counter = _RetryCounter()
    logging.getLogger("translator").addHandler(retry_counter)
    items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["items"]
    sources = [item["source"] for item in items]
    model = Path(args.model) if Path(args.model).is_file() else LLM_DIR / "models" / args.model

    vram_before = gpu_memory_used_mb()
    server = LlamaServerProcess(Path(args.server), model)
    started = time.perf_counter()
    translator = ChatCompletionTranslator(
        ChatServerSettings(server.start(), engine_name=server.model_name), is_server_alive=server.is_running
    )
    try:
        translator.translate_batch([TranslationRequest("Hello.", source_language="en")], CancellationToken())
        first_seconds = time.perf_counter() - started
        vram_after = gpu_memory_used_mb()
        single_outputs, single_ms, single_failures = translate_all(translator, sources, single=True)
        batch_runs = [translate_all(translator, sources, single=False) for _ in range(args.repeats)]
    finally:
        server.stop()

    batch_outputs = batch_runs[0][0]
    results = {
        "label": args.label or model.stem,
        "model": model.name,
        "first_translation_seconds": round(first_seconds, 1),
        "vram_increase_mb": (vram_after - vram_before) if vram_before is not None and vram_after is not None else None,
        "single": {
            "median_ms": round(statistics.median(single_ms)),
            "p95_ms": round(sorted(single_ms)[int(len(single_ms) * 0.95)]),
            "failures": single_failures,
            **summarize(items, single_outputs),
        },
        "batch": {
            "lines": len(sources),
            "total_ms": [round(sum(latencies)) for _, latencies, _ in batch_runs],
            "failures": sum(failures for _, _, failures in batch_runs),
            **summarize(items, batch_outputs),
        },
        "format_retries": retry_counter.count,
        "outputs": [
            {"id": item["id"], "single": s, "batch": b} for item, s, b in zip(items, single_outputs, batch_outputs)
        ],
    }
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
