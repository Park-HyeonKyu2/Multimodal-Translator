"""PaddleOCR 설정별 지연·정확도·자원 사용량을 재는 벤치마크 (Phase 2 3단계).

실제 바탕 화면은 개인정보 문제가 있고 설정마다 같은 화면을 다시 만들 수 없으므로, 정답 텍스트를
아는 합성 화면을 그려 쓴다. 장면은 다음과 같다.

- `box_ja`: 860 × 170 게임 대사 창 (그라데이션, 반투명 패널, 외곽선 일본어 두 줄 + 화자 이름)
- `box_en`: 같은 창에 긴 영어 한 줄 (타자 효과가 끝난 문장 길이)
- `desktop`: 3840 × 1080 (1920 × 1080 모니터 두 대) 바탕 화면 모양. 창 패널마다 12~28px의
  영어·일본어 UI 글자 약 100줄
- `monitor`: `desktop`의 왼쪽 모니터 한 대 (모니터별 OCR의 효과 확인용)

PySide6 없이 numpy, Pillow, PaddleOCR만으로 실행되므로 다른 PaddlePaddle 버전의 가상환경에서도
돌릴 수 있다. 앱의 `PaddleTextRecognizer`(입력 변환·출력 검증 포함)를 그대로 거쳐 잰다.

실행 예:

    python tests/benchmark/ocr_benchmark.py --device cpu --runs 3
    python tests/benchmark/ocr_benchmark.py --device gpu:0 --det-limit-type max --det-limit-side 1920

결과 JSON 한 줄을 표준 출력 마지막 줄에 쓴다. 이미지는 디스크에 저장하지 않는다.
"""

import argparse
import difflib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 저장소의 src를 가져올 수 있게 한다 (다른 가상환경에서 실행할 때).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from translator.adapters.ocr.paddle import PaddleOcrSettings, PaddleTextRecognizer  # noqa: E402
from translator.core.cancellation import CancellationToken  # noqa: E402
from translator.core.models import CapturedImage, ScreenRegion  # noqa: E402

# 글꼴 파일 (Windows 기본 글꼴)
FONT_JA = "C:/Windows/Fonts/YuGothM.ttc"
FONT_JA_BOLD = "C:/Windows/Fonts/YuGothB.ttc"
FONT_UI = "C:/Windows/Fonts/segoeui.ttf"
# 같은 줄로 인정할 최소 유사도 (정답 판정용)
LINE_MATCH_RATIO = 0.8
# 작은 글자로 따로 집계할 글자 크기 상한 (px)
SMALL_TEXT_PX = 14

# 바탕 화면 장면에 쓰는 영어·일본어 문구 조각 (직접 작성한 문장)
EN_WORDS = (
    "inventory quest settings village castle river forest potion shield sword journal map "
    "options audio display controls online friends market guild skill level status party "
    "battle reward chapter save load continue mission ancient gate moon harbor tower"
).split()
JA_PHRASES = (
    "設定を保存しました", "冒険を続けますか", "新しいクエストがあります", "城の門は北にあります",
    "回復薬を三つ手に入れた", "仲間が待っています", "次の町まで歩きましょう", "装備を確認してください",
    "今日は風が強いですね", "地図を開きますか",
)


@dataclass(frozen=True)
class Truth:
    """그린 텍스트 한 줄의 정답: 내용, 이미지 안 위치 (왼쪽, 위, 오른쪽, 아래), 글자 크기."""

    text: str
    box: tuple[int, int, int, int]
    size_px: int


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """글꼴을 연다."""
    return ImageFont.truetype(path, size)


def _draw_text(draw: ImageDraw.ImageDraw, xy, text, font, fill, truths, size_px, stroke=0) -> None:
    """텍스트를 그리고 실제로 그려진 범위를 정답에 추가한다."""
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))
    left, top, right, bottom = draw.textbbox(xy, text, font=font, stroke_width=stroke)
    truths.append(Truth(text, (left, top, right, bottom), size_px))


def _gradient(width: int, height: int, start, end) -> Image.Image:
    """가로 그라데이션 배경을 만든다."""
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for x in range(width):
        ratio = x / max(1, width - 1)
        color = tuple(round(a + (b - a) * ratio) for a, b in zip(start, end))
        draw.line([(x, 0), (x, height)], fill=color)
    return image


def scene_box(lines: tuple[str, ...], speaker: str, font_path: str) -> tuple[Image.Image, list[Truth]]:
    """게임 대사 창 장면 (860 × 170)."""
    image = _gradient(860, 170, (30, 45, 80), (90, 40, 90))
    draw = ImageDraw.Draw(image, "RGBA")
    for x in range(0, 860, 37):
        draw.line([(x, 0), (x + 60, 170)], fill=(255, 255, 255, 25), width=2)
    draw.rounded_rectangle([10, 10, 850, 160], radius=12, fill=(0, 0, 0, 150))
    truths: list[Truth] = []
    if speaker:
        _draw_text(draw, (32, 22), speaker, _font(FONT_JA_BOLD, 18), (255, 215, 120), truths, 18)
    for index, line in enumerate(lines):
        _draw_text(draw, (32, 60 + index * 40), line, _font(font_path, 28), (250, 250, 250), truths, 28, stroke=2)
    return image, truths


def scene_desktop(seed: int = 7) -> tuple[Image.Image, list[Truth]]:
    """바탕 화면 장면 (3840 × 1080): 모니터마다 창 패널 여러 개와 UI 글자."""
    rng = random.Random(seed)
    image = _gradient(3840, 1080, (40, 60, 90), (20, 30, 40))
    draw = ImageDraw.Draw(image, "RGBA")
    truths: list[Truth] = []
    # (왼쪽, 위, 너비, 높이) 창 패널 위치
    panels = [
        (40, 40, 860, 500), (960, 40, 900, 320), (960, 420, 900, 600), (40, 600, 860, 420),
        (1960, 40, 880, 460), (2900, 40, 900, 980), (1960, 560, 880, 480),
    ]
    sizes = (12, 13, 14, 16, 18, 22, 28)
    for panel_index, (left, top, width, height) in enumerate(panels):
        dark = panel_index % 2 == 0
        background = (32, 34, 40) if dark else (240, 240, 236)
        foreground = (230, 230, 230) if dark else (25, 25, 25)
        draw.rectangle([left, top, left + width, top + height], fill=background)
        draw.rectangle([left, top, left + width, top + 30], fill=(60, 90, 160))
        _draw_text(draw, (left + 10, top + 6), f"Window {panel_index + 1} - {rng.choice(EN_WORDS).title()}",
                   _font(FONT_UI, 14), (255, 255, 255), truths, 14)
        y = top + 44
        while y < top + height - 40:
            size = rng.choice(sizes)
            if rng.random() < 0.35:
                text = rng.choice(JA_PHRASES)
                font = _font(FONT_JA, size)
            else:
                text = " ".join(rng.choice(EN_WORDS) for _ in range(rng.randint(2, 7))).capitalize()
                font = _font(FONT_UI, size)
            _draw_text(draw, (left + 16, y), text, font, foreground, truths, size)
            y += int(size * 1.9)
    return image, truths


def to_captured(image: Image.Image) -> CapturedImage:
    """Pillow 이미지를 앱의 캡처 형식(BGRA)으로 바꾼다."""
    width, height = image.size
    bgra = image.convert("RGBA").tobytes("raw", "BGRA")
    return CapturedImage(ScreenRegion(0, 0, width, height), width, height, bgra)


def score(segments, truths: list[Truth]) -> dict:
    """정답 줄마다 겹치는 인식 줄 중 가장 비슷한 것을 찾아 정확도를 집계한다."""
    ratios: list[float] = []
    small: list[float] = []
    for truth in truths:
        left, top, right, bottom = truth.box
        best = 0.0
        for segment in segments:
            bounds = segment.bounds
            if bounds.right <= left or bounds.left >= right or bounds.bottom <= top or bounds.top >= bottom:
                continue
            ratio = difflib.SequenceMatcher(
                None, "".join(truth.text.split()), "".join(segment.text.split())
            ).ratio()
            best = max(best, ratio)
        ratios.append(best)
        if truth.size_px <= SMALL_TEXT_PX:
            small.append(best)
    return {
        "lines": len(truths),
        "line_recall": round(sum(r >= LINE_MATCH_RATIO for r in ratios) / len(ratios), 3),
        "char_similarity": round(statistics.fmean(ratios), 3),
        "small_line_recall": round(sum(r >= LINE_MATCH_RATIO for r in small) / len(small), 3) if small else None,
        "segments": len(segments),
    }


def build_engine_factory(args: argparse.Namespace):
    """명령줄 설정으로 PaddleOCR 엔진을 만드는 함수를 돌려준다."""

    def factory(settings: PaddleOcrSettings):
        """PaddleOCR 엔진을 만든다 (벤치마크 전용 인자 포함)."""
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        kwargs = dict(
            text_detection_model_name=settings.detection_model,
            text_recognition_model_name=settings.recognition_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=settings.device,
            enable_mkldnn=settings.enable_onednn,
        )
        if args.det_limit_type:
            kwargs["text_det_limit_type"] = args.det_limit_type
        if args.det_limit_side:
            kwargs["text_det_limit_side_len"] = args.det_limit_side
        if args.cpu_threads:
            kwargs["cpu_threads"] = args.cpu_threads
        if args.rec_batch:
            kwargs["text_recognition_batch_size"] = args.rec_batch
        return PaddleOCR(**kwargs)

    return factory


def gpu_memory_mb() -> dict | None:
    """PaddlePaddle이 잡은 GPU 메모리 최댓값 (MB). GPU가 아니면 None."""
    try:
        import paddle

        if not paddle.device.is_compiled_with_cuda():
            return None
        return {
            "max_allocated_mb": round(paddle.device.cuda.max_memory_allocated() / 2**20),
            "max_reserved_mb": round(paddle.device.cuda.max_memory_reserved() / 2**20),
        }
    except Exception as exc:  # 측정 실패는 결과에 남기고 계속한다
        return {"error": repr(exc)}


def measure(recognizer: PaddleTextRecognizer, image: CapturedImage, truths: list[Truth], runs: int) -> dict:
    """한 번 예열한 뒤 `runs`번 재서 지연(ms), CPU 사용(코어 수), 정확도를 돌려준다."""
    recognizer.recognize(image, CancellationToken())
    walls: list[float] = []
    cpu_total = 0.0
    result = None
    for _ in range(runs):
        cpu_before, wall_before = time.process_time(), time.perf_counter()
        result = recognizer.recognize(image, CancellationToken())
        wall = time.perf_counter() - wall_before
        cpu_total += time.process_time() - cpu_before
        walls.append(wall)
    return {
        "median_ms": round(statistics.median(walls) * 1000),
        "min_ms": round(min(walls) * 1000),
        "cpu_cores_used": round(cpu_total / sum(walls), 1),
        **score(result.segments, truths),
    }


def main() -> int:
    """명령줄 설정으로 모든 장면을 재고 JSON을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--onednn", type=int, default=0)
    parser.add_argument("--det-limit-type", default="")
    parser.add_argument("--det-limit-side", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--rec-batch", type=int, default=0)
    parser.add_argument("--det-model", default=PaddleOcrSettings().detection_model)
    parser.add_argument("--rec-model", default=PaddleOcrSettings().recognition_model)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--scenes", default="box_ja,box_en,monitor,desktop")
    parser.add_argument("--label", default="")
    # 0보다 크면 첫 장면을 이 시간(초) 동안 쉬지 않고 반복 인식한다 (게임 영향 측정에서 부하를 거는 용도).
    parser.add_argument("--loop-seconds", type=float, default=0.0)
    args = parser.parse_args()

    settings = PaddleOcrSettings(
        device=args.device,
        detection_model=args.det_model,
        recognition_model=args.rec_model,
        enable_onednn=bool(args.onednn),
    )
    recognizer = PaddleTextRecognizer(settings, engine_factory=build_engine_factory(args))
    desktop_image, desktop_truths = scene_desktop()
    monitor_truths = [t for t in desktop_truths if t.box[2] <= 1920]
    scenes = {
        "box_ja": lambda: scene_box(("おはようございます、勇者さま。", "今日は一緒にお城へ向かいましょう。"), "アリス", FONT_JA),
        "box_en": lambda: scene_box(("The ancient gate opens only when the three moons align.",), "Guard", FONT_UI),
        "monitor": lambda: (desktop_image.crop((0, 0, 1920, 1080)), monitor_truths),
        "desktop": lambda: (desktop_image, desktop_truths),
    }
    import paddle

    results: dict = {
        "label": args.label,
        "paddle": paddle.__version__,
        "config": {k: v for k, v in vars(args).items() if k not in ("scenes", "label")},
    }
    started = time.perf_counter()
    # 모델 로딩 시간: 작은 이미지로 첫 인식을 해서 잰다.
    first_image, _ = scene_box(("init",), "", FONT_UI)
    recognizer.recognize(to_captured(first_image), CancellationToken())
    results["first_call_seconds"] = round(time.perf_counter() - started, 2)
    if args.loop_seconds > 0:
        # 부하 모드: 준비가 끝났음을 알리고 첫 장면을 정해진 시간 동안 반복 인식한다.
        print("READY", flush=True)
        image, _ = scenes[args.scenes.split(",")[0]]()
        captured = to_captured(image)
        deadline = time.perf_counter() + args.loop_seconds
        count = 0
        while time.perf_counter() < deadline:
            recognizer.recognize(captured, CancellationToken())
            count += 1
        results["loop_count"] = count
        print(json.dumps(results, ensure_ascii=False), flush=True)
        return 0
    for name in args.scenes.split(","):
        image, truths = scenes[name]()
        results[name] = measure(recognizer, to_captured(image), truths, args.runs)
    results["gpu_memory"] = gpu_memory_mb()
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
