# Multimodal Translator

게임 및 영상 콘텐츠를 위한 맥락 인식 번역 기능을 제공하는 Windows 데스크톱 애플리케이션입니다.

현재 단계: **Phase 3 — 콘텐츠 식별** (활성 창으로 콘텐츠를 식별하고, 박스 설정을 콘텐츠별로 SQLite에 저장. 바탕이 되는 Phase 2: PaddleOCR 연결, 실제 화면 테스트, 성능 평가, Qwen 번역 연결 완료)

- OCR: PaddleOCR(PP-OCRv5 mobile)로 실제 글자를 읽습니다. NVIDIA GPU가 있으면 GPU, 없으면 CPU를 씁니다. PaddleOCR가 설치되어 있지 않으면 Mock OCR로 실행됩니다.
- 번역: 로컬 Qwen(기본 Qwen3.5-9B, llama.cpp 서버)으로 영어를 한국어로 번역합니다. Qwen 설정이 없으면 Mock 번역으로 실행됩니다.
  - 번역 품질은 아직 부족한 부분이 있습니다(욕설 강도 순화, 관용구 오역). [docs/phase2-qwen-translation.md](docs/phase2-qwen-translation.md)를 참고하세요.
- Mock OCR은 화면에서 "글자처럼 보이는" 영역을 찾아 위치를 돌려주며, 글자를 읽지는 못하므로 텍스트는 `단어1 단어2`처럼 표시됩니다.

## 기능

- 화면 위에 항상 표시되는 작은 오버레이
- **바운딩 박스 여러 개**: 드래그로 번역 영역을 원하는 만큼 추가하고, 박스마다 따로 이동·크기 조절·삭제
- **원문 위치 기준 번역 표시**: 번역문을 OCR이 인식한 원문 텍스트의 위·아래·왼쪽·오른쪽 반투명 말풍선으로 표시
  - 박스는 박스마다 말풍선 하나에 번역문을 읽는 순서로 모두 넣습니다. 원문이 다른 줄이면 줄을 바꾸고, 같은 줄이면 이어 씁니다.
  - 전체 화면은 원문마다 말풍선을 따로 놓습니다.
  - 기본은 아래이며, 박스마다 방향을 정할 수 있습니다.
  - 화면 밖으로 나가면 위치를 보정합니다.
- **원문 추적**: 원문이 움직이면 말풍선이 따라가고, 사라지면 잠시 뒤 숨깁니다. 한 번 인식되지 않았다고 깜빡이지는 않습니다.
- 박스 안 텍스트 감지
  - 수동: `영역 번역` 버튼, 또는 박스 우클릭 메뉴의 `지금 번역`
  - 자동: 박스별로 켜고 끕니다. 화면이 바뀌면 그 캡처로 OCR하고, 같은 텍스트가 두 번 연속 인식되면(타자 효과가 끝나면) 번역합니다. 이미 번역한 원문은 다시 번역하지 않습니다.
- **전체 화면 번역**: 켜 두는 동안 모든 모니터의 텍스트 옆에 말풍선을 표시하고, 원문이 움직이면 다시 번역하지 않고 위치만 따라갑니다. 박스 안 텍스트는 그 박스가 맡아 중복 번역하지 않습니다.
- **편집 / 잠금 모드**: 편집을 끄면 박스가 흐려지고, 박스와 말풍선이 클릭을 통과시켜 게임 조작을 방해하지 않습니다.
- **콘텐츠 자동 식별**: 지금 보고 있는 프로그램(활성 창의 실행 파일)을 약 0.5초마다 확인하고, 다른 프로그램이 1초 넘게 이어지면 콘텐츠가 바뀐 것으로 봅니다. 바뀌면 진행 중인 번역을 취소하고 말풍선을 지운 뒤 그 콘텐츠의 박스로 바꿉니다. 자세한 내용은 아래 [콘텐츠 식별](#콘텐츠-식별)을 참고하세요.
- **콘텐츠별 박스**: 박스 목록(위치·크기·방향·자동 감지)을 콘텐츠마다 따로 저장해, 같은 프로그램으로 돌아오면 그 박스가 다시 나타납니다.
- **설정 저장**: 오버레이·상태 창 위치, 불투명도는 레지스트리에, 박스 목록은 콘텐츠별로 SQLite에 저장하고 다음 실행에 복원

## 요구 사항

- Windows 10 2004 이상 (오버레이를 캡처에서 제외하는 기능에 필요)
- Python 3.11 이상

## 설치

프로젝트 폴더의 가상환경(`.venv`)을 권장합니다.

```bash
python -m venv .venv
```

NVIDIA GPU가 있으면 GPU 빌드를 설치합니다(권장). PyPI에 없으므로 Paddle 공식 인덱스(CUDA 12.9)를 함께 지정합니다.

```bash
.venv\Scripts\python -m pip install -e .[dev,ocr-gpu] --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

GPU가 없으면 CPU 빌드를 설치합니다.

```bash
.venv\Scripts\python -m pip install -e .[dev,ocr]
```

- `ocr`·`ocr-gpu`를 둘 다 빼면 PaddleOCR 없이 Mock OCR로 실행됩니다. 둘은 같은 `paddle` 모듈을 제공하므로 함께 설치하지 않습니다.
- PaddlePaddle은 **3.2.x**로 고정합니다. 3.3.x는 CPU 가속(oneDNN)을 켜면 오류가 납니다.
- 2026-10-12에 다음 조합으로 동작을 확인했습니다: `paddleocr 3.7.0`, `paddlepaddle-gpu 3.2.2`(cu129, RTX 5060 Ti), `paddlepaddle 3.2.2`(CPU).
- GPU 빌드는 약 0.7GB이고 CUDA 런타임 패키지까지 합치면 가상환경이 약 5GB가 됩니다.
  - 공식 인덱스(`paddlepaddle.org.cn`)에서 내려받기가 멈추면, 같은 파일을 공식 CDN에서 직접 받아 설치할 수 있습니다: `https://paddle-whl.cdn.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.2.2-cp311-cp311-win_amd64.whl`
- PyPI의 `paddlepaddle-gpu`(2.6.x)는 RTX 50 시리즈를 지원하지 않습니다.
- **처음 인식할 때** PP-OCRv5 mobile 모델(검출·인식 각 수 MB~수십 MB)을 내려받아 `%USERPROFILE%\.paddlex\official_models`에 저장합니다. 이때만 네트워크를 쓰며, 화면 내용은 전송하지 않습니다.

### Qwen 번역 설정

번역은 [llama.cpp](https://github.com/ggml-org/llama.cpp) 서버를 앱이 직접 띄워서 씁니다. 다음 두 가지를 기본 폴더 `%LOCALAPPDATA%\MultimodalTranslator\llm`에 둡니다.

| 무엇 | 위치 | 받는 곳 (2026-10-12 확인) |
| --- | --- | --- |
| llama.cpp CUDA 빌드 | `llm\llama.cpp\<빌드 이름>\llama-server.exe` (압축 두 개를 같은 폴더에 풉니다) | GitHub 릴리스 `b11557`: `llama-b11557-bin-win-cuda-13.4-x64.zip`, `cudart-llama-bin-win-cuda-13.4-x64.zip` |
| 번역 모델 (GGUF) | `llm\models\Qwen3.5-9B-Q4_K_M.gguf` | Hugging Face `unsloth/Qwen3.5-9B-GGUF` (5.4GB) |

- 내려받은 파일은 공개된 sha256 값과 비교해 확인합니다(GitHub 릴리스 정보, Hugging Face 파일 정보).
- 빌드 폴더가 여러 개면 가장 최근에 바뀐 빌드를 씁니다.
- 앱을 켜면 서버가 바로 모델을 불러옵니다(약 4초, VRAM 약 5.3GB). 서버는 이 컴퓨터 안(127.0.0.1)에서만 접근할 수 있고, 앱이 끝나면 함께 끝납니다.

## 실행

```bash
python -m translator.main
```

설치 후에는 `multimodal-translator` 명령으로도 실행할 수 있습니다.

### OCR·번역 엔진 선택

환경 변수로 OCR 엔진과 번역 엔진을 고릅니다.

| 환경 변수 | 값 | 동작 |
| --- | --- | --- |
| `TRANSLATOR_OCR_ENGINE` | `auto` (기본) | PaddleOCR가 설치되어 있으면 PaddleOCR, 없으면 경고 로그를 남기고 Mock OCR |
| | `paddle` | 항상 PaddleOCR. 설치되어 있지 않으면 인식할 때마다 상태 창에 설치 안내를 보여 줍니다. |
| | `mock` | Mock OCR (모델 없이 UI 확인) |
| `TRANSLATOR_TRANSLATION_ENGINE` | `auto` (기본) | Qwen 설정(아래 주소 또는 llama.cpp·모델 파일)이 있으면 Qwen, 없으면 경고 로그를 남기고 Mock 번역 |
| | `qwen` | 항상 Qwen. 설정이 없으면 번역할 때마다 상태 창에 안내를 보여 줍니다. |
| | `mock` | Mock 번역 (원문에 표식만 붙임) |
| `TRANSLATOR_LLM_MODEL`, `TRANSLATOR_LLAMA_SERVER` | 파일 경로 | 기본 폴더 대신 쓸 GGUF 모델과 llama-server 실행 파일 |
| `TRANSLATOR_LLM_URL` | 예: `http://127.0.0.1:8080` | 이미 떠 있는 OpenAI 호환 서버를 씁니다(앱이 서버를 띄우지 않음). 이 컴퓨터 밖의 주소는 원문이 외부로 전송되므로 `TRANSLATOR_LLM_ALLOW_REMOTE=1`일 때만 씁니다. |
| `TRANSLATOR_OCR_DEVICE` | `auto` (기본) | GPU 빌드가 설치되어 있고 NVIDIA GPU가 있으면 `gpu:0`, 아니면 `cpu`. GPU에서 모델을 불러오지 못하면 CPU로 한 번 다시 시도합니다. |
| | `cpu`, `gpu:0` 등 | 그 장치만 씁니다. |

- 모델은 프로그램 시작이 아니라 **처음 인식할 때** 백그라운드에서 불러옵니다. 그래서 첫 번역은 몇 초 더 걸립니다.
- 모델을 불러오지 못하면(설치 안 됨, 다운로드 실패) 상태 창에 해결 방법을 보여 주고, 프로그램을 다시 시작할 때까지 다시 시도하지 않습니다.
- **속도** (2026-10-12, 실제 화면 측정): GPU에서는 박스 OCR 약 30~50ms, 모니터 두 대 전체 화면 약 1.5초입니다.
  CPU에서는 박스 약 0.1~0.2초, 전체 화면 약 6.5초이고 코어 4개를 씁니다.
- **CPU일 때 전체 화면**: 확인 간격을 늘리고(`FULLSCREEN_DETECT_INTERVAL_CPU_MS`), 켤 때 상태 창에 느리다는 안내를 보여 줍니다. 바운딩 박스 사용을 권장합니다.

### 사용 방법

화면 위쪽 가운데에 오버레이가 나타납니다. 빈 공간(`⋮⋮`)을 드래그하면 옮길 수 있습니다.

| 버튼 | 동작 |
| --- | --- |
| **영역 추가** | 화면이 어두워지면 번역할 부분을 드래그합니다(Esc나 우클릭으로 취소). 박스가 하나 추가되고 바로 번역됩니다. 편집 중에 추가하면 편집을 끌 때 번역됩니다. |
| **전체 화면** | 켜고 끄는 버튼입니다. 켜면 모든 모니터의 텍스트를 번역하고, 화면이 바뀌면 원문 위치와 번역을 갱신합니다. 끄면 전체 화면 번역을 지웁니다. 박스 안 텍스트는 각 박스가 맡습니다. |
| **영역 번역** | 모든 박스를 지금 번역합니다. |
| **자동** | 모든 박스의 자동 감지를 한 번에 켜거나 끕니다. 모든 박스가 자동 감지 중이면 켜진 상태로 표시됩니다. |
| **편집** | 켜면 박스를 수정할 수 있고, 끄면 박스가 흐려지며 클릭이 통과됩니다(잠금). 번역을 실행하면 자동으로 꺼집니다. |
| **✕** | 프로그램을 종료합니다. |

- **바운딩 박스** (편집 모드에서 조작)
  - 안쪽을 드래그하면 이동하고, 가장자리나 모서리를 드래그하면 크기가 바뀝니다.
  - 왼쪽 위 이름표에 박스 ID, 출력 방향, 자동 감지 여부가 보입니다(예: `box-1 · ↓ 아래 · 자동`).
  - 우클릭 메뉴에서 **번역 출력 방향**(위·아래·왼쪽·오른쪽), **자동 감지**, **지금 번역**, **영역 삭제**를 고릅니다. 오른쪽 위 `✕`로도 삭제할 수 있습니다.
  - 박스마다 색이 다르며, 그 박스의 말풍선 왼쪽 띠도 같은 색입니다.
  - 테두리 바깥 여백은 캡처하지 않고 테두리 안쪽만 캡처합니다.
- **상태·설정 창**
  - 잠금 모드에서는 오류나 "인식된 텍스트가 없습니다" 알림을 잠깐 보여 주고 스스로 사라집니다.
  - 편집 모드에서는 설정 창으로 나타납니다. 드래그로 옮기고, 오른쪽 아래 모서리로 크기를 조절합니다.
  - 설정 항목은 **말풍선 배경 불투명도**(20~100%, 이 창 배경에도 함께 적용)와 **전체 화면 번역 위치**(전체 화면 번역의 말풍선을 원문의 위·아래·왼쪽·오른쪽 중 어디에 띄울지)입니다.

오버레이, 박스, 말풍선, 상태 창은 캡처 이미지에 찍히지 않도록 설정되어 있어 번역 대상에 섞이지 않습니다.

오버레이·상태 창 위치와 불투명도는 Windows 사용자 레지스트리(`HKCU\Software\MultimodalTranslator`)에 저장되며, 오버레이 `✕`로 정상 종료할 때 저장합니다.
- 저장된 위치가 지금 모니터 구성에서 화면 밖이면 기본 위치를 씁니다.

박스 목록은 콘텐츠별로 SQLite 파일 `%LOCALAPPDATA%\MultimodalTranslator	ranslator.db`에 저장됩니다.
- 박스를 옮기거나 바꾸면 0.3초 뒤에 마지막 상태를 저장하고, 콘텐츠를 바꿀 때와 종료할 때는 바로 저장합니다.
- 자동 감지가 켜져 있던 박스는 재실행하면 바로 감지를 시작합니다.
- 지금 모니터 구성에서 화면 밖인 박스는 화면에 만들지 않지만 설정은 지우지 않고 보존합니다. 모니터를 다시 연결하면 복원됩니다.
- 이전 버전이 레지스트리에 저장한 박스는 첫 실행에 자동으로 옮겨집니다. 박스가 없는 콘텐츠가 처음 식별되면 그 콘텐츠가 가져갑니다.
- SQLite 파일을 열 수 없으면(손상, 권한) 상태 창에 파일 경로와 함께 안내하고, 박스를 레지스트리에 계속 저장합니다. 파일 문제를 해결하고 다시 시작하면 SQLite로 옮겨집니다. 더 새로운 버전이 만든 파일은 건드리지 않습니다.
- 저장되는 것은 박스 설정(위치·크기·방향·자동 감지)과 콘텐츠 ID(실행 파일 이름)뿐입니다. 화면, 번역문, 창 제목은 저장하지 않습니다. 박스가 하나도 없는 콘텐츠는 기록이 남지 않습니다.
- 저장된 박스를 모두 지우려면 앱을 끈 뒤 `translator.db`(와 같은 폴더의 `-wal`, `-shm` 파일)를 삭제합니다. 같은 콘텐츠를 계속 저장해도 행이 늘지 않고(저장은 그 콘텐츠의 행을 바꿔 쓰는 방식), 임시 DB로 2만 번 저장했을 때 파일은 약 4MB(대부분 로그 파일 `-wal`)에서 더 늘지 않았습니다.

### 콘텐츠 식별

- **무엇을 보나**: 활성 창을 가진 프로세스의 실행 파일 이름으로 콘텐츠 ID를 만듭니다(예: `app:chrome.exe`). **창 제목과 주소는 읽지도 저장하지도 않습니다.** 그래서 브라우저는 프로그램 하나가 콘텐츠 하나입니다. 탭·영상 단위 식별은 [검토 문서](docs/phase3-youtube-identification-review.md)에 정리했고 아직 구현하지 않았습니다.
- **언제 바뀌나**: 새 프로그램이 약 1초(0.5초 간격 3회) 이어지면 바뀝니다. Alt-Tab 중간 화면처럼 잠깐 스쳐 가는 창은 무시합니다. 값은 `tuning.py`의 `CONTENT_*`이며 초기 추정값입니다.
- **무시하는 창**: 이 앱 자신의 창(오버레이·말풍선·박스), 작업 표시줄·바탕 화면·Alt-Tab(`explorer.exe`), 시작 메뉴·검색·알림 팝업, 입력기, 잠금 화면, Xbox Game Bar·Steam 오버레이, UWP 앱 틀 창(`ApplicationFrameHost.exe`)입니다. 이런 창이 앞에 있어도 현재 콘텐츠를 유지합니다. 목록은 `adapters/content/windows.py`의 `_IGNORED_PROCESS_NAMES`이고, 실제 게임에서 엉뚱하게 바뀌면 여기에 추가합니다(이름은 알려진 이름이며 모두 확인한 것은 아닙니다).
- **한계**: 폴더가 달라도 실행 파일 이름이 같으면 같은 콘텐츠입니다. UWP 앱과 관리자 권한으로 실행 중인 프로그램은 식별하지 못할 수 있습니다(식별 실패 시 현재 콘텐츠 유지). 파일 탐색기 창은 콘텐츠가 될 수 없습니다. 활성 창 확인이 예외로 5번 연속 실패하면 자동 식별을 끄고 상태 창에 알리며, 앱을 다시 시작해야 복구됩니다. 전체 화면 독점 모드 게임에서의 활성 창 판정은 확인하지 못했습니다.

### 문서

| 문서 | 내용 |
| --- | --- |
| [docs/provisional-decisions.md](docs/provisional-decisions.md) | 임시 조정값과 동작 결정 (숫자 값은 `src/translator/tuning.py`) |
| [docs/phase2-detection-design.md](docs/phase2-detection-design.md) | 자동 감지 분석, OCR 기반 감지 전환 시 의존성, 타자 효과 안정화 설계 |
| [docs/dpi-verification.md](docs/dpi-verification.md) | 모니터별 DPI·좌표 변환 검증 현황 (실측 / 시뮬레이션 / 미검증 구분) |
| [docs/phase2-real-screen-test.md](docs/phase2-real-screen-test.md) | PaddleOCR 실제 화면 테스트 결과, 측정한 지연·부하, 발견한 문제 |
| [docs/phase2-qwen-translation.md](docs/phase2-qwen-translation.md) | Qwen 번역 연결: 실행 방식, 모델 비교 실험, 출력 검증 장치, 실제 화면 결과, 한계 |
| [docs/phase2-performance-evaluation.md](docs/phase2-performance-evaluation.md) | PaddleOCR 설정별(CPU·oneDNN·GPU·입력 축소·서버 모델) 지연·정확도·자원, 게임 영향 대리 측정, 권장 사항 |
| [docs/phase3-youtube-identification-review.md](docs/phase3-youtube-identification-review.md) | YouTube 영상·브라우저 탭 식별 후보 비교(미디어 세션, 확장 + 네이티브 메시징 등), 약관·개인정보 검토, 권고안 (구현 전 검토 문서) |

### 로그

로그는 콘솔과 `%LOCALAPPDATA%\MultimodalTranslator\logs\translator.log`에 기록됩니다.
화면 내용이나 번역 텍스트는 기록하지 않으며, 캡처 이미지는 디스크에 저장하지 않습니다.

## 테스트

```bash
pytest
```

테스트는 Qt `offscreen` 플랫폼과 가짜 캡처 객체를 사용하므로 실제 화면이나 GPU 없이 실행됩니다.
PaddleOCR 어댑터 테스트도 가짜 엔진을 쓰므로 모델 없이 실행됩니다(numpy 필요).

실제 PaddleOCR 모델로 합성 이미지(영어·일본어 두 줄)를 인식하는 테스트는 기본으로 건너뜁니다.

```bash
TRANSLATOR_PADDLE_OCR_TESTS=1 pytest tests/test_paddle_ocr_real.py
```

실제 화면에 가짜 대사 창을 띄우고 PaddleOCR로 앱 흐름(박스 번역, 타자 효과, 대사 교체, 원문 이동, 전체 화면, 종료)을 확인하는 테스트입니다. CPU 전체 화면 OCR 때문에 1분 넘게 걸립니다.

```bash
TRANSLATOR_REAL_SCREEN_TESTS=1 TRANSLATOR_PADDLE_OCR_TESTS=1 pytest tests/test_paddle_real_screen.py -s
```

이 검사는 앱과 같이 번역 엔진을 고릅니다(Qwen 설정이 있으면 Qwen). 실제 Qwen 서버로 번역 형식과 종료만 확인하는 테스트는 다음과 같습니다.

```bash
TRANSLATOR_QWEN_TESTS=1 pytest tests/test_qwen_real.py
```

실제 Windows 화면에 창을 띄워 좌표 변환·캡처·말풍선 배치를 확인하는 테스트는 기본으로 건너뜁니다. 직접 실행하려면 다음 명령을 씁니다.

```bash
TRANSLATOR_REAL_SCREEN_TESTS=1 pytest tests/test_dpi_real_screen.py
```

| 파일 | 내용 |
| --- | --- |
| `tests/test_capture.py` | MSS 캡처 어댑터, 오류 처리, 일시 실패 재시도(횟수 제한) |
| `tests/test_mock_layout_ocr.py` | 위치를 돌려주는 Mock OCR (단어 찾기, 이동, 잡음 무시) |
| `tests/test_paddle_ocr.py` | PaddleOCR 어댑터 (가짜 엔진): 입력 변환, 화면 좌표, 출력 검증, 설치·로딩·추론 오류, 취소, 동시 호출, 파이프라인 연결 |
| `tests/test_paddle_ocr_real.py` | 실제 PaddleOCR 모델 인식 (선택 실행): 영어·일본어 줄, 다른 스레드에서 추론, 대사 묶기 |
| `tests/test_ocr_factory.py` | OCR 엔진 선택 (환경 변수, 설치되지 않았을 때의 처리) |
| `tests/test_chat_completion.py` | Qwen 번역 어댑터 (가짜 HTTP): 묶음 요청·스키마, 출력 검증, 줄 단위 재시도, 한국어 원문 건너뛰기, 서버 대기·오류·취소, 로그에 원문 없음 |
| `tests/test_llama_server.py` | llama.cpp 서버 프로세스: 실행 인자(127.0.0.1), 시작·종료·강제 종료, Job Object |
| `tests/test_translation_factory.py` | 번역 엔진 선택 (기본 폴더, 환경 변수, 외부 주소 제한) |
| `tests/test_qwen_real.py` | 실제 Qwen 서버 번역 (선택 실행) |
| `tests/test_paddle_real_screen.py` | 실제 화면 + PaddleOCR 앱 흐름 (선택 실행). 결과는 [docs/phase2-real-screen-test.md](docs/phase2-real-screen-test.md) |
| `tests/test_text_grouping.py` | OCR 조각을 줄·대사로 묶기, 관계없는 텍스트 분리, 띄어쓰기 |
| `tests/test_pipeline.py` | 캡처 → OCR → 묶기 → 번역, 프레임 재사용(다시 캡처하지 않음), 텍스트 자리 지문, 제외 영역, 캐시, 출력 검증, 취소 |
| `tests/test_ocr_service.py` | 같은 캡처의 OCR 재사용·중복 합치기, 바뀐 화면 재인식, 취소·오류 전달 |
| `tests/test_frames.py` | 프레임 번호(스레드 간 유일), 캡처 내용 식별값 |
| `tests/test_controller.py` | 공용 추론 큐(Mock 작업): 우선순위(번역 우선), 동시 실행 수 제한, 출처별 취소, 오래된 결과 무시, 종료, UI 응답성 |
| `tests/test_text_display.py` | 텍스트 고유 ID, 이동·내용 변경 추적, 안정화 후 번역, 중복 번역 방지, OCR 누락과 실제 사라짐 구분, 오래된 프레임 무시 |
| `tests/test_text_stabilizer.py` | 타자 효과 안정화, 최대 대기 시간, 커서 문자 무시 |
| `tests/test_recheck.py` | 후속 OCR·재검사 예약, 간격 늘리기, 최대 횟수 제한 |
| `tests/test_change_detection.py` | 화면 지문(영역 지문 포함), 변화 판단 정책, 움직이는 배경에서도 OCR 요청, 큰 박스 안 작은 글자 변화 |
| `tests/test_auto_detector.py` | 자동 감지 주기 실행, 캡처를 담은 프레임 전달, 중지, 연속 실패 시 자동 중지, 수동 OCR 화면 기록, 감지 방식 교체 |
| `tests/test_bubble_layout.py` | 말풍선 상하좌우 배치, 화면 경계 보정, 너비 제한 |
| `tests/test_translation_overlay.py` | 말풍선 생성·이동·정리, 박스별 방향, 박스 하나 = 말풍선 하나, 긴 문장 줄바꿈, 클릭 통과 |
| `tests/test_bubble_text.py` | 박스의 번역문을 읽는 순서로 합치기 (다른 줄은 줄바꿈, 같은 줄은 이어 쓰기) |
| `tests/test_region_box.py` | 박스 캡처 영역, 이동/크기 조절, 우클릭 메뉴, 이름표, 잠금 시 클릭 통과 |
| `tests/test_region_manager.py` | 여러 박스의 ID·색, 독립 이동, 박스별 방향·자동 감지, 삭제, 복원 |
| `tests/test_region_selector.py` | 드래그 선택, 선택 결과 좌표 |
| `tests/test_screen_geometry.py` | 모니터별 DPI 배율, 물리↔논리 변환, 반올림, 화면 밖 판정 (가짜 모니터로 계산식 검증) |
| `tests/test_dpi_real_screen.py` | 실제 화면 캡처와 말풍선 배치 (선택 실행, Qt 배율 시뮬레이션) |
| `tests/test_ui_settings.py` | 설정 저장/복원, 레지스트리 이전 형식 박스, SQLite에 저장한 박스 복원, 손상된 값 무시, 화면 밖 위치 처리 |
| `tests/test_content_detector.py` | 콘텐츠 정보 검증, 실행 파일 → 콘텐츠 ID, 활성 창 식별기(자기 앱 창 무시, 오류 처리), Mock 식별기, 실제 Windows API 호출 |
| `tests/test_content_ignored_processes.py` | 시작 메뉴·셸·게임 오버레이 등 시스템·보조 창은 콘텐츠로 보지 않음 |
| `tests/test_content_switch.py` | 콘텐츠 바뀜 판단: 연속 관찰 확인, 잠깐 다른 창을 본 경우 무시, 식별 실패 처리 |
| `tests/test_content_monitor.py` | 콘텐츠 감시기(실제 타이머·워커 스레드): 확정 알림, 연속 실패 시 중지, 멈춘 뒤 늦은 결과 무시 |
| `tests/test_sqlite_profiles.py` | SQLite 저장소: 저장·복원, 콘텐츠 분리, 옮기기, 스키마 버전, 손상·닫힘, 트랜잭션 롤백(커밋 실패 포함), 잠금 대기 제한 |
| `tests/test_region_profiles.py` | 콘텐츠 전환 시 박스 교체, 콘텐츠 간 섞임 방지, 처음 식별되는 콘텐츠의 기존 박스 인수, 저장 모으기, 저장소 실패 알림, 화면 밖 박스 보존 |
| `tests/test_region_migration.py` | 레지스트리의 이전 형식 박스를 SQLite로 이전 (한 번만, 덮어쓰지 않음, 실패 시 재시도) |
| `tests/test_profile_store_fallback.py` | SQLite를 열 수 없을 때 레지스트리 대체 저장소: 기존 박스 유지, 콘텐츠별 저장, 안내 문구, 다시 SQLite로 이전 |
| `tests/test_content_profiles_app.py` | 콘텐츠별 박스를 앱 전체(coordinator)로 확인: 전환·복귀, 종료 후 복원, 이전 형식 박스 이전, 저장 실패 시 동작 |
| `tests/test_overlay.py` | 오버레이 버튼, 상태 창 알림·설정·불투명도 |
| `tests/test_coordinator.py` | 여러 박스, 편집/잠금, 원문 기준 말풍선, 전체 화면 추적, 자동 감지, 안정화·재검사, 동시 요청, 종료, 콘텐츠 전환을 화면과 연결한 전체 흐름 |
| `tests/test_workflow.py` | 번역 흐름 단위 테스트(가짜 컨트롤러·박스·감지기, 위젯·스레드 없음): 수동·자동 요청, 안정화 뒤 번역, 중복 요청 방지, 전체 화면, 재검사, 종료, 세션 전환, 콘텐츠 감시기 연결, 전환 시 박스 교체 순서 |
| `tests/test_session.py` | 콘텐츠·세션 식별 정보 (빈 ID 거부, 새 세션 생성) |
| `tests/test_architecture.py` | 계층 규칙: core·application이 ui·Qt 위젯을 가져오지 않음 |

### 성능 측정

pytest와 따로 실행하는 측정 스크립트입니다. PySide6 없이 돌아가므로 다른 PaddlePaddle 버전의 가상환경에서도 실행할 수 있습니다.

```bash
python tests/benchmark/ocr_benchmark.py --device cpu --runs 3
```

```bash
python tests/benchmark/contention_benchmark.py --ocr-python .venv/Scripts/python.exe -- --scenes desktop
```

```bash
python tests/benchmark/translation_benchmark.py --model Qwen3.5-9B-Q4_K_M.gguf
```

## 폴더 구조

```
src/translator/
  main.py         진입점 및 의존성 조립
  tuning.py       임의로 정한 조정값 모음
  core/           공유 데이터 모델(좌표가 있는 OCR 조각, 번역 묶음 등), 인터페이스, 예외, 취소 토큰,
                  콘텐츠·세션 식별 정보(session), 식별한 콘텐츠(content)
  application/    번역 흐름(workflow: 출처별 요청·안정화·재검사·전체 화면 추적·세션 전환)과 그 인터페이스(workflow_ports),
                  인식·번역 작업(pipeline), 프레임(frames), OCR 공유 캐시(ocr_service),
                  공용 추론 큐(scheduler), 출처별 요청 관리(controller), 텍스트 묶기(text_grouping),
                  번역 텍스트 추적(text_display), 말풍선 텍스트 합치기(bubble_text), 텍스트 안정화(text_stabilizer),
                  후속 OCR·재검사 예약(recheck), 박스 설정(regions),
                  화면 변화 판단(change_detection), 자동 감지(auto_detector),
                  콘텐츠 바뀜 판단(content_switch)과 활성 창 감시(content_monitor),
                  콘텐츠별 박스 저장소 인터페이스(profile_store)
  adapters/
    capture/      MSS 화면 캡처
    content/      콘텐츠 식별 구현 (Windows 활성 창의 프로세스, Mock)
    storage/      콘텐츠별 박스 저장 (SQLite)
    ocr/          OCR 구현 (PaddleOCR, Mock), 엔진 선택(factory)
    translation/  번역 엔진 구현 (OpenAI 호환 채팅 API(Qwen), llama.cpp 서버 관리, Mock), 엔진 선택(factory)
  ui/             오버레이, 영역 선택, 바운딩 박스(region_box)와 관리(region_manager),
                  번역 말풍선(translation_bubble, translation_overlay, bubble_layout),
                  상태·설정 창(result_window), 창 연결(coordinator: 버튼·창·설정만 맡음),
                  박스 관리자를 번역 흐름에 연결하는 어댑터(region_box_port),
                  좌표 변환(screen_geometry), UI 설정 저장(ui_settings: 레지스트리, SQLite 대체 저장소 포함),
                  이전 형식 박스 이전(region_migration)
  utils/          로깅
docs/             임의 결정 사항, 설계, 검증 현황 문서
tests/            pytest 테스트
```

## 확장 방법

- **OCR 엔진 교체**: `core/interfaces.py`의 `TextRecognizer`를 구현한 클래스를 `adapters/ocr/`에 추가하고
  `main.py`의 `build_coordinator`에 전달합니다.
  - 인식한 텍스트 조각마다 `TextSegment`(텍스트, 화면 물리 좌표, 단위)를 돌려주면, 묶기·추적·말풍선 배치는 그대로 동작합니다.
  - 이미지 안 좌표는 `CapturedImage.to_screen_region`으로 화면 좌표로 바꿉니다.
- **콘텐츠 식별 방식 추가**: `core/interfaces.py`의 `ContentDetector`를 구현한 클래스를 `adapters/content/`에 추가하고
  `main.py`의 `build_coordinator(content_detector=...)`에 전달합니다. 식별하지 못하면 `None`을 돌려주면 됩니다(예외 대신).
  `ContentIdentity`에는 식별 근거(`IdentificationMethod`)를 담습니다.
- **콘텐츠(세션) 전환**: 활성 창 감시기가 콘텐츠가 바뀌었다고 판단하면 `workflow.change_content(콘텐츠 ID)`를 부릅니다.
  이전 세션의 요청은 모두 취소되고 늦게 도착하는 결과는 버려지며, 말풍선을 지우고 박스를 새 콘텐츠의 것으로 바꾼 뒤
  자동 감지 박스와 전체 화면은 현재 화면을 새 세션에서 다시 인식합니다. 다른 경로(수동 선택 등)에서도 같은 함수를 부르면 됩니다.
- **번역 엔진 교체**: `Translator`를 구현합니다. 여러 줄을 한 번에 번역할 수 있으면 `translate_batch`를 재정의합니다.
- **Context Memory**: `TranslationRequest.context`에 관련 문맥을 채워 번역 엔진에 전달하도록
  `application/pipeline.py`에 연결합니다. 이때 번역 캐시 키에도 문맥을 넣어야 합니다.
- **콘텐츠별 저장 확장**: 박스 목록은 `application/profile_store.py`의 `ContentProfileStore`를 통해 콘텐츠 ID별로 저장됩니다.
  Phase 4의 문맥 기억 테이블은 `adapters/storage/sqlite_profiles.py`의 `_MIGRATIONS`에 새 항목(스키마 버전 +1)을 추가하는 방식으로
  같은 SQLite 파일에 더합니다. 이미 배포한 항목은 고치지 않습니다.
