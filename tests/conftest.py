"""공용 테스트 설정."""

import os

# 실제 화면 없이 Qt 위젯을 테스트한다. QApplication 생성 전에 설정해야 한다.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
