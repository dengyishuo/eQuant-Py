"""Must be imported before any ``efactorcraft``/``ebacktestcraft`` import.

``ebacktestcraft``'s ``signals.py``/``weights.py`` internally do
``from equant.utils.panel import validate_panel``. The ``equant`` umbrella
package (``eQuant-Py/equant/``) was never pip-installed on its own — only
its five sub-packages (eTTR, eClassic, eFactorCraft, eBacktestCraft,
eAlpha101) were editable-installed — so ``import equant`` only resolves
when the repo root happens to be on ``sys.path``. Streamlit's cwd varies
depending on how it's launched, so we pin the repo root here explicitly.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
