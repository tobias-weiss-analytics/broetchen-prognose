import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broetchen.data import load_sales  # noqa: E402
from broetchen.model import train  # noqa: E402


@pytest.fixture(scope="session")
def sales():
    return load_sales()


@pytest.fixture(scope="session")
def trained(sales):
    return train(sales, "ridge")
