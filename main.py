"""Startet die Brötchen-Prognose:  python main.py [daten.csv]"""

import sys

from broetchen.app import run

if __name__ == "__main__":
    sys.exit(run())
