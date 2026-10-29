# -*- mode: python ; coding: utf-8 -*-
# Build:  uv run pyinstaller BroetchenApp.spec   ->  dist/BroetchenApp.exe
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ["main.py"],
    datas=[
        ("broetchen/resources", "broetchen/resources"),
        ("data/broetchen_bedarf.csv", "data"),
        *collect_data_files("holidays"),  # Übersetzungen der Feiertagsnamen
    ],
    hiddenimports=["matplotlib.backends.backend_qtagg", *collect_submodules("holidays.countries")],
    excludes=["tkinter", "IPython", "jupyter", "notebook", "matplotlib.tests", "pytest"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BroetchenApp",
    console=False,
    upx=False,
    version="version_info.txt",
    icon=["broetchen/resources/app.ico"],
)
