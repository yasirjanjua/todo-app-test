# PyInstaller spec for 2048 Auto-Solver.
#
# Built natively per OS (PyInstaller cannot cross-compile) -- see
# .github/workflows/build.yml for the CI matrix that invokes this file on each platform:
#   pyinstaller packaging/pyinstaller.spec --noconfirm
#
# Assumptions documented here rather than scattered across the workflow:
#   * macOS builds target universal2 (Apple Silicon + Intel) via PyInstaller's
#     --target-arch universal2, passed on the command line in CI (a .spec file's Analysis()
#     also accepts target_arch, set below from the TARGET_ARCH env var so local builds can
#     override it too).
#   * Unused Qt modules (WebEngine, Multimedia, Bluetooth, ...) are excluded to keep the
#     bundle down, since PySide6 alone is already the single biggest size contributor
#     (~60MB, accepted per the project README).
#   * opencv-python-headless (no Qt/GUI plugins of its own) is used specifically so it
#     doesn't drag in a second, conflicting copy of Qt's platform plugins.

import os
import sys

block_cipher = None

app_name = "2048AutoSolver"
target_arch = os.environ.get("TARGET_ARCH") or None  # e.g. "universal2" on macOS CI

excluded_qt_modules = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtSensors",
    "PySide6.QtPositioning",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtNetworkAuth",
    "PySide6.QtRemoteObjects",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtWebSockets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
]

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 - SPECPATH is injected by PyInstaller

a = Analysis(
    [os.path.join(project_root, "main.py")],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=[
        "core.board",
        "core.heuristics",
        "core.solver",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_qt_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    target_arch=target_arch,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{app_name}.app",
        icon=None,
        bundle_identifier="com.example.twentyfortyeightautosolver",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "NSScreenCaptureUsageDescription": (
                "Lets the app see your game on screen, the same way a screenshot does."
            ),
            "NSAppleEventsUsageDescription": (
                "Lets the app press arrow keys for you while it plays."
            ),
        },
    )
