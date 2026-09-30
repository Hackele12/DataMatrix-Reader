"""create_v24_release.py — Baut DataDetector_v24_Release.zip: Projekt + Python-Laufzeit + fixierte Offline-Wheels."""

import os
import shutil
import subprocess
import time
import zipfile

VERSION = "v24"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASES_DIR = os.path.join(PROJECT_DIR, "releases")
BUILD_DIR = os.path.join(RELEASES_DIR, f"_build_{VERSION}")
WHEEL_DIR = os.path.join(BUILD_DIR, "packages")
CONSTRAINTS_FILE = os.path.join(BUILD_DIR, "constraints.txt")
LOCK_FILE = os.path.join(BUILD_DIR, "requirements_lock.txt")
OUTPUT_ZIP = os.path.join(RELEASES_DIR, f"DataDetector_{VERSION}_Release.zip")
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")
LOCAL_WHEELS = os.path.join(PROJECT_DIR, "packages")

EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".vscode", ".idea",
    "AppDevProjekt_KLIQ", "android", "build", "node_modules", ".gradle",
    "scratch", ".gemini", "releases", "temp_ls_export",
    "packages", "training_data_splits", "benchmarks_split_results",
}
EXCLUDE_EXTENSIONS = {".zip", ".pyc", ".log"}
EXCLUDE_FILES = {"app_debug.log", "vision_network_main.log", "diagnose_output.txt", "test_5_output.txt", "debug_gamma_output.txt", "crash.log"}
STORED_EXTENSIONS = {".whl", ".pt", ".pth", ".onnx", ".jpg", ".jpeg", ".png"}


def find_base_python() -> str:
    """Basis-Interpreter der getesteten .venv (portables CPython) – wird als python_runtime/ mitgeliefert."""
    with open(os.path.join(PROJECT_DIR, ".venv", "pyvenv.cfg"), encoding="utf-8") as f:
        home = next(line.split("=", 1)[1].strip() for line in f if line.lower().startswith("home"))
    base = os.path.realpath(home)
    if not os.path.isfile(os.path.join(base, "python.exe")):
        raise SystemExit(f"[FEHLER] python.exe nicht gefunden in {base}")
    return base


def download_wheels() -> list[str]:
    """Lädt die Abhängigkeiten von requirements.txt exakt in den Versionen der getesteten .venv."""
    shutil.rmtree(WHEEL_DIR, ignore_errors=True)
    os.makedirs(WHEEL_DIR)

    freeze = subprocess.run(
        [VENV_PYTHON, "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True
    ).stdout
    pins = [line for line in freeze.splitlines() if "==" in line and not line.startswith("-e")]
    with open(CONSTRAINTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(pins) + "\n")

    cmd = [
        VENV_PYTHON, "-m", "pip", "download",
        "-r", os.path.join(PROJECT_DIR, "requirements.txt"),
        "-c", CONSTRAINTS_FILE,
        "--only-binary=:all:",
        "-d", WHEEL_DIR,
        "--disable-pip-version-check",
    ]
    if os.path.isdir(LOCAL_WHEELS):
        cmd += ["--find-links", LOCAL_WHEELS]
    subprocess.run(cmd, check=True)

    wheels = sorted(f for f in os.listdir(WHEEL_DIR) if f.endswith(".whl"))
    if not wheels:
        raise SystemExit("[FEHLER] Keine Wheels heruntergeladen.")
    return wheels


def write_lock(wheels: list[str]) -> None:
    # Wheel-Dateiname: {name}-{version}-{python}-{abi}-{plattform}.whl
    lines = []
    for wheel in wheels:
        name, version = wheel.split("-")[:2]
        lines.append(f"{name.replace('_', '-').lower()}=={version}")
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def iter_project_files():
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file in EXCLUDE_FILES or os.path.splitext(file)[1].lower() in EXCLUDE_EXTENSIONS:
                continue
            abs_path = os.path.join(root, file)
            yield abs_path, os.path.relpath(abs_path, PROJECT_DIR)


def iter_runtime_files(base_python: str):
    for root, dirs, files in os.walk(base_python):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for file in files:
            if file.endswith(".pyc"):
                continue
            abs_path = os.path.join(root, file)
            yield abs_path, os.path.join("python_runtime", os.path.relpath(abs_path, base_python))


def add_file(z: zipfile.ZipFile, abs_path: str, arc_path: str, stats: dict) -> None:
    ext = os.path.splitext(abs_path)[1].lower()
    if ext == ".bat":
        # cmd.exe verarbeitet Sprungmarken in Batch-Dateien nur mit CRLF zuverlässig
        with open(abs_path, "rb") as f:
            data = f.read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        z.writestr(zipfile.ZipInfo.from_file(abs_path, arc_path), data, compress_type=zipfile.ZIP_DEFLATED)
    else:
        compress = zipfile.ZIP_STORED if ext in STORED_EXTENSIONS else zipfile.ZIP_DEFLATED
        z.write(abs_path, arc_path, compress_type=compress)
    stats["files"] += 1
    stats["bytes"] += os.path.getsize(abs_path)


def create_release():
    t0 = time.time()
    os.makedirs(BUILD_DIR, exist_ok=True)
    base_python = find_base_python()

    print(f"[1/3] Lade fixierte Wheels nach {WHEEL_DIR} ...")
    wheels = download_wheels()
    write_lock(wheels)
    print(f"      {len(wheels)} Wheels, Lock-Datei: {LOCK_FILE}")

    print(f"[2/3] Schreibe {OUTPUT_ZIP} ...")
    stats = {"files": 0, "bytes": 0}
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for abs_path, arc_path in iter_project_files():
            add_file(z, abs_path, arc_path, stats)
        for wheel in wheels:
            add_file(z, os.path.join(WHEEL_DIR, wheel), os.path.join("packages", wheel), stats)
        add_file(z, LOCK_FILE, "requirements_lock.txt", stats)
        for abs_path, arc_path in iter_runtime_files(base_python):
            add_file(z, abs_path, arc_path, stats)

    print("[3/3] Fertig.")
    print("\n==============================================")
    print(f"      DATA DETECTOR RELEASE {VERSION.upper()} COMPLETE")
    print("==============================================")
    print(f"Archive:            {OUTPUT_ZIP}")
    print(f"Python-Laufzeit:    {base_python}")
    print(f"Total files:        {stats['files']:,}")
    print(f"Uncompressed size:  {stats['bytes'] / (1024 * 1024):.2f} MB")
    print(f"Compressed size:    {os.path.getsize(OUTPUT_ZIP) / (1024 * 1024):.2f} MB")
    print(f"Time taken:         {time.time() - t0:.1f} seconds")
    print("==============================================")


if __name__ == "__main__":
    create_release()
