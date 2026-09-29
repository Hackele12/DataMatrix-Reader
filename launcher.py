"""
launcher.py — Startfenster von DataDetector: öffnet DataMatrixReader, Benchmark und Log Analyzer.

Jedes Programm läuft als eigener Prozess weiter, auch wenn das Startfenster geschlossen wird.
Nur der DataMatrixReader bekommt ein Konsolenfenster; darin läuft `launcher.py --watchdog vision_app.py`,
das ihn nach einem Absturz neu startet. Benchmark und Log Analyzer laufen ohne Konsole
(Ausgaben in benchmark_gui.log bzw. log_analyzer_app.log).
"""

import ctypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from tkinter import messagebox

import customtkinter as ctk

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(os.path.dirname(sys.executable), "python.exe")
RESTART_DELAY_S = 3

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

ACCENT = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
SUCCESS = "#16A34A"
BG_MAIN = "#E2E8F0"
BG_CARD = "#FFFFFF"
BORDER = "#CBD5E1"
TXT_MID = "#475569"
TXT_LIGHT = "#94A3B8"


@dataclass(frozen=True)
class Program:
    title: str
    description: str
    script: str
    console: bool  # eigenes Konsolenfenster mit Watchdog (Neustart nach Absturz)

    @property
    def log_file(self) -> str:
        return os.path.join(APP_DIR, os.path.splitext(self.script)[0] + ".log")


PROGRAMS = (
    Program("DataMatrixReader", "Kameras, Live-Scan und TCP-Port pro Kamera (9500, 9501, …)", "vision_app.py", True),
    Program("Benchmark", "Erkennungsrate auf Testbildern messen, Ground Truth pflegen", "benchmark_gui.py", False),
    Program("Log Analyzer", "Scan-Protokolle und Statistiken auswerten", "log_analyzer_app.py", False),
)


def run_watchdog(script: str) -> int:
    """Konsolen-Modus: startet `script` und nach einem Absturz neu; normales Schließen (Exit-Code 0) beendet."""
    program = next(p for p in PROGRAMS if p.console and p.script == script)
    ctypes.windll.kernel32.SetConsoleTitleW(f"{program.title} - Konsole")
    while True:
        print("=" * 60)
        print(f" {program.title} - {time.strftime('%d.%m.%Y %H:%M:%S')}")
        print(f" Dieses Fenster schliessen = {program.title} beenden")
        print("=" * 60, flush=True)
        try:
            exit_code = subprocess.call([sys.executable, script], cwd=APP_DIR)
            if exit_code == 0:
                return 0
            print(f"\n[WATCHDOG] {program.title} wurde unerwartet beendet (Exit Code {exit_code}). "
                  f"Neustart in {RESTART_DELAY_S} Sekunden ...\n", flush=True)
            time.sleep(RESTART_DELAY_S)
        except KeyboardInterrupt:
            return 0


def launch(program: Program) -> subprocess.Popen:
    if program.console:
        return subprocess.Popen(
            [PYTHON_EXE, os.path.abspath(__file__), "--watchdog", program.script],
            cwd=APP_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    with open(program.log_file, "w", encoding="utf-8") as log:
        return subprocess.Popen(
            [PYTHON_EXE, program.script], cwd=APP_DIR,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),  # Umlaute/Emojis in der Log-Datei statt Encoding-Fehler
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
        )


def _tail(path: str, lines: int = 15) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:]).strip()
    except OSError:
        return ""


class LauncherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DataDetector")
        self.geometry("540x470")
        self.resizable(False, False)
        self.configure(fg_color=BG_MAIN)
        self._processes: dict[str, subprocess.Popen] = {}
        self._widgets: dict[str, tuple[ctk.CTkButton, ctk.CTkLabel]] = {}

        ctk.CTkLabel(
            self, text="DataDetector",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color=ACCENT
        ).pack(anchor="w", padx=28, pady=(22, 0))
        ctk.CTkLabel(
            self, text="Programm auswählen", font=ctk.CTkFont(size=12), text_color=TXT_MID
        ).pack(anchor="w", padx=28, pady=(0, 10))

        for program in PROGRAMS:
            card = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER)
            card.pack(fill="x", padx=24, pady=6)
            card.grid_columnconfigure(0, weight=1)

            button = ctk.CTkButton(
                card, text=program.title, height=44,
                fg_color=ACCENT, hover_color=ACCENT_HOVER,
                font=ctk.CTkFont(size=15, weight="bold"),
                command=lambda p=program: self._open(p)
            )
            button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 6))
            ctk.CTkLabel(
                card, text=program.description, font=ctk.CTkFont(size=11), text_color=TXT_MID, anchor="w"
            ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 10))
            status = ctk.CTkLabel(
                card, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color=SUCCESS, anchor="e"
            )
            status.grid(row=1, column=1, sticky="e", padx=16, pady=(0, 10))
            self._widgets[program.script] = (button, status)

        ctk.CTkLabel(
            self, text="Geöffnete Programme laufen weiter, auch wenn dieses Fenster geschlossen wird.",
            font=ctk.CTkFont(size=10), text_color=TXT_LIGHT
        ).pack(pady=(8, 12))

        self.after(1000, self._poll)

    def _open(self, program: Program):
        try:
            self._processes[program.script] = launch(program)
        except OSError as e:
            messagebox.showerror("DataDetector", f"{program.title} konnte nicht gestartet werden:\n{e}")
            return
        self._set_running(program, True)

    def _set_running(self, program: Program, running: bool):
        button, status = self._widgets[program.script]
        button.configure(state="disabled" if running else "normal")
        status.configure(text="● läuft" if running else "")

    def _poll(self):
        for program in PROGRAMS:
            process = self._processes.get(program.script)
            if process is None or process.poll() is None:
                continue
            del self._processes[program.script]
            self._set_running(program, False)
            if process.returncode != 0 and not program.console:
                messagebox.showerror(
                    "DataDetector",
                    f"{program.title} wurde mit einem Fehler beendet (Code {process.returncode}).\n\n"
                    f"{_tail(program.log_file)}"
                )
        self.after(1000, self._poll)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--watchdog":
        sys.exit(run_watchdog(sys.argv[2]))
    LauncherApp().mainloop()
