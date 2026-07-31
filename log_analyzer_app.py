"""
log_analyzer_app.py — Moderne Analyse- und Diagnose-App für Scan-Logs

Lädt JSONL-Tageslogs und stellt Statistiken, KPI-Karten, interaktive Diagramme (Matplotlib)
sowie eine detaillierte Such- und Filtertabelle inklusive Bildvorschau zur Verfügung.
Helles, modernes Industriedesign analog zum AI Vision Core.
"""

import csv
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk

# Matplotlib in Tkinter einbetten
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- UI Styling (Light Theme) ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

ACCENT   = "#2563EB"  # Blue
SUCCESS  = "#16A34A"  # Green
WARN     = "#D97706"  # Orange
DANGER   = "#DC2626"  # Red
BG_SIDE  = "#F1F5F9"  # Light gray sidebar
BG_CARD  = "#FFFFFF"  # White card background
BG_MAIN  = "#E2E8F0"  # Gray main background
BORDER   = "#CBD5E1"  # Border gray
TXT_DARK = "#1E293B"  # Text dark slate
TXT_MID  = "#475569"  # Text medium gray
TXT_LIGHT= "#94A3B8"  # Text light gray
HOVER_ROW = "#F1F5F9"  # Subtle row hover (Fluent Design)
SELECT_ROW= "#DBEAFE"  # Selected row highlight
HOVER_CARD= "#F8FAFC"  # KPI card hover

class LogAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Scan Log Analyzer  —  DataDetector v4.0")
        self.geometry("1400x850")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_MAIN)

        try:
            import scan_logger
            self._log_dir = scan_logger.resolve_log_directory(r"U:\Temp\DataMatrixReader.logFiles")
        except Exception:
            self._log_dir = r"U:\Temp\DataMatrixReader.logFiles"

        self._current_log_filepath = None
        self._records = []            # Alle geparsten Records der geladenen Datei
        self._filtered_records = []   # Derzeit gefilterte Ansicht
        self._selected_record = None  # Aktuell ausgewählter Record
        self._canvas_chart = None     # Matplotlib Canvas Referenz

        self._auto_refresh_enabled = True
        self._last_mtime = 0
        self._last_log_files_hash = ""
        self._detail_panel_visible = True
        self._selected_scan_ids = set()
        self._row_widgets = {}        # scan_id → row_frame Widget
        self._row_checkboxes = {}     # scan_id → CTkCheckBox Widget
        self._highlighted_scan_id = None  # Aktuell hervorgehobene Zeile

        # Gleit-Animation & Einklapp-Zustand für Seitenleiste
        self._sidebar_expanded = True
        self._sidebar_width = 280
        self._current_sidebar_width = 280
        self._sidebar_anim_job = None
        self._table_display_limit = 80  # Max. Zeilen pro Rendering (für max. Render-Geschwindigkeit)

        self._build_ui()
        self._load_log_files_list()

        # Automatischen Live-Poller starten (prüft alle 2 Sekunden)
        self.after(2000, self._auto_refresh_check)

    # ------------------------------------------------------------------ #
    #  UI Aufbau                                                          #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0) # Sidebar
        self.grid_columnconfigure(1, weight=1) # Hauptinhalt (Dashboard/Explorer)
        self.grid_columnconfigure(2, weight=0) # Detail-Panel rechts
        self.grid_rowconfigure(0, weight=1)

        # ------------------------- SIDEBAR -------------------------
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=BG_SIDE,
                                    border_width=1, border_color=BORDER)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(3, weight=1)

        # Sidebar Header mit Titel und Schließen-Button (<)
        sidebar_header = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        sidebar_header.grid(row=0, column=0, padx=(20, 10), pady=(18, 2), sticky="ew")
        sidebar_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sidebar_header, text="Log Analyzer",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=ACCENT
        ).grid(row=0, column=0, sticky="w")

        self.btn_toggle_close = ctk.CTkButton(
            sidebar_header, text="<", width=32, height=32,
            fg_color="transparent", text_color=TXT_DARK, hover_color=BORDER,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            corner_radius=6,
            command=self._toggle_sidebar
        )
        self.btn_toggle_close.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            self.sidebar, text="Statistiken & Diagnose v4.0",
            font=ctk.CTkFont(size=11), text_color=TXT_LIGHT
        ).grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        ctk.CTkLabel(
            self.sidebar, text="Kamera / Log-Quelle:",
            font=ctk.CTkFont(weight="bold", size=13), text_color=TXT_DARK
        ).grid(row=2, column=0, padx=20, pady=(10, 2), sticky="w")

        # Scrollbare Liste der Kameras & Log-Quellen
        self.files_frame = ctk.CTkScrollableFrame(
            self.sidebar, fg_color="transparent", label_text="Verfügbare Kameras"
        )
        self.files_frame.grid(row=3, column=0, padx=16, pady=(4, 16), sticky="nsew")
        self.files_buttons = []

        # Live-Update Switch
        self.auto_refresh_switch = ctk.CTkSwitch(
            self.sidebar, text="Live Auto-Update (2s)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=SUCCESS,
            command=self._toggle_auto_refresh
        )
        self.auto_refresh_switch.select()
        self.auto_refresh_switch.grid(row=4, column=0, padx=20, pady=(6, 0), sticky="w")

        # Info am Fuß der Sidebar
        self.disk_label = ctk.CTkLabel(
            self.sidebar, text="Festplatte: Prüfe...",
            font=ctk.CTkFont(size=11), text_color=TXT_MID
        )
        self.disk_label.grid(row=5, column=0, padx=20, pady=12, sticky="s")
        self._update_disk_info()

        # ------------------------- MAIN AREA -------------------------
        self.main_container = ctk.CTkTabview(self, fg_color=BG_CARD, segmented_button_selected_color=ACCENT, command=self._on_tab_changed)
        self.main_container.grid(row=0, column=1, padx=16, pady=16, sticky="nsew")

        # Schwebe-Button (>) zum Ausklappen der Seitenleiste (bei eingeklapptem Zustand)
        self.btn_toggle_open = ctk.CTkButton(
            self, text=">", width=36, height=36,
            fg_color=ACCENT, text_color="#FFFFFF", hover_color="#1D4ED8",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            corner_radius=8,
            command=self._toggle_sidebar
        )
        # Initial ausgeblendet, da Sidebar geöffnet startet
        
        self.tab_dashboard = self.main_container.add("Dashboard")
        self.tab_explorer = self.main_container.add("Log-Explorer")

        self._build_dashboard_tab()
        self._build_explorer_tab()

        # ------------------------- DETAIL PANEL -------------------------
        self.detail_panel = ctk.CTkFrame(self, width=380, corner_radius=12, fg_color=BG_CARD,
                                         border_width=1, border_color=BORDER)
        self.detail_panel.grid(row=0, column=2, padx=(0, 16), pady=16, sticky="nsew")
        self.detail_panel.grid_propagate(False)
        self.detail_panel.grid_rowconfigure(2, weight=1)
        self.detail_panel.grid_columnconfigure(0, weight=1)

        # Header Frame mit Titel + Schließen-Button (X)
        detail_header = ctk.CTkFrame(self.detail_panel, fg_color="transparent")
        detail_header.grid(row=0, column=0, padx=16, pady=(12, 4), sticky="ew")

        ctk.CTkLabel(
            detail_header, text="SCAN-DETAILS",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=TXT_LIGHT
        ).pack(side="left")

        close_btn = ctk.CTkButton(
            detail_header, text="✕", width=26, height=24,
            fg_color="transparent", text_color=TXT_MID, hover_color=BORDER,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._hide_detail_panel
        )
        close_btn.pack(side="right")

        # Bild-Kachel
        self.image_container = ctk.CTkFrame(self.detail_panel, height=200, fg_color=BG_MAIN, corner_radius=8)
        self.image_container.grid(row=1, column=0, padx=16, pady=4, sticky="ew")
        self.image_container.grid_propagate(False)
        self.image_label = ctk.CTkLabel(self.image_container, text="Kein Bild geladen", text_color=TXT_MID)
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        # Detail-Text (kein Scrollbar nötig, Inhalt passt immer)
        self.details_frame = ctk.CTkFrame(self.detail_panel, fg_color="transparent")
        self.details_frame.grid(row=2, column=0, padx=16, pady=(8, 16), sticky="nsew")
        self.details_frame.columnconfigure(0, weight=1)

        self.detail_labels = {}
        for key in ["Scan-ID", "Zeitstempel", "Endergebnis", "Methode", "Konfidenz", "DMTX", "OCR", "YOLO Conf", "Dauer", "Kamera", "Belichtung/Gain", "Flags"]:
            frame = ctk.CTkFrame(self.details_frame, fg_color="transparent")
            frame.pack(fill="x", pady=2)
            lbl_key = ctk.CTkLabel(frame, text=f"{key}:", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID, anchor="w", width=110)
            lbl_key.pack(side="left")
            lbl_val = ctk.CTkLabel(frame, text="—", font=ctk.CTkFont(size=11), text_color=TXT_DARK, anchor="w")
            lbl_val.pack(side="left", fill="x", expand=True)
            self.detail_labels[key] = lbl_val

    def _hide_detail_panel(self):
        self.detail_panel.grid_remove()
        self._detail_panel_visible = False

    def _show_detail_panel(self):
        if not self._detail_panel_visible:
            self.detail_panel.grid(row=0, column=2, padx=(0, 16), pady=16, sticky="nsew")
            self._detail_panel_visible = True

    # ------------------------------------------------------------------ #
    #  Seitenleiste Animation & Steuerung                                 #
    # ------------------------------------------------------------------ #
    def _toggle_sidebar(self):
        """Klappt die linke Seitenleiste per 60 FPS Hardware-Surface-Displacement (place) ein oder aus."""
        if self._sidebar_anim_job is not None:
            self.after_cancel(self._sidebar_anim_job)
            self._sidebar_anim_job = None

        target_x = 0 if not self._sidebar_expanded else -self._sidebar_width
        start_x = 0 if self._sidebar_expanded else -self._sidebar_width

        if not self._sidebar_expanded:
            self.sidebar.grid(row=0, column=0, sticky="nsew")
            if hasattr(self, "btn_toggle_open"):
                self.btn_toggle_open.place_forget()

        steps = 14
        duration_ms = 130
        interval = max(1, duration_ms // steps)
        step_count = 0

        def animate():
            nonlocal step_count
            step_count += 1
            progress = step_count / steps
            # Smooth cubic ease-out calculation
            eased = 1.0 - (1.0 - progress) ** 3
            current_x = int(start_x + (target_x - start_x) * eased)

            # High-speed surface placement (0.01ms overhead per frame)
            self.sidebar.place(x=current_x, y=0, relheight=1.0)
            self.sidebar.lift()

            if step_count < steps:
                self._sidebar_anim_job = self.after(interval, animate)
            else:
                self._sidebar_anim_job = None
                if target_x == -self._sidebar_width:
                    self.sidebar.grid_remove()
                    if hasattr(self, "btn_toggle_open"):
                        self.btn_toggle_open.place(x=12, y=24)
                        self.btn_toggle_open.lift()
                    self._sidebar_expanded = False
                else:
                    self.sidebar.grid(row=0, column=0, sticky="nsew")
                    self.sidebar.place_forget()
                    self._sidebar_expanded = True

        animate()

    # ------------------------------------------------------------------ #
    #  Dashboard Tab                                                      #
    # ------------------------------------------------------------------ #
    def _build_dashboard_tab(self):
        self.tab_dashboard.grid_columnconfigure(0, weight=1)
        self.tab_dashboard.grid_columnconfigure(1, weight=1)
        self.tab_dashboard.grid_columnconfigure(2, weight=1)
        self.tab_dashboard.grid_columnconfigure(3, weight=1)
        self.tab_dashboard.grid_rowconfigure(1, weight=1)

        # --- KPI Cards ---
        self.kpis = {}
        kpi_defs = [
            ("Scans gesamt", "total", TXT_DARK),
            ("Erfolgsrate", "success_rate", SUCCESS),
            ("Ø Scandauer", "avg_duration", ACCENT),
            ("Bilder im Speicher", "images", WARN)
        ]

        for i, (title, key, color) in enumerate(kpi_defs):
            card = ctk.CTkFrame(self.tab_dashboard, fg_color=BG_SIDE, border_width=1, border_color=BORDER, corner_radius=8)
            card.grid(row=0, column=i, padx=10, pady=16, sticky="ew")
            # Hover-Effekt für KPI-Cards (Fluent Design)
            card.bind("<Enter>", lambda e, c=card: c.configure(fg_color=HOVER_CARD))
            card.bind("<Leave>", lambda e, c=card: c.configure(fg_color=BG_SIDE))
            
            lbl_title = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID)
            lbl_title.pack(pady=(12, 2))
            
            lbl_val = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=24, weight="bold"), text_color=color)
            lbl_val.pack(pady=(0, 4) if key == "success_rate" else (0, 12))
            
            self.kpis[key] = lbl_val

            # Sub-Statistiken für die Erfolgsrate (Positiv in Grün / Negativ in Rot)
            if key == "success_rate":
                sub_frame = ctk.CTkFrame(card, fg_color="transparent")
                sub_frame.pack(pady=(0, 12))
                
                self.lbl_success_count = ctk.CTkLabel(sub_frame, text="0", text_color=SUCCESS, font=ctk.CTkFont(weight="bold", size=12))
                self.lbl_success_count.pack(side="left")
                
                ctk.CTkLabel(sub_frame, text=" / ", text_color=TXT_MID, font=ctk.CTkFont(size=12)).pack(side="left")
                
                self.lbl_fail_count = ctk.CTkLabel(sub_frame, text="0", text_color=DANGER, font=ctk.CTkFont(weight="bold", size=12))
                self.lbl_fail_count.pack(side="left")

        # --- Diagramm-Bereich ---
        self.chart_frame = ctk.CTkFrame(self.tab_dashboard, fg_color="transparent")
        self.chart_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=(0, 16), sticky="nsew")

    # ------------------------------------------------------------------ #
    #  Explorer Tab                                                       #
    # ------------------------------------------------------------------ #
    def _build_explorer_tab(self):
        self.tab_explorer.grid_columnconfigure(0, weight=1)
        self.tab_explorer.grid_rowconfigure(1, weight=1)

        # Toolbar Frame (2 Reihen für responsive Layout ohne Overlap)
        toolbar_frame = ctk.CTkFrame(self.tab_explorer, fg_color="transparent")
        toolbar_frame.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        toolbar_frame.columnconfigure(0, weight=1)

        # --- Reihe 1: Suche + Primäre Aktions-Buttons + Counter ---
        row1 = ctk.CTkFrame(toolbar_frame, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 6))

        # Suche
        self.search_entry = ctk.CTkEntry(row1, placeholder_text="Suche nach Code, ID, Flags...", width=220)
        self.search_entry.pack(side="left", padx=(0, 8))
        self.search_entry.bind("<KeyRelease>", lambda e: self._apply_filters())

        # Aktions-Buttons (Sofort sichtbar, direkt neben der Suche)
        self.btn_delete = ctk.CTkButton(
            row1, text="🗑️ Löschen", width=100,
            fg_color=DANGER, hover_color="#991B1B",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._delete_selected_records
        )
        self.btn_delete.pack(side="left", padx=4)

        self.btn_export = ctk.CTkButton(
            row1, text="📥 Exportieren", width=110,
            fg_color=ACCENT, hover_color="#1D4ED8",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._export_records
        )
        self.btn_export.pack(side="left", padx=4)

        # Zeilenanzahl Anzeige (rechtsbündig)
        self.record_count_label = ctk.CTkLabel(row1, text="0 Einträge", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID)
        self.record_count_label.pack(side="right", padx=10)

        # --- Reihe 2: Filter-Optionen ---
        row2 = ctk.CTkFrame(toolbar_frame, fg_color="transparent")
        row2.pack(fill="x")

        ctk.CTkLabel(row2, text="Filter:", font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID).pack(side="left", padx=(0, 6))

        # Filter Grade
        self.filter_grade = ctk.CTkOptionMenu(
            row2, values=["Alle Grades", "Grade A", "Grade B", "Grade C", "Grade D"], width=130,
            command=lambda v: self._apply_filters(), fg_color=BG_SIDE, text_color=TXT_DARK, button_color=BORDER, button_hover_color=TXT_LIGHT
        )
        self.filter_grade.pack(side="left", padx=4)

        # Filter Status & Methode
        self.filter_status = ctk.CTkOptionMenu(
            row2, values=["Alle Status", "Erfolg", "Fehlgeschlagen", "Verifiziert", "OCR", "Rekonstruiert"], width=150,
            command=lambda v: self._apply_filters(), fg_color=BG_SIDE, text_color=TXT_DARK, button_color=BORDER, button_hover_color=TXT_LIGHT
        )
        self.filter_status.pack(side="left", padx=4)

        # Checkbox für langsame Scans (>12 Sekunden)
        self.filter_slow_cb = ctk.CTkCheckBox(
            row2, text="Nur langsame Scans (>12s)",
            command=lambda: self._apply_filters(), text_color=TXT_DARK,
            fg_color=ACCENT, hover_color=ACCENT
        )
        self.filter_slow_cb.pack(side="left", padx=10)

        # Scrollbare Tabelle
        self.table_scroll = ctk.CTkScrollableFrame(self.tab_explorer, fg_color="transparent")
        self.table_scroll.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.table_scroll.columnconfigure(0, weight=1)
        
        # Spaltenüberschriften
        header_frame = ctk.CTkFrame(self.table_scroll, fg_color=BG_SIDE, height=30)
        header_frame.pack(fill="x", pady=(0, 4))
        
        # Alle-Auswählen Checkbox im Header
        self.select_all_cb = ctk.CTkCheckBox(
            header_frame, text="", width=24, height=20,
            checkbox_width=18, checkbox_height=18,
            command=self._toggle_select_all,
            fg_color=ACCENT, hover_color=ACCENT
        )
        self.select_all_cb.pack(side="left", padx=(10, 4))

        headers = [("Scan-ID", 180), ("Zeit", 180), ("Ergebnis", 120), ("Methode", 130), ("Konf.", 80), ("Grade", 80), ("Flags", 150)]
        for text, width in headers:
            lbl = ctk.CTkLabel(header_frame, text=text, font=ctk.CTkFont(size=11, weight="bold"), text_color=TXT_MID, width=width, anchor="w")
            lbl.pack(side="left", padx=10)

        self.table_rows_frame = ctk.CTkFrame(self.table_scroll, fg_color="transparent")
        self.table_rows_frame.pack(fill="both", expand=True)
        self.row_buttons = []

    # ------------------------------------------------------------------ #
    #  Log-Dateien & Kameras auflisten                                    #
    # ------------------------------------------------------------------ #
    def _load_log_files_list(self):
        for btn in self.files_buttons:
            btn.destroy()
        self.files_buttons.clear()

        try:
            import scan_logger
            self._log_dir = scan_logger.resolve_log_directory(r"U:\Temp\DataMatrixReader.logFiles")
        except Exception:
            pass

        if not os.path.exists(self._log_dir):
            return

        try:
            # 1. Existierende Tages-Logs (scans_*.jsonl) automatisch in scans.jsonl konsolidieren
            try:
                import scan_logger
                for root, dirs, _ in os.walk(self._log_dir):
                    try:
                        sl = scan_logger.ScanLogger(log_dir=root)
                    except Exception:
                        pass
            except Exception:
                pass

            # 2. Config-Datei laden für Kamera-Namen-Zuordnung (z.B. cam1 -> Kamera 1)
            config_map = {}
            config_file = "config.json"
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                        if "cameras" in cfg_data and isinstance(cfg_data["cameras"], list):
                            for c in cfg_data["cameras"]:
                                cid = c.get("id")
                                cname = c.get("name")
                                if cid and cname:
                                    config_map[cid.lower().strip()] = cname
                except Exception:
                    pass

            def get_pretty_camera_name(folder_name: str) -> str:
                clean = folder_name.strip()
                lower = clean.lower()
                if lower in config_map:
                    return config_map[lower]
                import re
                m = re.match(r"^(?:cam|camera)[_\s]*(\d+)$", lower, re.IGNORECASE)
                if m:
                    return f"Kamera {m.group(1)}"
                if lower == "cam1":
                    return "Kamera 1"
                if lower == "cam2":
                    return "Kamera 2"
                return clean.capitalize()

            camera_sources = []  # Tuples: (display_name, target_path)

            # Kamera-Unterordner in _log_dir suchen
            subdirs = [d for d in os.listdir(self._log_dir) if os.path.isdir(os.path.join(self._log_dir, d))]
            subdirs.sort()

            has_camera_subdirs = False
            for sd in subdirs:
                if sd.lower() == "images":
                    continue
                sd_path = os.path.join(self._log_dir, sd)
                has_jsonl = False
                for root, _, files in os.walk(sd_path):
                    if any(f.endswith(".jsonl") for f in files):
                        has_jsonl = True
                        break
                if has_jsonl or sd.lower().startswith("cam"):
                    has_camera_subdirs = True
                    disp_name = f"📷 {get_pretty_camera_name(sd)}"
                    camera_sources.append((disp_name, sd_path))

            # Direkte .jsonl-Dateien im Hauptverzeichnis prüfen
            root_jsonl = [f for f in os.listdir(self._log_dir) if f.endswith(".jsonl") and os.path.isfile(os.path.join(self._log_dir, f))]
            if root_jsonl:
                if not has_camera_subdirs:
                    camera_sources.append(("📷 Standard-Kamera", self._log_dir))
                else:
                    camera_sources.append(("📷 Hauptverzeichnis (Standard)", self._log_dir))

            # Top Button: "★ Alle Kameras (Gesamtübersicht)"
            all_name = "★ Alle Kameras (Gesamtübersicht)"
            all_btn = ctk.CTkButton(
                self.files_frame, text=all_name,
                fg_color="transparent", text_color=ACCENT,
                hover_color=BORDER, anchor="w", font=ctk.CTkFont(weight="bold"),
                command=lambda: self._load_log_file("__ALL__", all_name)
            )
            all_btn.pack(fill="x", pady=2)
            self.files_buttons.append(all_btn)

            # Kamera Buttons hinzufügen
            for disp_name, target_path in camera_sources:
                btn = ctk.CTkButton(
                    self.files_frame, text=disp_name,
                    fg_color="transparent", text_color=TXT_DARK,
                    hover_color=BORDER, anchor="w",
                    command=lambda p=target_path, name=disp_name: self._load_log_file(p, name)
                )
                btn.pack(fill="x", pady=2)
                self.files_buttons.append(btn)

            # Aktuelle Selektion beibehalten oder Standard "★ Alle Kameras (Gesamtübersicht)" laden
            if not self._current_log_filepath or self._current_log_filepath == "__ALL__":
                self._load_log_file("__ALL__", all_name)
            else:
                matched = False
                for disp_name, target_path in camera_sources:
                    if target_path == self._current_log_filepath:
                        self._load_log_file(target_path, disp_name)
                        matched = True
                        break
                if not matched:
                    self._load_log_file("__ALL__", all_name)

        except Exception as e:
            print(f"Fehler beim Auflisten der Logs: {e}")

    # ------------------------------------------------------------------ #
    #  Log-Datei / Kamera laden                                            #
    # ------------------------------------------------------------------ #
    def _load_log_file(self, filepath: str, display_name: str = ""):
        self._current_log_filepath = filepath
        for btn in self.files_buttons:
            btn_text = btn.cget("text")
            if btn_text == display_name or btn_text == filepath:
                btn.configure(fg_color=BORDER, font=ctk.CTkFont(weight="bold"))
            else:
                btn.configure(fg_color="transparent", font=ctk.CTkFont(weight="normal" if not btn_text.startswith("★") else "bold"))

        self._reload_current_file()

    # ------------------------------------------------------------------ #
    #  Live Auto-Refresh Polling Loop                                     #
    # ------------------------------------------------------------------ #
    def _toggle_auto_refresh(self):
        self._auto_refresh_enabled = bool(self.auto_refresh_switch.get())
        if self._auto_refresh_enabled:
            self.auto_refresh_switch.configure(text_color=SUCCESS, text="Live Auto-Update (2s)")
        else:
            self.auto_refresh_switch.configure(text_color=TXT_MID, text="Live Auto-Update (PAUSIERT)")

    def _auto_refresh_check(self):
        # Auto-Refresh verschieben, falls gerade eine Animation läuft
        if getattr(self, "_sidebar_anim_job", None) is not None:
            self.after(1000, self._auto_refresh_check)
            return

        if self._auto_refresh_enabled and self._log_dir and os.path.exists(self._log_dir):
            try:
                latest_mtime = 0.0
                file_count = 0
                for root, _, files in os.walk(self._log_dir):
                    for f in files:
                        if f.endswith(".jsonl"):
                            file_count += 1
                            try:
                                mt = os.path.getmtime(os.path.join(root, f))
                                if mt > latest_mtime:
                                    latest_mtime = mt
                            except Exception:
                                pass

                # Nur neu laden, wenn es echte Änderungen gab
                if latest_mtime != self._last_mtime and self._last_mtime != 0:
                    self._last_mtime = latest_mtime
                    self._reload_current_file()
                elif self._last_mtime == 0:
                    self._last_mtime = latest_mtime

                files_hash = f"{file_count}-{latest_mtime}"
                if files_hash != self._last_log_files_hash:
                    if self._last_log_files_hash != "":
                        self._load_log_files_list()
                    self._last_log_files_hash = files_hash
                    self._update_disk_info()
            except Exception:
                pass

        self.after(2000, self._auto_refresh_check)

    def _reload_current_file(self):
        if not self._current_log_filepath:
            return

        self._records.clear()
        try:
            target_files = []
            if self._current_log_filepath == "__ALL__":
                for root, _, files in os.walk(self._log_dir):
                    for f in files:
                        if f.endswith(".jsonl"):
                            target_files.append(os.path.join(root, f))
            elif os.path.exists(self._current_log_filepath):
                if os.path.isdir(self._current_log_filepath):
                    for root, _, files in os.walk(self._current_log_filepath):
                        for f in files:
                            if f.endswith(".jsonl"):
                                target_files.append(os.path.join(root, f))
                else:
                    target_files.append(self._current_log_filepath)

            latest_mtime = 0.0
            for tf in target_files:
                try:
                    mt = os.path.getmtime(tf)
                    if mt > latest_mtime:
                        latest_mtime = mt
                except Exception:
                    pass
            self._last_mtime = latest_mtime

            for tf in target_files:
                try:
                    with open(tf, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                if record.get("type") == "SESSION_END":
                                    continue
                                self._records.append(record)
                            except json.JSONDecodeError:
                                pass
                except Exception as e:
                    print(f"Fehler beim Lesen der Datei {tf}: {e}")

            # Chronologisch nach Zeitstempel sortieren (neueste zuerst)
            self._records.sort(key=lambda r: r.get("ts", ""), reverse=True)
        except Exception as e:
            print(f"Fehler beim Nachladen der Log-Datei: {e}")

        self._apply_filters()
        self._update_dashboard()

    # ------------------------------------------------------------------ #
    #  Filter anwenden & Tabelle aktualisieren                            #
    # ------------------------------------------------------------------ #
    def _apply_filters(self):
        self._table_display_limit = 80  # Limit bei neuen Filtern zurücksetzen
        search_query = self.search_entry.get().lower()
        grade_filter = self.filter_grade.get()
        status_filter = self.filter_status.get()

        self._filtered_records = []
        for r in self._records:
            # 1. Suchanfrage filtern
            code = str(r.get("result", {}).get("code", "")).lower()
            scan_id = str(r.get("scan_id", "")).lower()
            flags = " ".join(r.get("quality", {}).get("flags", [])).lower()
            if search_query and (search_query not in code and search_query not in scan_id and search_query not in flags):
                continue

            # 2. Grade filtern
            grade = r.get("quality", {}).get("grade", "")
            if grade_filter != "Alle Grades" and f"Grade {grade}" != grade_filter:
                continue

            # 3. Status & Methode filtern
            success = r.get("result", {}).get("success", False)
            method = r.get("result", {}).get("method") or "Fehler"
            if status_filter == "Erfolg" and not success:
                continue
            if status_filter == "Fehlgeschlagen" and success:
                continue
            if status_filter in ["Verifiziert", "OCR", "Rekonstruiert", "Fehler"] and method != status_filter:
                continue

            # 4. Langsame Scans filtern (> 12 Sekunden)
            total_ms = r.get("timing", {}).get("total_ms", r.get("duration_ms", 0))
            if hasattr(self, "filter_slow_cb") and self.filter_slow_cb.get() == 1 and total_ms <= 12000:
                continue

            self._filtered_records.append(r)

        # Tabelle neu zeichnen
        self._update_table()

    def _update_table(self):
        # Alte Zeilen entfernen
        for child in self.table_rows_frame.winfo_children():
            child.destroy()
        self._row_widgets.clear()
        self._row_checkboxes.clear()

        self._update_selection_counter()

        # Header Checkbox Status anpassen
        all_filtered_ids = {r.get("scan_id") for r in self._filtered_records if r.get("scan_id")}
        if all_filtered_ids and all_filtered_ids.issubset(self._selected_scan_ids):
            self.select_all_cb.select()
        else:
            self.select_all_cb.deselect()

        # Zeilen zeichnen (begrenzt auf _table_display_limit für max. GUI-Performance)
        visible_records = self._filtered_records[:self._table_display_limit]
        for idx, r in enumerate(visible_records):
            scan_id = r.get("scan_id", "")
            is_checked = scan_id in self._selected_scan_ids
            is_highlighted = scan_id == self._highlighted_scan_id
            success = r.get("result", {}).get("success", False)

            # Hintergrundfarbe: Highlight > Checked > Normal
            if is_highlighted:
                bg_color = SELECT_ROW
            elif is_checked:
                bg_color = "#EFF6FF"
            else:
                bg_color = BG_CARD

            row_frame = ctk.CTkFrame(self.table_rows_frame, fg_color=bg_color, height=35, corner_radius=4)
            row_frame.pack(fill="x", pady=2)
            row_frame.bind("<Button-1>", lambda e, rec=r: self._select_record(rec))

            # Robustes Hover-Handling (verhindert Flackern bei Bewegung über untergeordnete Labels)
            def _on_row_enter(e, f=row_frame, sid=scan_id):
                if sid != self._highlighted_scan_id:
                    f.configure(fg_color=HOVER_ROW)

            def _on_row_leave(e, f=row_frame, sid=scan_id):
                try:
                    x, y = f.winfo_pointerxy()
                    target = f.winfo_containing(x, y)
                    if target is not None and (target == f or str(target).startswith(str(f))):
                        return
                except Exception:
                    pass
                f.configure(fg_color=SELECT_ROW if sid == self._highlighted_scan_id
                           else ("#EFF6FF" if sid in self._selected_scan_ids else BG_CARD))

            row_frame.bind("<Enter>", _on_row_enter)
            row_frame.bind("<Leave>", _on_row_leave)

            # Referenz speichern
            self._row_widgets[scan_id] = row_frame

            # Row Checkbox
            row_cb = ctk.CTkCheckBox(
                row_frame, text="", width=24, height=20,
                checkbox_width=18, checkbox_height=18,
                fg_color=ACCENT, hover_color=ACCENT,
                command=lambda s_id=scan_id: self._toggle_row_selection(s_id)
            )
            if is_checked:
                row_cb.select()
            row_cb.pack(side="left", padx=(10, 4))
            self._row_checkboxes[scan_id] = row_cb

            # Daten extrahieren
            ts = r.get("ts", "")
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%H:%M:%S")
            except Exception:
                time_str = ts[:19].replace("T", " ")

            code = r.get("result", {}).get("code") or "—"
            method = r.get("result", {}).get("method") or "—"
            conf = f"{r.get('result', {}).get('confidence', 0.0) * 100:.0f}%"
            grade = r.get("quality", {}).get("grade", "—")
            flags = ", ".join(r.get("quality", {}).get("flags", []))

            # UI Labels — alle mit Click-Binding und Hover-Propagation
            for widget_data in [
                (scan_id, 180, TXT_DARK, None),
                (time_str, 180, TXT_MID, None),
                (code, 120, SUCCESS if success else DANGER, ctk.CTkFont(weight="bold")),
                (method, 130, TXT_MID, None),
                (conf, 80, TXT_MID, None),
            ]:
                text, width, color, font = widget_data
                lbl_kwargs = {"text": text, "width": width, "anchor": "w", "text_color": color}
                if font:
                    lbl_kwargs["font"] = font
                lbl = ctk.CTkLabel(row_frame, **lbl_kwargs)
                lbl.pack(side="left", padx=10)
                lbl.bind("<Button-1>", lambda e, rec=r: self._select_record(rec))

            # Grade Badge
            color_grade = SUCCESS if grade == "A" else (ACCENT if grade == "B" else (WARN if grade == "C" else DANGER))
            col_grade = ctk.CTkLabel(row_frame, text=f" {grade} ", font=ctk.CTkFont(weight="bold"), text_color="#FFFFFF", fg_color=color_grade, corner_radius=4, width=50)
            col_grade.pack(side="left", padx=(10, 40))
            col_grade.bind("<Button-1>", lambda e, rec=r: self._select_record(rec))

            # Flags
            col_flags = ctk.CTkLabel(row_frame, text=flags, text_color=TXT_LIGHT, anchor="w")
            col_flags.pack(side="left", padx=10, fill="x", expand=True)
            col_flags.bind("<Button-1>", lambda e, rec=r: self._select_record(rec))

            # Hover-Events auch an alle untergeordneten Zeilen-Widgets binden
            for child in row_frame.winfo_children():
                child.bind("<Enter>", _on_row_enter)
                child.bind("<Leave>", _on_row_leave)

        # Button zum Laden weiterer Einträge anzeigen, falls mehr Daten existieren
        if len(self._filtered_records) > len(visible_records):
            remaining = len(self._filtered_records) - len(visible_records)
            btn_more = ctk.CTkButton(
                self.table_rows_frame,
                text=f"▼  Weitere {min(80, remaining)} von {remaining} Einträgen anzeigen...",
                fg_color="transparent", text_color=ACCENT, hover_color=BORDER,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=self._load_more_table_rows
            )
            btn_more.pack(fill="x", pady=8)

    def _load_more_table_rows(self):
        self._table_display_limit += 80
        self._update_table()

    def _on_tab_changed(self, tab_name: str = ""):
        """Setzt die Filter auf Default zurück, wenn der Nutzer zum Dashboard wechselt."""
        if tab_name == "Dashboard" or (hasattr(self, "main_container") and self.main_container.get() == "Dashboard"):
            self._reset_filters()

    def _reset_filters(self):
        """Setzt alle Explorer-Filter auf Standardwerte zurück."""
        if hasattr(self, "filter_grade"):
            self.filter_grade.set("Alle Grades")
        if hasattr(self, "filter_status"):
            self.filter_status.set("Alle Status")
        if hasattr(self, "filter_slow_cb"):
            self.filter_slow_cb.deselect()
        if hasattr(self, "search_entry"):
            self.search_entry.delete(0, "end")
        self._apply_filters()

    # ------------------------------------------------------------------ #
    #  Zeilen-Highlight (ohne Tabellen-Rebuild)                           #
    # ------------------------------------------------------------------ #
    def _highlight_row(self, scan_id: str):
        """Hebt eine Zeile visuell hervor und setzt die vorherige zurück."""
        # Alte Zeile zurücksetzen
        if self._highlighted_scan_id and self._highlighted_scan_id in self._row_widgets:
            old_frame = self._row_widgets[self._highlighted_scan_id]
            old_bg = "#EFF6FF" if self._highlighted_scan_id in self._selected_scan_ids else BG_CARD
            try:
                old_frame.configure(fg_color=old_bg)
            except Exception:
                pass

        # Neue Zeile hervorheben
        self._highlighted_scan_id = scan_id
        if scan_id in self._row_widgets:
            try:
                self._row_widgets[scan_id].configure(fg_color=SELECT_ROW)
            except Exception:
                pass

    def _update_selection_counter(self):
        """Aktualisiert nur das Zähler-Label ohne Tabellen-Rebuild."""
        selected_count = len(self._selected_scan_ids)
        if selected_count > 0:
            self.record_count_label.configure(text=f"{len(self._filtered_records)} Einträge ({selected_count} ausgewählt)")
        else:
            self.record_count_label.configure(text=f"{len(self._filtered_records)} Einträge")

    # ------------------------------------------------------------------ #
    #  Selektion, Löschen & Exportieren                                    #
    # ------------------------------------------------------------------ #
    def _toggle_select_all(self):
        is_checked = bool(self.select_all_cb.get())
        if is_checked:
            for r in self._filtered_records:
                s_id = r.get("scan_id")
                if s_id:
                    self._selected_scan_ids.add(s_id)
                    if s_id in self._row_checkboxes:
                        self._row_checkboxes[s_id].select()
        else:
            self._selected_scan_ids.clear()
            for cb in self._row_checkboxes.values():
                cb.deselect()
        self._update_selection_counter()

    def _toggle_row_selection(self, scan_id: str):
        if scan_id in self._selected_scan_ids:
            self._selected_scan_ids.remove(scan_id)
        else:
            self._selected_scan_ids.add(scan_id)
        self._update_selection_counter()

    def _delete_selected_records(self):
        ids_to_delete = list(self._selected_scan_ids)
        if not ids_to_delete and self._selected_record:
            rec_id = self._selected_record.get("scan_id")
            if rec_id:
                ids_to_delete = [rec_id]

        if not ids_to_delete:
            messagebox.showinfo("Löschen", "Bitte wähle mindestens einen Scan in der Tabelle aus (per Checkbox oder Anklicken).")
            return

        count = len(ids_to_delete)
        confirm = messagebox.askyesno(
            "Scans löschen",
            f"Möchtest du wirklich {count} ausgewählte(n) Scan(s) dauerhaft aus den Log-Dateien löschen?"
        )
        if not confirm:
            return

        delete_set = set(ids_to_delete)

        target_files = []
        if self._current_log_filepath == "__ALL__":
            if os.path.exists(self._log_dir):
                for root, _, files in os.walk(self._log_dir):
                    for f in files:
                        if f.endswith(".jsonl"):
                            target_files.append(os.path.join(root, f))
        elif self._current_log_filepath and os.path.exists(self._current_log_filepath):
            if os.path.isdir(self._current_log_filepath):
                for root, _, files in os.walk(self._current_log_filepath):
                    for f in files:
                        if f.endswith(".jsonl"):
                            target_files.append(os.path.join(root, f))
            else:
                target_files.append(self._current_log_filepath)

        for tf in target_files:
            try:
                if not os.path.exists(tf):
                    continue
                with open(tf, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        rec_id = rec.get("scan_id")
                        if rec_id in delete_set:
                            # Zugehöriges Bild auch von der Festplatte löschen
                            img_path = rec.get("image", {}).get("path")
                            if img_path:
                                for p in [os.path.join(os.path.dirname(tf), img_path), os.path.join(self._log_dir, img_path)]:
                                    if os.path.exists(p):
                                        try:
                                            os.remove(p)
                                        except Exception:
                                            pass
                            continue
                        new_lines.append(line)
                    except Exception:
                        new_lines.append(line)

                with open(tf, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            except Exception as e:
                print(f"Fehler beim Löschen aus Datei {tf}: {e}")

        self._selected_scan_ids.difference_update(delete_set)
        if self._selected_record and self._selected_record.get("scan_id") in delete_set:
            self._selected_record = None

        self._reload_current_file()
        messagebox.showinfo("Erfolg", f"{count} Scan-Eintrag/Einträge erfolgreich gelöscht.")

    def _export_records(self):
        export_list = [r for r in self._filtered_records if r.get("scan_id") in self._selected_scan_ids]
        if not export_list:
            export_list = self._filtered_records

        if not export_list:
            messagebox.showinfo("Export", "Keine Einträge zum Exportieren vorhanden.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Scan-Logs exportieren",
            defaultextension=".csv",
            filetypes=[("CSV Datei", "*.csv"), ("JSON Datei", "*.json")]
        )
        if not filepath:
            return

        try:
            if filepath.endswith(".json"):
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(export_list, f, indent=2, ensure_ascii=False)
            else:
                with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(["Scan-ID", "Zeitstempel", "Code", "Erfolg", "Methode", "Konfidenz", "Grade", "Flags", "Dauer_ms", "Kamera"])
                    for r in export_list:
                        res = r.get("result", {})
                        qual = r.get("quality", {})
                        timing = r.get("timing", {})
                        meta = r.get("meta", {})
                        writer.writerow([
                            r.get("scan_id", ""),
                            r.get("ts", ""),
                            res.get("code", ""),
                            "JA" if res.get("success") else "NEIN",
                            res.get("method", ""),
                            f"{res.get('confidence', 0.0)*100:.1f}%",
                            qual.get("grade", ""),
                            ", ".join(qual.get("flags", [])),
                            timing.get("total_ms", 0),
                            meta.get("camera_model", "")
                        ])
            messagebox.showinfo("Export erfolgreich", f"{len(export_list)} Datensätze erfolgreich exportiert nach:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Fehler", f"Fehler beim Exportieren: {e}")

    # ------------------------------------------------------------------ #
    #  Dashboard berechnen & Diagramme zeichnen                            #
    # ------------------------------------------------------------------ #
    def _update_dashboard(self):
        total = len(self._records)
        if total == 0:
            for k in self.kpis:
                self.kpis[k].configure(text="—")
            self.lbl_success_count.configure(text="—")
            self.lbl_fail_count.configure(text="—")
            return

        # KPI-Berechnungen
        successful = sum(1 for r in self._records if r.get("result", {}).get("success", False))
        success_rate = (successful / total) * 100

        total_dur = sum(r.get("timing", {}).get("total_ms", 0) for r in self._records)
        avg_dur = total_dur / total

        images_saved = sum(1 for r in self._records if r.get("image", {}).get("path") is not None)

        self.kpis["total"].configure(text=str(total))
        self.kpis["success_rate"].configure(text=f"{success_rate:.1f}%")
        self.lbl_success_count.configure(text=str(successful))
        self.lbl_fail_count.configure(text=str(total - successful))
        self.kpis["avg_duration"].configure(text=f"{avg_dur:.0f} ms")
        self.kpis["images"].configure(text=str(images_saved))

        # Diagramme zeichnen
        self._draw_charts()

    def _draw_charts(self):
        # Altes Diagramm entfernen
        if self._canvas_chart is not None:
            self._canvas_chart.get_tk_widget().destroy()
            self._canvas_chart = None

        # Grade-Verteilung zählen
        grades = {"A": 0, "B": 0, "C": 0, "D": 0}
        methods = {}
        for r in self._records:
            grade = r.get("quality", {}).get("grade", "D")
            grades[grade] = grades.get(grade, 0) + 1
            
            method = r.get("result", {}).get("method") or "Fehler"
            methods[method] = methods.get(method, 0) + 1

        # Matplotlib-Figure erstellen
        fig = Figure(figsize=(10, 4.5), dpi=100, facecolor="#FFFFFF")
        
        # 1. Donut Chart für Grades (interaktiv — Klick filtert Log-Explorer)
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor("#FFFFFF")
        
        labels = [f"Grade {k}" for k in grades.keys() if grades[k] > 0]
        sizes = [grades[k] for k in grades.keys() if grades[k] > 0]
        GRADE_COLORS = {
            "A": "#22C55E",  # Grün (Verifiziert)
            "B": "#EAB308",  # Gelb (OCR)
            "C": "#F97316",  # Orange (Rekonstruiert)
            "D": "#EF4444",  # Rot (Fehler)
        }
        used_colors = [GRADE_COLORS.get(k.replace("Grade ", ""), "#94A3B8") for k in grades.keys() if grades[k] > 0]

        if sizes:
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, autopct='%1.0f%%', startangle=90,
                colors=used_colors, pctdistance=0.75,
                textprops=dict(color=TXT_DARK, fontsize=9)
            )
            # Loch in der Mitte erzeugen (Donut)
            centre_circle = matplotlib.patches.Circle((0,0), 0.55, fc='white')
            ax1.add_artist(centre_circle)
            for i, wedge in enumerate(wedges):
                wedge.set_picker(True)
                wedge.set_label(labels[i])
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
        else:
            ax1.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
        ax1.set_title("Qualitäts-Verteilung (Grades)", fontsize=11, weight="bold", color=TXT_DARK)

        # 2. Balkendiagramm für Methoden (semantische Farben: Grün → Gelb → Orange → Rot)
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor("#FFFFFF")
        
        # Gitterlinien
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        ax2.set_axisbelow(True)

        if methods:
            sorted_methods = sorted(methods.items(), key=lambda x: -x[1])
            m_labels = [x[0] for x in sorted_methods]
            m_counts = [x[1] for x in sorted_methods]
            
            METHOD_COLORS = {
                "Verifiziert": "#22C55E",
                "OCR": "#EAB308",
                "Rekonstruiert": "#F97316",
                "Fehler": "#EF4444",
            }
            m_colors = [METHOD_COLORS.get(l, "#94A3B8") for l in m_labels]

            bars = ax2.bar(m_labels, m_counts, color=m_colors, width=0.5, edgecolor="none", picker=True)
            for bar, label in zip(bars, m_labels):
                bar.set_label(label)
            
            # Werte über den Balken anzeigen
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, color=TXT_MID)
        else:
            ax2.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
            
        ax2.set_title("Erfolgsmethoden", fontsize=11, weight="bold", color=TXT_DARK)
        ax2.tick_params(colors=TXT_MID, labelsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color(BORDER)
        ax2.spines['bottom'].set_color(BORDER)

        fig.tight_layout()

        self._chart_wedges = wedges if sizes else []
        self._chart_bars = bars if methods else []
        for b in self._chart_bars:
            b._orig_x = b.get_x()
            b._orig_w = b.get_width()

        # Canvas in Tkinter einbetten + interaktive Pick/Hover-Events verbinden
        self._canvas_chart = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self._canvas_chart.mpl_connect('pick_event', self._on_chart_pick)
        self._canvas_chart.mpl_connect('motion_notify_event', self._on_chart_hover)
        self._canvas_chart.draw()
        self._canvas_chart.get_tk_widget().pack(fill="both", expand=True)

    def _on_chart_hover(self, event):
        """Erzeugt einen flüssigen Hover-Effekt (Vergrößern + Helligkeit + Hand-Cursor)."""
        found = None
        if event.inaxes is not None:
            if hasattr(self, "_chart_wedges"):
                for w in self._chart_wedges:
                    if w.contains(event)[0]:
                        found = w
                        break
            if not found and hasattr(self, "_chart_bars"):
                for b in self._chart_bars:
                    if b.contains(event)[0]:
                        found = b
                        break

        if found != getattr(self, "_hovered_chart_artist", None):
            # Vorheriges Element zurücksetzen
            prev = getattr(self, "_hovered_chart_artist", None)
            if prev is not None:
                try:
                    if hasattr(prev, "set_radius"):
                        prev.set_radius(1.0)
                        prev.set_alpha(1.0)
                        prev.set_linewidth(1)
                        prev.set_edgecolor("white")
                    elif hasattr(prev, "_orig_x"):
                        prev.set_x(prev._orig_x)
                        prev.set_width(prev._orig_w)
                        prev.set_alpha(1.0)
                        prev.set_linewidth(0)
                except Exception:
                    pass

            self._hovered_chart_artist = found

            # Neues Element vergrößern & hervorheben
            if found is not None:
                try:
                    if hasattr(found, "set_radius"):
                        found.set_radius(1.08)
                        found.set_alpha(0.85)
                        found.set_linewidth(2)
                        found.set_edgecolor("#1E293B")
                    elif hasattr(found, "_orig_x"):
                        found.set_x(found._orig_x - 0.04)
                        found.set_width(found._orig_w + 0.08)
                        found.set_alpha(0.85)
                        found.set_linewidth(2)
                        found.set_edgecolor("#1E293B")
                except Exception:
                    pass

            if self._canvas_chart is not None:
                try:
                    cursor_style = "hand2" if found is not None else ""
                    self._canvas_chart.get_tk_widget().configure(cursor=cursor_style)
                    self._canvas_chart.draw_idle()
                except Exception:
                    pass

    def _on_chart_pick(self, event):
        """Wird aufgerufen wenn auf ein Diagramm-Element geklickt wird."""
        artist = event.artist
        label = getattr(artist, '_label', '') or artist.get_label()

        if not label or label.startswith('_'):
            return

        if label.startswith("Grade "):
            # Donut-Chart: Grade-Filter setzen
            self.filter_grade.set(label)
            self.filter_status.set("Alle Status")
            self.search_entry.delete(0, "end")
        elif label in ("Verifiziert", "OCR", "Rekonstruiert", "Fehler"):
            # Balkendiagramm: Status/Methoden-OptionMenu direkt auf die geklickte Methode setzen
            self.filter_grade.set("Alle Grades")
            self.filter_status.set(label)
            self.search_entry.delete(0, "end")

        # Zum Log-Explorer wechseln und Filter anwenden
        self.main_container.set("Log-Explorer")
        self._apply_filters()

    # ------------------------------------------------------------------ #
    #  Record-Auswahl & Detailansicht                                      #
    # ------------------------------------------------------------------ #
    def _select_record(self, record: dict):
        self._selected_record = record
        self._show_detail_panel()

        # Zeile visuell hervorheben (ohne Tabellen-Rebuild)
        scan_id = record.get("scan_id", "")
        self._highlight_row(scan_id)

        # Bild sofort ausblenden (wird weiter unten ggf. wieder eingeblendet)
        self.image_container.grid_remove()
        self.image_label.image = None

        # Detail-Labels belegen
        res = record.get("result", {})
        timing = record.get("timing", {})
        dmtx = record.get("dmtx", {})
        ocr = record.get("ocr", {})
        det = record.get("detection", {})
        qual = record.get("quality", {})
        meta = record.get("meta", {})

        self.detail_labels["Scan-ID"].configure(text=record.get("scan_id", "—"))
        
        ts = record.get("ts", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_str = dt.strftime("%d.%m.%Y  %H:%M:%S")
        except Exception:
            ts_str = ts.replace("T", " ")
        self.detail_labels["Zeitstempel"].configure(text=ts_str)

        self.detail_labels["Endergebnis"].configure(
            text=res.get("code") or "Fehlgeschlagen",
            text_color=SUCCESS if res.get("success") else DANGER
        )
        self.detail_labels["Methode"].configure(text=res.get("method") or "—")
        self.detail_labels["Konfidenz"].configure(text=f"{res.get('confidence', 0.0) * 100:.1f}%")
        self.detail_labels["DMTX"].configure(text=dmtx.get("text") or "Nicht lesbar")
        self.detail_labels["OCR"].configure(text=ocr.get("text") or (ocr.get("partial") or "Nicht lesbar"))
        self.detail_labels["YOLO Conf"].configure(text=f"{det.get('yolo_conf', 0.0) * 100:.1f}%")
        self.detail_labels["Dauer"].configure(
            text=f"Gesamt: {timing.get('total_ms', 0)}ms  (YOLO: {timing.get('yolo_ms', 0)}ms, OCR: {timing.get('ocr_ms', 0)}ms, DMTX: {timing.get('dmtx_ms', 0)}ms)"
        )
        self.detail_labels["Kamera"].configure(text=meta.get("camera_model") or "—")
        exp_us = meta.get("exposure_us")
        if exp_us is None or float(exp_us) == 0:
            exp_us = float(meta.get("last_exposure", 6.0)) * 1000.0 if meta.get("last_exposure") else 6000.0
        else:
            exp_us = float(exp_us)

        gain_val = meta.get("gain")
        if gain_val is None or float(gain_val) == 0:
            gain_val = float(meta.get("last_gain", 2.0)) if meta.get("last_gain") else 2.0
        else:
            gain_val = float(gain_val)

        self.detail_labels["Belichtung/Gain"].configure(
            text=f"Exp: {exp_us / 1000.0:.1f} ms  |  Gain: {gain_val:.2f}"
        )
        self.detail_labels["Flags"].configure(text=", ".join(qual.get("flags", [])) or "Keine")

        # Bild laden — Container dynamisch ein-/ausblenden
        img_path = record.get("image", {}).get("path")
        has_image = False

        if img_path:
            possible_paths = [
                os.path.join(self._log_dir, img_path),
                os.path.join(self._log_dir, record.get("meta", {}).get("cam_id", ""), img_path)
            ]
            if os.path.exists(self._log_dir):
                try:
                    for sub in os.listdir(self._log_dir):
                        sub_p = os.path.join(self._log_dir, sub)
                        if os.path.isdir(sub_p):
                            possible_paths.append(os.path.join(sub_p, img_path))
                except Exception:
                    pass

            full_img_path = None
            for p in possible_paths:
                if p and os.path.exists(p):
                    full_img_path = p
                    break

            if full_img_path and os.path.exists(full_img_path):
                try:
                    pil_img = Image.open(full_img_path)
                    w_max, h_max = 340, 190
                    w, h = pil_img.size
                    scale = min(w_max / w, h_max / h)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(nw, nh))
                    self.image_label.configure(image=ctk_img, text="")
                    self.image_label.image = ctk_img
                    has_image = True
                except Exception as e:
                    self.image_label.configure(image=None, text=f"Fehler beim Laden:\n{e}", text_color=DANGER)
                    self.image_label.image = None

        # Bild-Container ein-/ausblenden
        if has_image:
            self.image_container.grid(row=1, column=0, padx=16, pady=4, sticky="ew")
        else:
            self.image_container.grid_remove()

    # ------------------------------------------------------------------ #
    #  Hilfsfunktionen                                                     #
    # ------------------------------------------------------------------ #
    def _update_disk_info(self):
        try:
            path = os.path.abspath(self._log_dir)
            import shutil
            usage = shutil.disk_usage(path)
            used_percent = (usage.used / usage.total) * 100
            free_gb = usage.free / (1024**3)
            
            color = DANGER if used_percent > 70 else (WARN if used_percent > 50 else TXT_MID)
            self.disk_label.configure(
                text=f"Laufwerk: {used_percent:.1f}% belegt\n({free_gb:.1f} GB frei)",
                text_color=color
            )
        except Exception:
            self.disk_label.configure(text="Festplatte: N/A")

if __name__ == "__main__":
    app = LogAnalyzerApp()
    app.mainloop()
