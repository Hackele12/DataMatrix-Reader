"""
Scan-Pipelines: scan_2class() für YOLO-Detektionen (DataMatrix + Text) und scan() für ein Gesamtbild
bzw. einen Etikett-Ausschnitt. Priorität: keine Fehllesungen – lieber "Fehler" als ein falscher Code.

Ablauf: schnelle DataMatrix-Stufen (scan_fast: zxing, Modul-Decoder) laufen immer; langsame Rückfallebenen
(OCR-Gegenprobe, Gesamtbild-Scan, Gamma-Fallback) nur im Zeitbudget config.SCAN_TIME_BUDGET_S.
"""

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config
from .code_format import dmx_text_to_code, is_valid_horden_code, normalize_ocr_confusions
from .dmx_decoder import decode_zxing_stream, scan_datamatrix_pipeline, zxing_dmx_stream
from .dmx_module_reader import read_dmx_modules, text_crop_from_quad
from .fusion import merge_results, try_reconstruct
from .horde_matching import correct_ocr_confusion, match_horde_image
from .image_ops import deskew_crop, gamma_lut, to_gray
from .ocr import read_ocr_with_status
from .onnx_models import predict_pacc
from .ref_images import scan_reference_image_pipeline
from .results import (aborted_result, dmx_blocked, dmx_final_result, dmx_soft_result, error_result, ocr_failed,
                      scan_result)

logger = logging.getLogger(__name__)

MIN_DMX_YOLO_CONF = 0.30
MIN_TXT_YOLO_CONF = 0.25

# Erste zxing-Runde vor dem Modul-Decoder: Rohausschnitte + die günstigen Varianten der besten Region
# (Benchmark: 146 von 157 Varianten-Treffern liegen in den ersten 24 Varianten, Median 14 ms).
_ZXING_FIRST_PASS = 28

# Gamma-Werte des Aufhell-Fallbacks für dunkle/unterbelichtete Bilder
_GAMMA_FALLBACK_VALUES = (0.25, 0.3, 0.35, 0.4, 0.5)

# Konfidenz, ab der ein unverifiziertes scan()-Fallback-Ergebnis akzeptiert wird (W031 mit 0.70 ja, W631 mit 0.63 nein)
_MIN_FALLBACK_CONF = 0.68

# Bekannte OCR-Ziffernverwechslungen für den Soft-Match der Gegenprobe
_DIGIT_CONFUSION_PAIRS = {('0', '4'), ('4', '0'), ('0', '6'), ('6', '0'), ('0', '8'), ('8', '0'),
                          ('3', '8'), ('8', '3'), ('0', '9'), ('9', '0')}


def _elapsed_ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def new_deadline(budget_s: float | None = None) -> float:
    """Deadline (time.perf_counter()) für einen Scan; ohne Angabe gilt config.SCAN_TIME_BUDGET_S."""
    return time.perf_counter() + (config.SCAN_TIME_BUDGET_S if budget_s is None else budget_s)


def _time_left(deadline: float | None) -> float:
    return float("inf") if deadline is None else deadline - time.perf_counter()


# --------------------------------------------------------------------------- #
#  Schnelle DataMatrix-Stufen                                                  #
# --------------------------------------------------------------------------- #

def _module_result(module: dict) -> dict:
    """Modul-Decoder-Ergebnis → Scan-Ergebnis (Reed-Solomon-bestätigt = Verifiziert, sonst Modulabgleich)."""
    if module["tier"] == "soft":
        result = dmx_soft_result(module["code"], f"DataMatrix Modulabgleich (NCC {module['ncc']:.2f}, "
                                                 f"Marge {module['margin']:.2f})", round(module["ncc"], 2))
    else:
        erasures = f" + {module['erasures']} Löschungen" if module["tier"] == "rs_erasure" else ""
        result = dmx_final_result(module["code"], f"DataMatrix dekodiert (Modul-Decoder, Reed-Solomon{erasures})")
    result["_module"] = {
        "tier": module["tier"], "ncc": round(module["ncc"], 3), "margin": round(module["margin"], 3),
        "frame_t": round(module["frame_t"], 2), "rs_errors": module["rs_errors"], "erasures": module["erasures"],
        "quad": None if module["quad"] is None else [[round(float(v), 1) for v in p] for p in module["quad"]],
    }
    return result


def _dmx_search_boxes(detections: list[dict]) -> list[tuple]:
    """YOLO-DataMatrix-Boxen (beste zuerst); ohne DMX-Detektion die aus der Textbox abgeleitete Box."""
    boxes = [det["box"] for det in sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
             if det.get("cls") == 0]
    if not boxes and detections:
        dmx_det, _ = select_label_detections(detections)
        if dmx_det is not None:
            boxes = [dmx_det["box"]]
    return boxes


def scan_fast(frame: np.ndarray, detections=(), deadline: float | None = None, zxing_rest: bool = True,
              hints: dict | None = None) -> dict | None:
    """
    Schnelle DataMatrix-Stufen, in dieser Reihenfolge: zxing auf Rohausschnitten und günstigen Varianten →
    Modul-Decoder mit Reed-Solomon-Bestätigung → restliche zxing-Varianten → Modul-Decoder per
    Codebuch-Abgleich (Soft).

    Args:
        frame: Kamerabild (BGR oder Graustufen).
        detections: YOLO-Detektionen wie bei scan_2class(); leer = Suche im Gesamtbild.
        deadline: time.perf_counter()-Zeitpunkt für den Modul-Decoder (weitere Kandidaten).
        zxing_rest: restliche zxing-Varianten (bis ~0,7 s) prüfen; für Bestätigungsbilder abschaltbar.
        hints: optionales dict; erhält unter "module" ein nicht angenommenes Modul-Decoder-Ergebnis
               (Symbolviereck + bester Kandidat) für den OCR-Abgleich in scan_2class().

    Returns:
        Scan-Ergebnis ("Verifiziert" oder "Modulabgleich") oder None.
    """
    if frame is None or frame.size == 0:
        return None
    boxes = _dmx_search_boxes(list(detections))
    stream = zxing_dmx_stream(frame, boxes)
    code, detail = decode_zxing_stream(stream, _ZXING_FIRST_PASS)
    if code:
        return dmx_final_result(code, f"DataMatrix dekodiert (zxing-cpp, {detail})")

    module = None
    if config.USE_MODULE_READER:
        try:
            module = read_dmx_modules(frame, boxes, deadline)
        except Exception as e:
            logger.warning(f"[MODUL] Modul-Decoder Fehler: {e}")
        if module is not None and module["tier"] in ("rs", "rs_erasure"):
            return _module_result(module)

    code, detail = decode_zxing_stream(stream) if zxing_rest else (None, None)
    if code:
        return dmx_final_result(code, f"DataMatrix dekodiert (zxing-cpp, {detail})")
    if module is not None and module["tier"] == "soft":
        return _module_result(module)
    if hints is not None and module is not None and module["quad"] is not None:
        hints["module"] = module
    return None


# Modul-Decoder-Kandidat, der nur zusammen mit einer übereinstimmenden OCR-Lesung angenommen wird
_OCR_CONFIRM_MIN_NCC = 0.50
_OCR_CONFIRM_MIN_MARGIN = 0.10


def _usable_module_hint(hint: dict | None) -> dict | None:
    """Hinweis nur mit echtem Symbolrahmen (Rahmen-t wie bei Soft-Ergebnissen) und Viereck."""
    if hint and hint.get("quad") is not None and hint.get("frame_t", 0.0) >= config.SOFT_MIN_FRAME_T:
        return hint
    return None


def _ocr_confirmed_module_result(hint: dict, ocr_result: dict) -> dict | None:
    """Zwei unabhängige Quellen: Codebuch-Abgleich der DataMatrix und OCR der Klarschrift nennen denselben Code."""
    ocr_text = ocr_result.get("text") if ocr_result.get("status") == "ok" else None
    if not ocr_text or ocr_text != hint.get("best"):
        return None
    if hint["ncc"] < _OCR_CONFIRM_MIN_NCC or hint["margin"] < _OCR_CONFIRM_MIN_MARGIN:
        return None
    result = scan_result(True, ocr_text, "Verifiziert", round(max(hint["ncc"], ocr_result.get("confidence", 0.0)), 2),
                         dmtx=hint["best"], ocr=ocr_text, verified=True, partial=ocr_text)
    result["method_detail"] = (f"DataMatrix-Modulabgleich + OCR übereinstimmend (NCC {hint['ncc']:.2f}, "
                               f"Marge {hint['margin']:.2f})")
    return result


# --------------------------------------------------------------------------- #
#  Crop-Pipelines                                                              #
# --------------------------------------------------------------------------- #

def scan_datamatrix(frame: np.ndarray) -> dict:
    """
    Reine DataMatrix-Auswertung eines zugeschnittenen DMX-Crops (zxing, pylibdmtx, RefImg, Rekonstruktion).

    Returns:
        dict mit status, text, confidence, method_detail, observed_grid.
    """
    if frame is None or frame.size == 0:
        return dmx_blocked("Kein Bild")

    t0 = time.time()
    dmx_result = scan_datamatrix_pipeline(frame)
    if dmx_result.get("status") == "decoded" and dmx_result.get("text"):
        code = dmx_result["text"]
        if is_valid_horden_code(code):
            logger.info(f"[2CLASS-DMX] DataMatrix direkt erkannt: '{code}' ({_elapsed_ms(t0)}ms)")
            dmx_result["confidence"] = 1.0
            return dmx_result

    ref_result = scan_reference_image_pipeline(frame)
    if ref_result.get("status") == "matched" and ref_result.get("text"):
        ref_text = ref_result["text"]
        ref_conf = ref_result.get("confidence", 0.0)
        if ref_conf >= 0.75 and is_valid_horden_code(ref_text):
            logger.info(f"[2CLASS-DMX] RefImg Match: '{ref_text}' (Conf={ref_conf:.2f}, {_elapsed_ms(t0)}ms)")
            if dmx_result.get("text"):
                # Ein DMX-Ergebnis hat Vorrang vor dem Referenzbild
                dmx_result["_ref_img_text"] = ref_text
                dmx_result["_ref_img_conf"] = ref_conf
                return dmx_result
            return {"status": "matched_refimg", "text": ref_text, "confidence": ref_conf,
                    "method_detail": "RefImg", "observed_grid": None}

    # Blinde 4.000-Code-Rekonstruktion ist auf Crops unzuverlässig → Konfidenz auf 0.85 begrenzt
    recon_result = try_reconstruct(frame, None, 0.0, None)
    if recon_result is not None and recon_result.get("success"):
        recon_conf = min(0.85, recon_result.get("confidence", 0.85))
        logger.info(f"[2CLASS-DMX] Rekonstruktion: '{recon_result['result']}' (Conf={recon_conf:.2f}, {_elapsed_ms(t0)}ms)")
        return {"status": "reconstructed", "text": recon_result["result"], "confidence": recon_conf,
                "method_detail": recon_result.get("method", "Rekonstruiert"), "observed_grid": None}

    logger.info(f"[2CLASS-DMX] Keine DataMatrix erkannt ({_elapsed_ms(t0)}ms)")
    return dmx_result


def scan_ocr(frame: np.ndarray, text_crop: bool = False) -> dict:
    """
    Reine OCR-Auswertung eines zugeschnittenen Text-Crops (EasyOCR, optional PACC).
    text_crop: entzerrter Klarschrift-Ausschnitt aus der Symbolgeometrie (ganzes Bild auswerten).

    Returns:
        dict mit status, text, confidence, partial_display, readable_chars, missing_positions, raw_candidate.
    """
    if frame is None or frame.size == 0:
        return ocr_failed()

    t0 = time.time()
    ocr_result = read_ocr_with_status(frame, text_crop)
    ocr_text = ocr_result.get("text")

    pacc_text, pacc_conf = predict_pacc(frame)
    if pacc_text and pacc_conf > 0.0:
        logger.info(f"[2CLASS-OCR] PACC: '{pacc_text}' (Conf={pacc_conf:.2f})")
        if not ocr_text and pacc_conf >= 0.70:
            ocr_result.update(text=pacc_text, confidence=pacc_conf, status="ok", partial_display=pacc_text)
        elif ocr_text and pacc_text == ocr_text:
            ocr_result["confidence"] = max(ocr_result.get("confidence", 0.0), pacc_conf)

    logger.info(
        f"[2CLASS-OCR] Ergebnis: status={ocr_result.get('status')}, "
        f"text='{ocr_result.get('text')}', conf={ocr_result.get('confidence', 0.0):.2f} ({_elapsed_ms(t0)}ms)"
    )
    return ocr_result


# --------------------------------------------------------------------------- #
#  2-Klassen-Pipeline                                                          #
# --------------------------------------------------------------------------- #

def _derive_neighbor_box(box) -> tuple[int, int, int, int]:
    """Suchbereich für die fehlende Klasse: DataMatrix und Text liegen auf dem Etikett direkt nebeneinander."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return (max(0, x1 - int(w * 1.5)), max(0, y1 - int(h * 1.5)), x2 + int(w * 2.5), y2 + int(h * 2.5))


def select_label_detections(detections: list[dict]) -> tuple[dict | None, dict | None]:
    """Beste DataMatrix- (Klasse 0) und Text-Detektion (Klasse 1); fehlt eine, wird sie aus der anderen abgeleitet."""
    dmx_det = None
    txt_det = None
    for det in detections:
        cls = det.get("cls", -1)
        if cls == 0:
            if det["conf"] < MIN_DMX_YOLO_CONF:
                logger.info(f"[2CLASS] DMX-Detection verworfen: conf={det['conf']:.2f} < {MIN_DMX_YOLO_CONF}")
            elif dmx_det is None or det["conf"] > dmx_det["conf"]:
                dmx_det = det
        elif cls == 1 and det["conf"] >= MIN_TXT_YOLO_CONF:
            if txt_det is None or det["conf"] > txt_det["conf"]:
                txt_det = det

    if dmx_det is not None and txt_det is None:
        derived = _derive_neighbor_box(dmx_det["box"])
        txt_det = {"cls": 1, "box": derived, "conf": dmx_det["conf"], "derived": True}
        logger.info(f"[2CLASS] Text-Box aus DataMatrix-Box abgeleitet: {derived}")
    elif txt_det is not None and dmx_det is None:
        derived = _derive_neighbor_box(txt_det["box"])
        dmx_det = {"cls": 0, "box": derived, "conf": txt_det["conf"], "derived": True}
        logger.info(f"[2CLASS] DataMatrix-Box aus Text-Box abgeleitet: {derived}")
    return dmx_det, txt_det


def _scan_crops_parallel(dmx_crop: np.ndarray, txt_crop: np.ndarray, text_crop: bool = False) -> tuple[dict, dict]:
    """DataMatrix- (nur mit Legacy-Kaskade) und OCR-Auswertung der Crops parallel; Ausnahmen liefern Leer-Ergebnisse."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        dmx_future = executor.submit(scan_datamatrix, dmx_crop) if config.USE_LEGACY_DMX_CASCADE else None
        ocr_future = executor.submit(scan_ocr, txt_crop, text_crop)
        try:
            dmx_result = dmx_future.result() if dmx_future else dmx_blocked("Legacy-DMX-Kaskade deaktiviert")
        except Exception as e:
            logger.warning(f"[2CLASS] DataMatrix-Scan Fehler: {e}")
            dmx_result = dmx_blocked("DMX Fehler")
        try:
            ocr_result = ocr_future.result()
        except Exception as e:
            logger.warning(f"[2CLASS] OCR-Scan Fehler: {e}")
            ocr_result = ocr_failed()
    return dmx_result, ocr_result


def _fullframe_fallback(result: dict, frame: np.ndarray, deadline: float | None = None) -> dict:
    """Crop-Auswertung ohne Erfolg → scan() auf dem Gesamtbild; nur verifizierte oder sichere Ergebnisse zählen."""
    if _time_left(deadline) <= 0:
        logger.info("[2CLASS] Zeitbudget erschöpft – kein Gesamtbild-Fallback.")
        return result
    logger.info("[2CLASS] 2-Klassen-Crop ohne Erfolg. Starte Fallback auf scan().")
    fallback_res = scan(frame, try_dotpeen=False, deadline=deadline)
    if fallback_res.get("success"):
        fb_conf = fallback_res.get("confidence", 0)
        fb_method = fallback_res.get("method", "")
        if fallback_res.get("verified", False) or fb_conf >= _MIN_FALLBACK_CONF or fb_method == "Verifiziert":
            return fallback_res
        logger.warning(f"[2CLASS] scan()-Fallback '{fallback_res.get('result')}' hat niedrige Konfidenz "
                       f"({fb_conf:.2f}, method={fb_method}). Nicht akzeptiert.")
    return result


def _is_digit_confusion(code_a: str, code_b: str) -> bool:
    """Genau eine abweichende Stelle, und die ist ein bekanntes OCR-Verwechslungspaar (z. B. 0↔8)."""
    diff_positions = [i for i in range(4) if code_a[i] != code_b[i]]
    return len(diff_positions) == 1 and (code_a[diff_positions[0]], code_b[diff_positions[0]]) in _DIGIT_CONFUSION_PAIRS


def _cross_validate_with_fullframe(result: dict, frame: np.ndarray, deadline: float | None = None) -> dict:
    """
    Gegenprobe eines unverifizierten 2-Klassen-Ergebnisses mit scan() auf dem Gesamtbild. Läuft aus
    Sicherheitsgründen auch nach Ablauf des Zeitbudgets; scan() lässt dann nur die optionalen Stufen weg.
    """
    logger.info(
        f"[2CLASS-CROSSVAL] Ergebnis '{result['result']}' nicht verifiziert "
        f"(method={result['method']}, conf={result.get('confidence', 0):.2f}). "
        f"Starte Gegenprobe mit scan() auf Gesamtbild..."
    )
    crossval_res = scan(frame, try_dotpeen=False, deadline=deadline, mandatory_ocr=True)

    if not crossval_res.get("success"):
        # Nur ein sicheres Ergebnis, das keine blinde Rekonstruktion ist, wird ohne Bestätigung akzeptiert
        if result.get("confidence", 0) >= 0.95 and result.get("method") != "Rekonstruiert":
            logger.info(f"[2CLASS-CROSSVAL] scan() fehlgeschlagen, aber 2class hat hohe Konfidenz "
                        f"({result.get('confidence', 0):.2f}). Akzeptiere '{result['result']}'.")
            return result
        logger.warning(f"[2CLASS-CROSSVAL] scan() fehlgeschlagen und 2class-Ergebnis unverifiziert/Rekonstruktion "
                       f"({result.get('method')}, conf={result.get('confidence', 0):.2f}). Melde Fehler.")
        return scan_result(False, "Unsicheres Ergebnis.", "Fehler", 0.0, dmtx=result.get("dmtx_result"),
                           ocr=result.get("ocr_result"), partial=result.get("ocr_partial_display"))

    code_2class = result.get("result")
    code_fullframe = crossval_res.get("result")

    if code_2class == code_fullframe:
        best_conf = max(result.get("confidence", 0), crossval_res.get("confidence", 0))
        logger.info(f"[2CLASS-CROSSVAL] Übereinstimmung! Beide Pipelines: '{code_2class}' (Konfidenz={best_conf:.2f})")
        crossval_res["confidence"] = best_conf
        return crossval_res

    crossval_verified = crossval_res.get("verified", False) or (
        crossval_res.get("method") == "Verifiziert" and crossval_res.get("confidence", 0) >= 0.98
    )
    if crossval_verified:
        logger.info(f"[2CLASS-CROSSVAL] Widerspruch: 2class='{code_2class}' vs scan()='{code_fullframe}'. "
                    f"scan() ist verifiziert → bevorzuge '{code_fullframe}'.")
        return crossval_res

    # Weichen die Codes nur in einer typisch verwechselten Ziffer ab, gewinnt die höhere Konfidenz
    if code_2class and code_fullframe and len(code_2class) == 4 and len(code_fullframe) == 4 \
            and _is_digit_confusion(code_2class, code_fullframe):
        conf_2class = result.get("confidence", 0)
        conf_scan = crossval_res.get("confidence", 0)
        chosen = crossval_res if conf_scan >= conf_2class else result
        logger.info(f"[2CLASS-CROSSVAL] Digit-Confusion Soft-Match: '{code_2class}' vs '{code_fullframe}'. "
                    f"Bevorzuge '{chosen.get('result')}' (Conf 2class={conf_2class:.2f}, scan={conf_scan:.2f}).")
        return chosen

    logger.warning(f"[2CLASS-CROSSVAL] Widerspruch ohne Verifikation: 2class='{code_2class}' "
                   f"vs scan()='{code_fullframe}'. Melde Fehler.")
    return scan_result(False, f"Widerspruch: {code_2class} vs {code_fullframe}", "Fehler", 0.0,
                       dmtx=result.get("dmtx_result"), ocr=crossval_res.get("ocr_result"),
                       partial=crossval_res.get("ocr_partial_display"))


def scan_2class(frame: np.ndarray, detections: list[dict], cancellation_check=None, deadline: float | None = None,
                run_fast: bool = True, module_hint: dict | None = None) -> dict:
    """
    2-Klassen-Pipeline für YOLO-Detektionen (Klasse 0 = DataMatrix, Klasse 1 = Text).

    1. Schnelle DataMatrix-Stufen (scan_fast: zxing, Modul-Decoder) – ein Decode ist endgültig.
    2. Text-Crop per OCR auswerten (optional Legacy-DMX-Kaskade) und fusionieren. Hat der Modul-Decoder den
       Symbolrahmen gefunden, kommt der Text-Crop entzerrt aus dessen Geometrie; nennen OCR und Codebuch-Abgleich
       denselben Code, gilt er als verifiziert.
    3. Unverifizierte Ergebnisse per Gegenprobe mit scan() auf dem Gesamtbild absichern.

    Args:
        frame: Kamerabild (BGR).
        detections: [{"cls": int, "conf": float, "box": (x1, y1, x2, y2)}, ...]
        cancellation_check: Optionaler Callback; True bricht den Scan ab (neuer Trigger).
        deadline: time.perf_counter()-Zeitpunkt, ab dem keine optionalen Stufen mehr starten
                  (Standard: jetzt + config.SCAN_TIME_BUDGET_S).
        run_fast: False, wenn der Aufrufer scan_fast() bereits ausgeführt hat (Mehrbild-Auswertung).
        module_hint: hints["module"] aus dem scan_fast()-Aufruf des Aufrufers (bei run_fast=False).
    """
    if cancellation_check and cancellation_check():
        logger.info("[2CLASS] Scan wurde vor Start abgebrochen/verworfen (neuer Trigger).")
        return aborted_result()
    if frame is None:
        return error_result("Kein Bild vorhanden.")
    if deadline is None:
        deadline = new_deadline()

    dmx_det, txt_det = select_label_detections(detections)
    dmx_info = f"DMX=Ja(conf={dmx_det['conf']:.2f})" if dmx_det else "DMX=Nein"
    txt_info = f"TXT=Ja(conf={txt_det['conf']:.2f})" if txt_det else "TXT=Nein"
    logger.info(f"[2CLASS] Detections: {dmx_info}, {txt_info}")
    detections_info = {
        "dmx_box": dmx_det["box"] if dmx_det else None,
        "dmx_conf": dmx_det["conf"] if dmx_det else 0.0,
        "txt_box": txt_det["box"] if txt_det else None,
        "txt_conf": txt_det["conf"] if txt_det else 0.0,
    }

    def finish(result: dict, timing: dict) -> dict:
        result["_internal_timing"] = timing
        result["_2class_mode"] = True
        result["_detections"] = detections_info
        return result

    # DataMatrix zuerst: ein echter Decode ist endgültig, OCR und Gegenprobe entfallen dann.
    if run_fast:
        t_dmx_start = time.time()
        hints = {}
        fast = scan_fast(frame, detections, deadline, hints=hints)
        if fast is not None:
            dmx_ms = _elapsed_ms(t_dmx_start)
            logger.info(f"[2CLASS] {fast['method_detail']}: '{fast['result']}' ({dmx_ms}ms). OCR übersprungen.")
            return finish(fast, {"total_2class_ms": dmx_ms, "dmtx_ms": dmx_ms, "ocr_ms": 0})
        module_hint = hints.get("module")
    hint = _usable_module_hint(module_hint)

    if cancellation_check and cancellation_check():
        return aborted_result()
    if dmx_det is None and txt_det is None and hint is None:
        logger.info("[2CLASS] Keine YOLO-Detections. Starte Fallback auf scan().")
        return scan(frame, cancellation_check, try_dotpeen=False, deadline=deadline)
    if _time_left(deadline) <= 0:
        logger.info("[2CLASS] Zeitbudget nach den DataMatrix-Stufen erschöpft – keine OCR-Auswertung.")
        return finish(scan_result(False, "Kein Code erkannt.", "Fehler", 0.0), {"total_2class_ms": 0})

    dmx_crop = deskew_crop(frame, dmx_det["box"], padding=40) if dmx_det is not None else frame
    logger.info(f"[2CLASS] DataMatrix-Crop: {dmx_crop.shape[1]}x{dmx_crop.shape[0]}")
    if hint is not None:
        txt_crop = text_crop_from_quad(frame, hint["quad"])
        logger.info(f"[2CLASS] Text-Crop entzerrt aus der Symbolgeometrie (Kandidat '{hint['best']}', "
                    f"NCC {hint['ncc']:.2f}, Marge {hint['margin']:.2f}).")
    else:
        txt_crop = deskew_crop(frame, txt_det["box"], padding=30)
    logger.info(f"[2CLASS] Text-Crop: {txt_crop.shape[1]}x{txt_crop.shape[0]}")

    t_start = time.time()
    dmx_result, ocr_result = _scan_crops_parallel(dmx_crop, txt_crop, text_crop=hint is not None)

    if cancellation_check and cancellation_check():
        logger.info("[2CLASS] Scan während der Auswertung durch neuen Trigger storniert!")
        return aborted_result()

    timing = {"total_2class_ms": _elapsed_ms(t_start)}

    if hint is not None:
        confirmed = _ocr_confirmed_module_result(hint, ocr_result)
        if confirmed is not None:
            logger.info(f"[2CLASS] {confirmed['method_detail']}: '{confirmed['result']}'.")
            return finish(confirmed, timing)

    # Ein echter DMX-Decode im Crop (Reed-Solomon-geprüft) ist endgültig; OCR nur noch als Info.
    crop_code = dmx_text_to_code(dmx_result.get("text")) if dmx_result.get("status") == "decoded" else None
    if crop_code:
        logger.info(f"[2CLASS] DataMatrix im Crop dekodiert: '{crop_code}' (OCR: '{ocr_result.get('text')}'). Endgültig.")
        return finish(dmx_final_result(crop_code, dmx_result.get("method_detail", "DataMatrix dekodiert"),
                                       ocr_result.get("text")), timing)

    result = finish(merge_results(ocr_result, dmx_result, dmx_crop), timing)
    logger.info(f"[2CLASS] Ergebnis: success={result['success']}, method={result['method']}, "
                f"result='{result['result']}' ({timing['total_2class_ms']}ms)")

    if not result.get("success"):
        return _fullframe_fallback(result, frame, deadline)

    if result.get("verified", False) and result.get("confidence", 0) >= 1.0:
        logger.info(f"[2CLASS] Ergebnis doppelt verifiziert. Akzeptiere '{result['result']}'.")
        return result

    return _cross_validate_with_fullframe(result, frame, deadline)


# --------------------------------------------------------------------------- #
#  Gesamtbild-Pipeline                                                         #
# --------------------------------------------------------------------------- #

def _gamma_fallback(frame: np.ndarray, deadline: float | None = None) -> dict | None:
    """
    OCR auf mehrfach aufgehellten Bildern mit Mehrheitsentscheid. Bei dunklen Bildern liest OCR
    systematisch 6 statt 0 (W031 → W631), daher verliert eine 6-Variante gegen ihre gefundene 0-Variante.
    """
    logger.info("[GAMMA-FALLBACK] Normaler Scan fehlgeschlagen. Versuche Gamma-Korrektur...")
    gray = to_gray(frame)
    readings = []
    for gamma in _GAMMA_FALLBACK_VALUES:
        if _time_left(deadline) <= 0:
            logger.info(f"[GAMMA-FALLBACK] Zeitbudget erschöpft vor gamma_{gamma}.")
            break
        ocr_gamma = read_ocr_with_status(cv2.LUT(gray, gamma_lut(gamma)))
        if ocr_gamma.get("status") == "ok" and ocr_gamma.get("text"):
            code = normalize_ocr_confusions(ocr_gamma["text"])
            conf = ocr_gamma.get("confidence", 0.0)
            if is_valid_horden_code(code):
                readings.append((code, conf))
                logger.info(f"[GAMMA-FALLBACK] gamma_{gamma}: '{code}' (Conf={conf:.2f})")
    if not readings:
        return None

    votes = Counter()
    max_conf = {}
    for code, conf in readings:
        votes[code] += 1
        if code not in max_conf or conf > max_conf[code]:
            max_conf[code] = conf

    best_code = None
    best_score = 0
    for code, n_votes in votes.items():
        score = n_votes * 10 + max_conf[code]
        for i in range(1, 4):
            if code[i] == '6' and code[:i] + '0' + code[i + 1:] in votes:
                score -= 5
        if score > best_score:
            best_score = score
            best_code = code
    if best_code is None:
        return None

    logger.info(f"[GAMMA-FALLBACK] Voting: '{best_code}' (Votes={votes[best_code]}, "
                f"MaxConf={max_conf[best_code]:.2f}, Score={best_score:.1f})")
    # Reines OCR ohne DataMatrix-Bestätigung → nie verifiziert
    return scan_result(True, best_code, "OCR", max_conf[best_code], ocr=best_code, partial=best_code)


def scan(frame: np.ndarray, cancellation_check=None, try_dotpeen: bool = True, deadline: float | None = None,
         mandatory_ocr: bool = False) -> dict:
    """
    Gesamtbild-Scan: schnelle DataMatrix-Stufen (scan_fast), danach OCR (optional Legacy-DMX-Kaskade und
    Referenzbild-Abgleich) mit Fusion sowie Gamma-Fallback für dunkle Bilder im Zeitbudget.

    Args:
        frame: Kamerabild oder Etikett-Ausschnitt (BGR oder Graustufen).
        cancellation_check: Optionaler Callback; True bricht den Scan ab (neuer Trigger).
        try_dotpeen: schnelle DataMatrix-Stufen vorschalten (scan_2class hat sie bereits ausgeführt).
        deadline: time.perf_counter()-Zeitpunkt, ab dem keine optionalen Stufen mehr starten
                  (Standard: jetzt + config.SCAN_TIME_BUDGET_S).
        mandatory_ocr: OCR auch nach Ablauf des Budgets (Gegenprobe eines unverifizierten Ergebnisses).
    """
    if frame is None:
        return error_result("Kein Bild vorhanden.")
    if deadline is None:
        deadline = new_deadline()

    h, w = frame.shape[:2]
    logger.info(f"Triple-Validation Scan v5.0 gestartet auf Bild mit {w}x{h} Pixeln.")

    if cancellation_check and cancellation_check():
        return aborted_result()

    if try_dotpeen:
        t_dot = time.time()
        fast = scan_fast(frame, (), deadline)
        if fast is not None:
            dot_ms = _elapsed_ms(t_dot)
            logger.info(f"[FAST-PATH] {fast['method_detail']}: '{fast['result']}' ({dot_ms}ms). Skippe OCR.")
            fast["_internal_timing"] = {"ocr_ms": 0, "dmtx_ms": dot_ms, "refimg_ms": 0}
            return fast

    t_dmx = time.time()
    dmx_result = dmx_blocked("Legacy-DMX-Kaskade deaktiviert")
    if config.USE_LEGACY_DMX_CASCADE and _time_left(deadline) > 0:
        try:
            dmx_result = scan_datamatrix_pipeline(frame)
        except Exception as e:
            logger.warning(f"DataMatrix Pipeline Exception: {e}")
            dmx_result = dmx_blocked("DMX Pipeline Fehler")
    dmx_ms = _elapsed_ms(t_dmx)

    if dmx_result.get("status") == "decoded" and dmx_result.get("text") and is_valid_horden_code(dmx_result["text"]):
        code = dmx_result["text"]
        logger.info(f"[FAST-PATH] DataMatrix direkt erkannt '{code}'. Skippe OCR.")
        result = scan_result(True, code, "Verifiziert", 1.0, dmtx=code, ocr=code, verified=True, partial=code)
        result["_internal_timing"] = {"ocr_ms": 0, "dmtx_ms": dmx_ms, "refimg_ms": 0}
        return result

    if config.USE_HORDE_DB_MATCHING:
        horde_result = match_horde_image(frame)
        if horde_result is not None:
            horde_result["_internal_timing"] = {"ocr_ms": 0, "dmtx_ms": dmx_ms, "refimg_ms": 0}
            return horde_result

    if cancellation_check and cancellation_check():
        return aborted_result()

    # OCR und RefImg sequentiell (verhindert PyTorch/OpenCV-Multithreading-Crashes)
    t_ocr = time.time()
    ocr_result = ocr_failed()
    if mandatory_ocr or _time_left(deadline) > 0:
        try:
            ocr_result = read_ocr_with_status(frame)
        except Exception as e:
            logger.warning(f"OCR Fehler: {e}")
    else:
        logger.info("Zeitbudget erschöpft – Gesamtbild-OCR übersprungen.")
    ocr_ms = _elapsed_ms(t_ocr)

    t_refimg = time.time()
    ref_img_result = {"status": "blocked", "text": None, "confidence": 0.0, "method_detail": "RefImg deaktiviert"}
    if config.USE_LEGACY_DMX_CASCADE and _time_left(deadline) > 0:
        try:
            ref_img_result = scan_reference_image_pipeline(frame)
        except Exception as e:
            logger.warning(f"RefImg Pipeline Fehler: {e}")
            ref_img_result = {"status": "blocked", "text": None, "confidence": 0.0, "method_detail": "RefImg Fehler"}
    refimg_ms = _elapsed_ms(t_refimg)

    timing = {"ocr_ms": ocr_ms, "dmtx_ms": dmx_ms, "refimg_ms": refimg_ms}
    result = merge_results(ocr_result, dmx_result, frame, ref_img_result)
    result["_internal_timing"] = timing

    if not result.get("success") and _time_left(deadline) > 0:
        gamma_result = _gamma_fallback(frame, deadline)
        if gamma_result is not None:
            result = gamma_result
            result["_internal_timing"] = timing

    if config.USE_HORDE_DB_MATCHING and result.get("success") and result.get("result") \
            and not result.get("verified", False):
        original_code = result["result"]
        corrected_code = correct_ocr_confusion(original_code, frame)
        if corrected_code != original_code:
            result["result"] = corrected_code
            result["ocr_result"] = corrected_code
            result["ocr_partial_display"] = corrected_code
            result["_ocr_postprocessed"] = True
            result["_ocr_original"] = original_code

    logger.info(f"Scan Ergebnis: success={result['success']}, method={result['method']}, result='{result['result']}'")
    return result
