"""Einheitliche Ergebnis-Dictionaries der Scan-Pipelines."""


def scan_result(success: bool, result: str | None, method: str, confidence: float,
                dmtx: str | None = None, ocr: str | None = None, verified: bool = False,
                partial: str | None = None) -> dict:
    """Endergebnis eines Scans (Schnittstelle zu GUI, TCP-Server, ScanLogger und Benchmark)."""
    return {
        "success": success,
        "result": result,
        "method": method,
        "confidence": confidence,
        "dmtx_result": dmtx,
        "ocr_result": ocr,
        "verified": verified,
        "ocr_partial_display": partial,
    }


def error_result(message: str) -> dict:
    return scan_result(False, message, "Fehler", 0.0)


def aborted_result() -> dict:
    """Scan wurde durch einen neueren Trigger storniert."""
    return {"success": False, "result": "ABORTED", "method": "Abgebrochen", "confidence": 0.0, "cancelled": True}


def dmx_final_result(code: str, method_detail: str, ocr_text: str | None = None) -> dict:
    """Endergebnis für einen echten DataMatrix-Decode – wird nicht mehr durch OCR oder Gegenprobe überstimmt."""
    result = scan_result(True, code, "Verifiziert", 1.0, dmtx=code, ocr=ocr_text, verified=True, partial=ocr_text)
    result["method_detail"] = method_detail
    return result


# --- Zwischenergebnisse der Teil-Pipelines ---

def ocr_ok(text: str, confidence: float) -> dict:
    return {
        "status": "ok",
        "text": text,
        "partial_display": text,
        "readable_chars": text,
        "confidence": confidence,
        "readable_count": 4,
        "missing_positions": [],
        "raw_candidate": text,
    }


def ocr_failed() -> dict:
    return {
        "status": "failed",
        "text": None,
        "partial_display": None,
        "readable_chars": None,
        "confidence": 0.0,
        "readable_count": 0,
        "missing_positions": [],
        "raw_candidate": None,
    }


def dmx_blocked(method_detail: str, observed_grid=None) -> dict:
    return {
        "status": "blocked",
        "text": None,
        "method_detail": method_detail,
        "confidence": 0.0,
        "observed_grid": observed_grid,
    }
