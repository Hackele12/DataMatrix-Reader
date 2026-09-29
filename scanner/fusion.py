"""
Ergebnis-Fusion: führt OCR-, DataMatrix- und Referenzbild-Ergebnisse zusammen und
rekonstruiert Codes aus Teillesungen über den Abgleich mit Referenzgittern.
"""

import logging

import numpy as np

from . import config
from .code_format import (clean_to_4chars, format_partial_display, generate_10_candidates_from_partial,
                          is_valid_horden_code, normalize_ocr_confusions, normalize_partial_3chars)
from .config import REQUIRED_LENGTH, VALID_PREFIXES
from .dmx_codec import all_code_grids, cached_reference_grid
from .dmx_grid import (ALL_GRID_METHODS, cross_validate_ocr_dmtx, extract_observed_grid, extract_observed_grid_soft,
                       match_soft_grid_matrix, template_match_candidates)
from .results import scan_result

logger = logging.getLogger(__name__)

# Binarisierungen für die Cross-Validation einer OCR-Teillesung (ohne "mean")
_PARTIAL_CV_METHODS = ("otsu", "adaptive", "clahe_otsu", "etch_denoise", "ridge_boost",
                       "sauvola_w11", "sauvola_w21", "niblack")


def is_dmx_consistent_with_ocr(dmx_text: str, ocr_text: str | None, ocr_readable: str | None) -> bool:
    """
    Prüft, ob ein rekonstruierter DataMatrix-Text zur OCR-Lesung passt (max. 1 abweichendes Zeichen
    bzw. Teillesung als Teilfolge). Ohne OCR-Information gilt er als konsistent.
    """
    if not dmx_text:
        return False

    dmx_norm = normalize_ocr_confusions(dmx_text)
    if ocr_text:
        ocr_norm = normalize_ocr_confusions(ocr_text)
        return sum(1 for c1, c2 in zip(dmx_norm, ocr_norm) if c1 != c2) <= 1

    if ocr_readable and len(ocr_readable) == 3:
        partial_norm, _ = normalize_partial_3chars(ocr_readable)
        return any(dmx_norm[:skip] + dmx_norm[skip + 1:] == partial_norm for skip in range(4))

    return True


# --------------------------------------------------------------------------- #
#  Gitter-Rekonstruktion (Schalter config.USE_GRID_RECONSTRUCTION)            #
# --------------------------------------------------------------------------- #

def _reconstruction_candidates(ocr_text: str | None, ocr_partial: str | None,
                               missing_positions: list[int] | None) -> set[str]:
    """Formatgültige Kandidaten aus OCR-Text (±1 Stelle), Teillesung oder – ohne OCR – alle 4.000 Codes."""
    candidates = set()
    if ocr_text and len(ocr_text) == REQUIRED_LENGTH:
        normalized = normalize_ocr_confusions(ocr_text)
        if is_valid_horden_code(normalized):
            candidates.add(normalized)
        for pos in range(REQUIRED_LENGTH):
            fillers = VALID_PREFIXES if pos == 0 else '0123456789'
            for filler in fillers:
                candidate = normalized[:pos] + filler + normalized[pos + 1:]
                if is_valid_horden_code(candidate):
                    candidates.add(candidate)
    elif ocr_partial and len(ocr_partial) == 3:
        partial_norm, prefix_detected = normalize_partial_3chars(ocr_partial)
        if prefix_detected:
            # Präfix bekannt → fehlende Ziffer an Stelle 1, 2 oder 3 einfügen (missing_positions sind 4-stellige Indizes)
            prefix, digits = partial_norm[0], partial_norm[1:]
            insert_positions = [pos - 1 for pos in (missing_positions or []) if 1 <= pos <= 3] or range(3)
            for insert_pos in insert_positions:
                for d in '0123456789':
                    candidate = prefix + digits[:insert_pos] + d + digits[insert_pos:]
                    if is_valid_horden_code(candidate):
                        candidates.add(candidate)
            logger.info(
                f"Rekonstruktion: Partial '{ocr_partial}' → normalisiert '{partial_norm}' "
                f"(Präfix '{prefix}' erkannt). {len(candidates)} Kandidaten erzeugt."
            )
        else:
            for prefix in VALID_PREFIXES:
                candidate = prefix + partial_norm
                if is_valid_horden_code(candidate):
                    candidates.add(candidate)
    else:
        logger.info("Rekonstruktion: Keine OCR-Daten. Erzeuge alle 4000 formatgültigen Kandidaten...")
        for prefix in VALID_PREFIXES:
            for num in range(1000):
                candidates.add(f"{prefix}{num:03d}")
    return candidates


def _is_reconstruction_accepted(best_candidate: str, score: float, margin: float, n_candidates: int,
                                ocr_text: str | None, ocr_conf: float, ocr_partial: str | None) -> bool:
    """Gestufte Qualitätsschwellen für eine Gitter-Rekonstruktion."""
    if score >= 0.80 and margin >= 0.04:
        return True
    if score >= 0.65 and margin >= 0.10:
        logger.info("Rekonstruktion: Akzeptiert über Stufe 2 (Teilverdeckung).")
        return True
    if ocr_text:
        normalized_ocr = normalize_ocr_confusions(ocr_text)
        if best_candidate == normalized_ocr and ocr_conf >= 0.40 and score >= 0.60:
            logger.info(f"Rekonstruktion: Akzeptiert über Stufe 3 (OCR-Übereinstimmung mit '{best_candidate}', "
                        f"Score={score:.1%}, OCR-Konfidenz={ocr_conf:.2f}).")
            return True
        if n_candidates <= 35 and score >= 0.60 and best_candidate == normalized_ocr:
            logger.info(f"Rekonstruktion: Akzeptiert über Stufe 4 (wenige Kandidaten, "
                        f"OCR-Match '{best_candidate}', Score={score:.1%}).")
            return True
    if ocr_partial and score >= 0.65 and margin >= 0.05:
        clean_partial = ocr_partial.replace("?", "").strip().upper()
        if len(clean_partial) >= 2 and best_candidate.startswith(clean_partial):
            logger.info(f"Rekonstruktion: Akzeptiert über Stufe 5 (Partial-Match '{ocr_partial}' -> "
                        f"'{best_candidate}', Score={score:.1%}, Margin={margin:.1%}).")
            return True
    return False


def try_reconstruct(frame: np.ndarray, ocr_text: str | None, ocr_conf: float, ocr_partial: str | None,
                    missing_positions: list[int] | None = None) -> dict | None:
    """
    Rekonstruiert den Code durch Abgleich beobachteter Gitter (alle Binarisierungen + Soft-Matching)
    mit den Referenzgittern der formatgültigen Kandidaten.

    Returns:
        Scan-Ergebnis bei Erfolg, {"success": False, "grid_detected": bool} bei Ablehnung,
        None wenn abgeschaltet oder kein Kandidat gewertet werden konnte.
    """
    if not config.USE_GRID_RECONSTRUCTION:
        return None

    candidates = _reconstruction_candidates(ocr_text, ocr_partial, missing_positions)
    if not candidates:
        return {"success": False, "grid_detected": False}

    observed_variants = {}
    for method in ALL_GRID_METHODS:
        grid_obs = extract_observed_grid(frame, binarization_method=method, strict_l_finder=False)
        if grid_obs is not None:
            observed_variants[method] = grid_obs
    if not observed_variants:
        logger.debug("Rekonstruktion: Kein 10x10 Grid im Bild unter irgendeiner Binarisierung gefunden.")
        return {"success": False, "grid_detected": False}

    logger.info(f"Rekonstruktion: Teste {len(candidates)} Kandidaten gegen {len(observed_variants)} Grid-Varianten...")

    best_candidate = None
    best_score = -1.0
    best_margin = -1.0
    best_method = None
    all_codes, all_matrix = all_code_grids()

    for method, observed in observed_variants.items():
        matching_bits = np.sum(all_matrix == observed.flatten(), axis=1)
        if len(candidates) < 4000:
            indices = [all_codes.index(c) for c in candidates if c in all_codes]
            if not indices:
                continue
            cand_indices = np.array(indices)
            sub_matches = matching_bits[cand_indices]
            sorted_indices = np.argsort(sub_matches)[::-1]
            method_code = all_codes[cand_indices[sorted_indices[0]]]
            method_score = sub_matches[sorted_indices[0]] / 100.0
            second_score = (sub_matches[sorted_indices[1]] / 100.0) if len(sorted_indices) > 1 else 0.0
            method_margin = method_score - second_score
        else:
            top2 = np.argpartition(matching_bits, -2)[-2:]
            best_idx, second_idx = top2[np.argsort(matching_bits[top2])[::-1]]
            method_code = all_codes[best_idx]
            method_score = matching_bits[best_idx] / 100.0
            method_margin = (matching_bits[best_idx] - matching_bits[second_idx]) / 100.0

        logger.debug(f"Rekonstruktion ({method}): Bester='{method_code}' Score={method_score:.1%}, "
                     f"Abstand={method_margin:.1%}")
        if method_score > best_score:
            best_score = method_score
            best_margin = method_margin
            best_candidate = method_code
            best_method = method

    soft_grid = extract_observed_grid_soft(frame)
    if soft_grid is not None:
        soft_cand, soft_score, soft_margin = match_soft_grid_matrix(soft_grid, candidates)
        if soft_cand is not None:
            logger.info(f"Rekonstruktion (Soft-Matching): Bester='{soft_cand}' Score={soft_score:.1%}, "
                        f"Abstand={soft_margin:.1%}")
            if soft_score > best_score:
                best_score = soft_score
                best_margin = soft_margin
                best_candidate = soft_cand
                best_method = "Soft-Matching"

    if best_candidate is None:
        return None

    logger.info(f"Rekonstruktion: Bester='{best_candidate}' Score={best_score:.1%} (Methode: {best_method}), "
                f"Abstand={best_margin:.1%}")

    if _is_reconstruction_accepted(best_candidate, best_score, best_margin, len(candidates),
                                   ocr_text, ocr_conf, ocr_partial):
        logger.info(f"[OK] REKONSTRUIERT: '{best_candidate}' (Score={best_score:.1%}, Abstand={best_margin:.1%})")
        display = ocr_text or ocr_partial
        return scan_result(True, best_candidate, "Rekonstruiert", min(1.0, max(0.98, best_score)),
                           ocr=display, partial=display)

    logger.info(f"Rekonstruktion ABGELEHNT: Score={best_score:.1%}, Abstand={best_margin:.1%}")
    return {"success": False, "grid_detected": True}


# --------------------------------------------------------------------------- #
#  Zusammenführung                                                             #
# --------------------------------------------------------------------------- #

def _merge_dmx_success(dmx_status: str, dmx_text: str, dmx_result: dict, ocr_text: str | None,
                       ocr_partial_display: str | None, ocr_raw_candidate: str | None,
                       ref_text: str | None) -> dict:
    """Fall 1: DataMatrix dekodiert oder Rahmen-rekonstruiert – OCR und Referenzbild dienen der Verifikation."""
    method = "Verifiziert" if dmx_status == "decoded" else "Rekonstruiert"
    confidence = dmx_result.get("confidence", 0.0)
    is_verified = False

    ocr_check_text = ocr_text or ocr_raw_candidate
    if ocr_check_text and dmx_text:
        dmtx_norm = normalize_ocr_confusions(dmx_text.strip())
        ocr_norm = normalize_ocr_confusions(ocr_check_text.strip())

        if dmtx_norm == ocr_norm or dmtx_norm in ocr_norm or ocr_norm in dmtx_norm:
            logger.info(f"[OK] VERIFIZIERT: DMX '{dmx_text}' ≈ OCR '{ocr_check_text}' "
                        f"(normalisiert: '{dmtx_norm}' == '{ocr_norm}')")
            method = "Verifiziert"
            confidence = 1.0
            is_verified = True
        elif ref_text and ref_text == dmtx_norm:
            logger.info(f"[REFIMG-TIEBREAK] RefImg bestätigt DMX '{dmx_text}' gegen OCR '{ocr_check_text}'")
            confidence = 0.98
        elif ref_text and ref_text == ocr_norm:
            logger.info(f"[REFIMG-TIEBREAK] RefImg bestätigt OCR '{ocr_check_text}' gegen DMX '{dmx_text}'")
            # OCR + Referenzbild überstimmen die DataMatrix (unverifiziert)
            return scan_result(True, ocr_check_text, "OCR+RefImg-Tiebreak", 0.85, dmtx=dmx_text,
                               ocr=ocr_check_text, partial=ocr_check_text)
        else:
            logger.warning(f"[WARN] ABWEICHUNG: DMX='{dmx_text}' vs OCR='{ocr_check_text}'. "
                           f"Nutze DMX ({dmx_result.get('method_detail')}).")
            confidence = 0.9

    if is_verified and ref_text and ref_text == dmx_text:
        logger.info(f"[TRIPLE-MATCH] Alle 3 Pipelines bestätigen: '{dmx_text}'")
        confidence = 1.0

    ocr_display = ocr_check_text if is_verified else (ocr_text or ocr_partial_display)
    return scan_result(True, dmx_text, method, confidence, dmtx=dmx_text, ocr=ocr_display,
                       verified=is_verified, partial=ocr_display)


def _merge_ocr_ok(ocr_text: str, ocr_conf: float, dmx_result: dict, frame: np.ndarray) -> dict | None:
    """Fall 2: DataMatrix blockiert, OCR vollständig – Rekonstruktion oder OCR mit ausreichender Konfidenz."""
    observed_grid = dmx_result.get("observed_grid")
    if observed_grid is not None:
        recon_result = try_reconstruct(frame, ocr_text, ocr_conf, None)
        if recon_result is not None and recon_result.get("success"):
            # Kandidaten stammen aus der OCR-Lesung → Übereinstimmung ist keine unabhängige Verifikation.
            recon_result["ocr_partial_display"] = ocr_text
            return recon_result

    if ocr_conf >= 0.98:
        logger.info(f"OCR-Direkt (≥0.98 Konfidenz): '{ocr_text}' (Conf={ocr_conf:.2f})")
        return scan_result(True, ocr_text, "OCR", ocr_conf, ocr=ocr_text, partial=ocr_text)

    if observed_grid is None:
        # Kein Gitter extrahierbar → OCR ist einzige Quelle, akzeptiert ab 0.40
        if ocr_conf >= 0.40 and is_valid_horden_code(ocr_text):
            logger.info(f"OCR-Fallback (kein Grid, gültiges Format): '{ocr_text}' (Conf={ocr_conf:.2f})")
            return scan_result(True, ocr_text, "OCR", ocr_conf, ocr=ocr_text, partial=ocr_text)
    elif ocr_conf >= 0.75 and is_valid_horden_code(ocr_text):
        logger.info(f"OCR-Fallback (Format-validiert ≥0.75, Rekonstruktion fehlgeschlagen): '{ocr_text}' "
                    f"(Conf={ocr_conf:.2f})")
        return scan_result(True, ocr_text, "OCR", ocr_conf, ocr=ocr_text, partial=ocr_text)

    logger.warning(f"OCR fand '{ocr_text}' (Conf: {ocr_conf:.2f}), aber Rekonstruktion fehlgeschlagen "
                   f"und Konfidenz zu niedrig für direktes OCR-Fallback.")
    return None


def _match_partial_against_grids(partial: str, candidates: list[str], dmx_result: dict,
                                 frame: np.ndarray) -> tuple[str, float] | None:
    """Strategien A-D für eine Teillesung → (Code, Konfidenz) oder None."""
    # A: Cross-Validation mit dem Gitter der DMX-Pipeline
    observed_grid = dmx_result.get("observed_grid")
    if observed_grid is not None and observed_grid.shape == (10, 10):
        cv_code, cv_score = cross_validate_ocr_dmtx(partial, observed_grid)
        if cv_code is not None:
            logger.info(f"[CROSS-VAL] Partial '{partial}' → Cross-Validation Match '{cv_code}' (Score: {cv_score:.1%})")
            return cv_code, min(1.0, max(0.90, cv_score))

    # B: Soft-Gitter gegen die Kandidaten
    soft_grid = extract_observed_grid_soft(frame)
    if soft_grid is not None:
        soft_cand, soft_score, soft_margin = match_soft_grid_matrix(soft_grid, candidates)
        if soft_cand is not None and soft_score >= 0.55:
            logger.info(f"[SOFT-CV] Partial '{partial}' → Soft-Grid Match '{soft_cand}' "
                        f"(Score: {soft_score:.1%}, Margin: {soft_margin:.1%})")
            return soft_cand, min(1.0, max(0.85, soft_score))

    # C: Gitter aller Binarisierungen gegen die Kandidaten
    for method in _PARTIAL_CV_METHODS:
        grid_obs = extract_observed_grid(frame, binarization_method=method, strict_l_finder=False)
        if grid_obs is not None:
            cv_code, cv_score = cross_validate_ocr_dmtx(partial, grid_obs)
            if cv_code is not None:
                logger.info(f"[CROSS-VAL-{method}] Partial '{partial}' → Match '{cv_code}' (Score: {cv_score:.1%})")
                return cv_code, min(1.0, max(0.88, cv_score))

    # D: Template-Matching synthetischer DMX-Bilder, wenn kein Gitter passt
    if len(candidates) <= 10:
        tmpl_cand, tmpl_score, tmpl_margin = template_match_candidates(frame, candidates)
        if tmpl_cand is not None and tmpl_score >= 0.25 and tmpl_margin >= 0.01:
            logger.info(f"[TEMPLATE-MATCH] Partial '{partial}' → Match '{tmpl_cand}' "
                        f"(Score: {tmpl_score:.3f}, Margin: {tmpl_margin:.3f})")
            return tmpl_cand, min(0.92, max(0.70, tmpl_score))
    return None


def _infer_missing_digit(partial_norm: str, pos: int, observed_grid) -> tuple[str, float] | None:
    """Setzt jede Ziffer an der fehlenden Stelle ein und wählt per Gitter-Abgleich (Score ≥ 0.60, Abstand ≥ 0.04)."""
    if observed_grid is None or not 1 <= pos <= 3:
        return None
    prefix, digits = partial_norm[0], partial_norm[1:]
    cand_scores = []
    for d in '0123456789':
        candidate = prefix + digits[:pos - 1] + d + digits[pos - 1:]
        if is_valid_horden_code(candidate):
            ref = cached_reference_grid(candidate)
            if ref is not None:
                cand_scores.append((candidate, float(np.sum(observed_grid == ref)) / 100.0))
    if not cand_scores:
        return None

    cand_scores.sort(key=lambda x: x[1], reverse=True)
    best, best_score = cand_scores[0]
    margin = best_score - (cand_scores[1][1] if len(cand_scores) > 1 else 0.0)
    if best_score >= 0.60 and margin >= 0.04:
        return best, best_score
    logger.warning(f"Partial-Inferenz verworfen: Bester='{best}' Score={best_score:.2f}, Margin={margin:.2f} zu gering.")
    return None


def _merge_ocr_partial(ocr_result: dict, dmx_result: dict, frame: np.ndarray) -> dict:
    """Fall 3: OCR-Teillesung (3 Zeichen) – Rekonstruktion, Gitter-Abgleich oder Inferenz der fehlenden Stelle."""
    ocr_readable = ocr_result.get("readable_chars")
    ocr_partial_display = ocr_result.get("partial_display")
    missing_positions = ocr_result.get("missing_positions", [])

    def reconstructed(code: str, confidence: float, dmtx: str | None) -> dict:
        return scan_result(True, code, "Rekonstruiert", confidence, dmtx=dmtx, ocr=ocr_partial_display,
                           partial=ocr_partial_display)

    recon_result = try_reconstruct(frame, None, 0.0, ocr_readable, ocr_result.get("missing_positions"))
    if recon_result is not None and recon_result.get("success"):
        recon_result["ocr_partial_display"] = ocr_partial_display
        return recon_result

    partial_for_cv = ocr_partial_display or format_partial_display(ocr_readable, missing_positions)
    cv_candidates = generate_10_candidates_from_partial(partial_for_cv)
    if cv_candidates:
        match = _match_partial_against_grids(partial_for_cv, cv_candidates, dmx_result, frame)
        if match is not None:
            return reconstructed(match[0], match[1], dmtx=match[0])

    # Letzte Rückfallebene: den Code aus der Teillesung selbst erschließen
    if len(ocr_readable) == 3:
        partial_norm, prefix_detected = normalize_partial_3chars(ocr_readable)
        if prefix_detected:
            raw_cand = ocr_result.get("raw_candidate")
            if raw_cand and len(raw_cand) == 4:
                inferred = clean_to_4chars(raw_cand)
                if inferred is not None:
                    logger.info(f"Partial-Inferenz: '{ocr_readable}' + raw='{raw_cand}' → '{inferred}'")
                    return reconstructed(inferred, 0.85, dmtx=None)

            if len(missing_positions) == 1:
                inferred = _infer_missing_digit(partial_norm, missing_positions[0], dmx_result.get("observed_grid"))
                if inferred is not None:
                    code, score = inferred
                    logger.info(f"Partial-Inferenz (fehlende Pos {missing_positions[0]}): "
                                f"'{ocr_partial_display}' → '{code}' (Score={score:.2f})")
                    return reconstructed(code, min(1.0, max(0.98, score)), dmtx=None)

    logger.warning(f"Rekonstruktion mit OCR-Partial '{ocr_partial_display}' fehlgeschlagen.")
    return scan_result(False, f"Teilweise erkannt: {ocr_partial_display}", "Fehler", 0.0,
                       ocr=ocr_partial_display, partial=ocr_partial_display)


def merge_results(ocr_result: dict, dmx_result: dict, frame: np.ndarray, ref_img_result: dict = None) -> dict:
    """
    Führt OCR-, DataMatrix- und Referenzbild-Ergebnis (Pipeline 3) zum Endergebnis zusammen.

    Args:
        ocr_result: Ergebnis von ocr.read_ocr_with_status().
        dmx_result: Ergebnis der DataMatrix-Pipeline.
        frame: Bild für Rekonstruktionen.
        ref_img_result: Ergebnis von ref_images.scan_reference_image_pipeline() (optional).
    """
    ocr_status = ocr_result["status"]
    ocr_text = ocr_result.get("text")
    ocr_partial_display = ocr_result.get("partial_display")
    ocr_readable = ocr_result.get("readable_chars")
    ocr_conf = ocr_result.get("confidence", 0.0)
    ocr_raw_candidate = ocr_result.get("raw_candidate")

    dmx_status = dmx_result["status"]
    dmx_text = dmx_result.get("text")

    ref_img_result = ref_img_result or {}
    ref_text = ref_img_result.get("text")
    ref_conf = ref_img_result.get("confidence", 0.0)
    ref_status = ref_img_result.get("status", "blocked")

    logger.info(f"Merge: OCR={ocr_status}('{ocr_text or ocr_partial_display}') "
                f"+ DMX={dmx_status}('{dmx_text}') + RefImg={ref_status}('{ref_text}')")

    if dmx_status == "reconstructed" and not is_dmx_consistent_with_ocr(
            dmx_text, ocr_text, ocr_readable or ocr_raw_candidate):
        logger.warning(f"[WARN] DMX-Rekonstruktion '{dmx_text}' verworfen, da unvereinbar mit "
                       f"OCR '{ocr_text or ocr_readable or ocr_raw_candidate}'.")
        dmx_status = "blocked"
        dmx_text = None

    if dmx_status in ("decoded", "reconstructed"):
        return _merge_dmx_success(dmx_status, dmx_text, dmx_result, ocr_text, ocr_partial_display,
                                  ocr_raw_candidate, ref_text)

    if ocr_status == "ok" and ocr_text:
        result = _merge_ocr_ok(ocr_text, ocr_conf, dmx_result, frame)
        if result is not None:
            return result

    if ocr_status == "partial" and ocr_readable:
        return _merge_ocr_partial(ocr_result, dmx_result, frame)

    # Fall 4: keine verwertbare Lesung → Gitter-Rekonstruktion gegen alle 4.000 Codes
    logger.info("DMX blockiert + OCR fehlgeschlagen → versuche Gitter-Rekonstruktion mit allen gültigen Codes...")
    recon_result = try_reconstruct(frame, None, 0.0, None)
    if recon_result is not None and recon_result.get("success"):
        recon_result["ocr_partial_display"] = recon_result["result"]
        recon_result["ocr_result"] = recon_result["result"]
        return recon_result

    # Referenzbild als letzte Rettung
    if ref_text and ref_status == "matched" and ref_conf >= 0.75:
        logger.info(f"[REFIMG-RESCUE] DMX + OCR fehlgeschlagen, RefImg liefert '{ref_text}' (Conf={ref_conf:.2f})")
        return scan_result(True, ref_text, "RefImg", min(0.90, ref_conf), partial=ref_text)

    logger.warning("Weder DataMatrix noch OCR noch RefImg konnten etwas lesen.")
    return scan_result(False, "Kein Code erkannt.", "Fehler", 0.0)
