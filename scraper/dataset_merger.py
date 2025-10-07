from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATAFRAMES_DIR = BASE_DIR.parent / "dataframes"
EXPECTED_COLUMNS = ["SearchTerm", "Title", "Description", "Qualifications", "Link"]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unisci tutti i CSV prodotti dallo scraper SimplyHired in un unico dataset."
    )
    parser.add_argument(
        "--pattern",
        default="simplyhired_jobs_*.csv",
        help="Pattern glob per individuare i CSV da unire (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Percorso del file CSV di destinazione. "
            "Se omesso, crea dataframes/simplyhired_jobs_merged_<timestamp>.csv."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra log di debug durante il merge.",
    )
    return parser.parse_args(argv)


def _clean_string_series(series: pd.Series) -> pd.Series:
    """Normalizza stringhe rimuovendo spazi superflui e BOM UTF-8."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )


def _combine_unique_parts(values: pd.Series) -> str:
    """Unisce sottostringhe separate da '|' mantenendo ordine e unicità."""
    seen: list[str] = []
    for value in values.dropna():
        parts = [part.strip() for part in str(value).split("|")]
        for part in parts:
            if part and part not in seen:
                seen.append(part)
    return " | ".join(seen)


def _pick_longest(values: pd.Series) -> str:
    """Seleziona la stringa più informativa (per lunghezza) da una serie."""
    candidates = [
        str(item).strip()
        for item in values.dropna()
        if str(item).strip()
    ]
    if not candidates:
        return ""
    return max(candidates, key=len)


def merge_output_csvs(pattern: str) -> tuple[pd.DataFrame, dict[str, int]]:
    csv_files = sorted(path for path in OUTPUT_DIR.glob(pattern) if path.is_file())
    stats = {
        "files_found": len(csv_files),
        "files_used": 0,
        "rows_loaded": 0,
        "rows_merged": 0,
        "links_unique": 0,
    }

    if not csv_files:
        logging.warning("Nessun file trovato in %s con pattern '%s'.", OUTPUT_DIR, pattern)
        return pd.DataFrame(columns=EXPECTED_COLUMNS), stats

    dataframes: list[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as exc:  # pragma: no cover - log per visibilità CLI
            logging.error("Impossibile leggere %s: %s", csv_path.name, exc)
            continue

        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            logging.error(
                "Il file %s non contiene le colonne attese: %s. Skipping.",
                csv_path.name,
                ", ".join(missing_cols),
            )
            continue

        # Ignora file senza righe utili (solo header).
        if df.empty:
            logging.info("File %s vuoto: nessuna riga da unire.", csv_path.name)
            continue

        df = df.copy()
        df["__source_file"] = csv_path.name
        dataframes.append(df)
        stats["files_used"] += 1
        stats["rows_loaded"] += len(df)

    if not dataframes:
        logging.warning("Nessun dato valido trovato nei CSV in %s.", OUTPUT_DIR)
        return pd.DataFrame(columns=EXPECTED_COLUMNS), stats

    combined = pd.concat(dataframes, ignore_index=True)

    # Normalizza stringhe e rimuove righe senza Link.
    combined["SearchTerm"] = _clean_string_series(combined["SearchTerm"])
    combined["Title"] = _clean_string_series(combined["Title"])
    combined["Description"] = _clean_string_series(combined["Description"])
    combined["Qualifications"] = _clean_string_series(combined["Qualifications"])
    combined["Link"] = _clean_string_series(combined["Link"])

    combined = combined.loc[combined["Link"] != ""].copy()
    if combined.empty:
        logging.warning("Nessuna riga contiene un Link valido dopo la pulizia.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS), stats

    combined.sort_values("__source_file", inplace=True)

    aggregation = {
        "SearchTerm": _combine_unique_parts,
        "Title": _pick_longest,
        "Description": _pick_longest,
        "Qualifications": _pick_longest,
        "__source_file": lambda s: " | ".join(dict.fromkeys(s)),
    }

    merged = combined.groupby("Link", as_index=False).agg(aggregation)

    stats["rows_merged"] = len(combined)
    stats["links_unique"] = len(merged)

    merged["SearchTerm"] = merged["SearchTerm"].str.replace("\ufeff", "", regex=False)
    merged.drop(columns="__source_file", inplace=True)
    merged = merged[EXPECTED_COLUMNS]
    merged.sort_values(["SearchTerm", "Title"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged, stats


def resolve_output_path(custom_path: Optional[Path]) -> Path:
    if custom_path:
        return custom_path.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DATAFRAMES_DIR / f"simplyhired_jobs_merged_{timestamp}.csv").resolve()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    dataset, stats = merge_output_csvs(args.pattern)

    if dataset.empty and stats["files_used"] == 0:
        logging.error("Merge interrotto: nessun dato da salvare.")
        return 1

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8-sig")

    logging.info(
        "Salvato dataset con %d righe uniche in %s",
        len(dataset),
        output_path,
    )
    logging.info(
        "File analizzati: %d (utilizzati: %d) | Righe originali: %d | Link unici: %d",
        stats["files_found"],
        stats["files_used"],
        stats["rows_loaded"],
        stats["links_unique"],
    )

    if stats["rows_merged"] and stats["rows_loaded"]:
        duplicates_removed = stats["rows_loaded"] - stats["rows_merged"]
        logging.info("Righe duplicate rimosse: %d", duplicates_removed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
