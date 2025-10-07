from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import logging
import re

from scraper_utilities import SimplyHiredScraper, ScraperConfig

SEARCH_TERMS = [
    #"Big Data",
    #"Data Science",
    "Business Intelligence",
    #"Data Mining",
    #"Machine Learning",
    "Data Analytics"
]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

def slugify(term: str) -> str:
    slug = _NON_ALNUM.sub("_", term.casefold())
    slug = slug.strip("_")
    return slug or "term"

def save_flat_csv(rows, out_path):
    fieldnames = ["SearchTerm", "Title", "Description", "Qualifications", "Link"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            quals = r.get("Qualifications") or []
            quals_str = " | ".join(q for q in quals if q) if isinstance(quals, list) else str(quals)
            # se non vuoi a capo in Description, decommenta la riga sotto
            # desc = (r.get("Description") or "").replace("\r\n", "\n").replace("\n", " ")
            w.writerow({
                "SearchTerm": r.get("SearchTerm", ""),
                "Title": r.get("Title", ""),
                "Description": r.get("Description", ""),  # oppure usa 'desc' se normalizzi
                "Qualifications": quals_str,
                "Link": r.get("Link", ""),
            })

def configure_logging(timestamp: str) -> Path:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"simplyhired_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger("selenium").setLevel(logging.WARNING)
    return log_path

def save_stats(stats_payload, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path.cwd() / "output"
    csv_path = output_dir / f"simplyhired_jobs_{timestamp}.csv"
    stats_path = output_dir / f"simplyhired_stats_{timestamp}.json"
    log_path = configure_logging(timestamp)

    if not SEARCH_TERMS:
        logging.warning("No search terms provided; nothing to scrape.")
        return

    cfg = ScraperConfig(headless=False, window_size="1280,900", timeout=15)

    checkpoint_root = output_dir / "checkpoints" / timestamp
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def empty_stats(term: str) -> dict:
        return {
            "search_term": term,
            "pages_attempted": 0,
            "pages_visited": 0,
            "jobs_collected": 0,
            "job_parse_failures": 0,
            "links_discovered": 0,
        }

    def write_checkpoint(term: str, rows: list[dict], stats: dict):
        slug = slugify(term)
        term_dir = checkpoint_root / slug
        term_dir.mkdir(parents=True, exist_ok=True)
        save_flat_csv(rows, term_dir / f"{slug}.csv")
        payload = {
            "search_term": term,
            "stats": stats,
            "rows_saved": len(rows),
            "generated_at": timestamp,
        }
        with (term_dir / f"{slug}_stats.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run_term(term: str):
        logging.info("Starting scrape for '%s'", term)
        with SimplyHiredScraper(cfg=cfg) as bot:
            return bot.run_search(term, max_pages=300)

    term_results = {}
    max_workers = min(3, len(SEARCH_TERMS))
    futures = {}
    interrupted = False
    persisted_terms: set[str] = set()
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {pool.submit(run_term, term): term for term in SEARCH_TERMS}
        for future in as_completed(futures):
            term = futures[future]
            try:
                rows, stats = future.result()
            except Exception:
                logging.exception("Search for '%s' failed", term)
                term_results[term] = ([], empty_stats(term))
                write_checkpoint(term, [], term_results[term][1])
                persisted_terms.add(term)
                continue
            term_results[term] = (rows, stats)
            write_checkpoint(term, rows, stats)
            persisted_terms.add(term)
    except KeyboardInterrupt:
        interrupted = True
        logging.warning("Interrupted by user; attempting graceful shutdown.")
        for future, term in futures.items():
            if term in persisted_terms:
                continue
            if term in term_results:
                rows, stats = term_results[term]
                write_checkpoint(term, rows, stats)
                persisted_terms.add(term)
                continue
            if future.done():
                try:
                    rows, stats = future.result()
                except Exception:
                    logging.exception("Search for '%s' failed during shutdown", term)
                    term_results[term] = ([], empty_stats(term))
                    write_checkpoint(term, [], term_results[term][1])
                else:
                    term_results[term] = (rows, stats)
                    write_checkpoint(term, rows, stats)
                persisted_terms.add(term)
            else:
                future.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    all_rows = []
    stats_per_term = []
    totals = {
        "pages_attempted": 0,
        "pages_visited": 0,
        "jobs_collected": 0,
        "job_parse_failures": 0,
        "links_discovered": 0,
    }

    for term in SEARCH_TERMS:
        rows, stats = term_results.get(term, ([], empty_stats(term)))
        all_rows.extend(rows)
        stats_per_term.append(stats)
        totals["pages_attempted"] += stats["pages_attempted"]
        totals["pages_visited"] += stats["pages_visited"]
        totals["jobs_collected"] += stats["jobs_collected"]
        totals["job_parse_failures"] += stats["job_parse_failures"]
        totals["links_discovered"] += stats["links_discovered"]

    save_flat_csv(all_rows, csv_path)

    stats_payload = {
        "run_timestamp": timestamp,
        "search_terms": SEARCH_TERMS,
        "log_file": str(log_path),
        "csv_file": str(csv_path),
        "stats_file": str(stats_path),
        "totals": totals,
        "per_term": stats_per_term,
        "total_rows": len(all_rows),
    }

    save_stats(stats_payload, stats_path)
    logging.info("Wrote %s job rows to %s", len(all_rows), csv_path)
    logging.info("Saved run statistics to %s", stats_path)
    print(f"Wrote {len(all_rows)} rows to: {csv_path}")
    print(f"Statistics saved to: {stats_path}")
    print(f"Log file: {log_path}")
    if interrupted:
        print("Run interrupted — salvati i risultati disponibili.")

if __name__ == "__main__":
    main()
