import logging
import re
import subprocess
import threading
from dataclasses import dataclass
from shutil import which
from typing import TypedDict, List, Tuple
from urllib.parse import quote_plus, urljoin

import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By # fornisce metodi di localizzazione elementi
from selenium.webdriver.support.ui import WebDriverWait # fornisce oggetto wait per aspettare che elemento sia caricato
from selenium.webdriver.support import expected_conditions as EC # contiene condizioni per cui posso considerare un' attesa risolta
from selenium.common.exceptions import TimeoutException

class JobRow(TypedDict):
    SearchTerm: str
    Title: str
    Description: str
    Qualifications: List[str]
    Link: str

class SearchStats(TypedDict):
    search_term: str
    pages_attempted: int
    pages_visited: int
    jobs_collected: int
    job_parse_failures: int
    links_discovered: int

@dataclass
class ScraperConfig:
    headless: bool = True
    window_size: str = "1280,900"
    timeout: int = 15
    chrome_version_main: int | None = None  # imposta versione principale di Chrome/Chromedriver
    chrome_binary: str | None = None        # forza un binario di Chrome specifico

_UC_DRIVER_LOCK = threading.Lock()
_DETECTED_CHROME_MAJOR: int | None = None

def _detect_chrome_major(binary: str | None) -> int | None:
    if not binary:
        return None
    if which(binary) is None:
        return None
    try:
        output = subprocess.check_output([binary, "--version"], text=True)
    except Exception:
        return None
    match = re.search(r"(\d+)\.", output)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None

def _resolve_chrome_major(cfg: ScraperConfig) -> int | None:
    global _DETECTED_CHROME_MAJOR
    if cfg.chrome_version_main:
        return cfg.chrome_version_main
    if _DETECTED_CHROME_MAJOR:
        return _DETECTED_CHROME_MAJOR

    candidates = [cfg.chrome_binary, "google-chrome", "chrome", "chromium-browser", "chromium"]
    for candidate in candidates:
        major = _detect_chrome_major(candidate)
        if major:
            _DETECTED_CHROME_MAJOR = major
            return major
    return None

def make_driver(cfg: ScraperConfig) -> webdriver.Chrome:
    common_args = [
        f"--window-size={cfg.window_size}",
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--no-sandbox",
    ]

    if cfg.headless:
        options = uc.ChromeOptions()
        for arg in common_args:
            options.add_argument(arg)
        options.add_argument("--headless=new")

        uc_kwargs: dict = {"options": options, "headless": True}
        if cfg.chrome_binary:
            uc_kwargs["browser_executable_path"] = cfg.chrome_binary

        major = _resolve_chrome_major(cfg)
        if major:
            uc_kwargs["version_main"] = major

        with _UC_DRIVER_LOCK:
            driver = uc.Chrome(**uc_kwargs)
    else:
        options = webdriver.ChromeOptions()
        for arg in common_args:
            options.add_argument(arg)
        driver = webdriver.Chrome(options=options)

    # Best-effort tweaks to reduce automation fingerprints.
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    })
    driver.implicitly_wait(0)  # solo explicit wait
    return driver

_BULLET_PREFIX = re.compile(r"^[\s\-\u2022\u2023\u25E6\u2043\u2219\u00B7\*\·▪•●]+")  # -, •, ·, *, ecc.
def _normalize_qualification(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = _BULLET_PREFIX.sub("", s)           # rimuove bullet/prefix
    s = re.sub(r"\s+", " ", s).strip()      # compatta spazi
    s = s.rstrip(".,;:·•-–—")               # toglie punteggiatura finale comune (NON tocco '+', utile per "C++")
    return s

def uniq_clean_qualifications(items: list[str]) -> list[str]:
    out, seen = [], set()
    for raw in items:
        s = _normalize_qualification(raw)
        if not s:
            continue                       # filtra vuoti
        key = s.casefold()                 # dedupe case-insensitive/unicode-safe
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

class SimplyHiredScraper:
    def __init__(self, driver: webdriver.Chrome | None = None, cfg: ScraperConfig = ScraperConfig()):
        self._owned = driver is None # se falso non chiudo driver in _exit_ perché non aperto qui
        self.cfg = cfg
        self.driver = driver or make_driver(cfg)
        self.wait = WebDriverWait(self.driver, cfg.timeout, poll_frequency=0.2)
        self._closed = False
        self.log = logging.getLogger(self.__class__.__name__)
    
    def __enter__(self): return self

    def __exit__(self, exc_type, exc, tb):
        if self._owned:
            self.close()

    def close(self):
        if self._closed:
            return
        try:
            self.driver.quit()
        except Exception:
            pass
        finally:
            self._closed = True
    
    def _build_search_url(self, search_term: str) -> str:
        q = quote_plus(search_term, safe="")
        return f"https://www.simplyhired.com/search?q={q}"

    def goto_next_page(self):
        sel = "a[data-testid='pageNumberBlockNext']"
        try:
            el = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
        except Exception:
            self.log.info("Next page button not found")
            return False

        href = el.get_attribute("href") or el.get_dom_attribute("href")
        if not href:
            self.log.info("Next page button missing href")
            return False

        current_url = self.driver.current_url
        self.driver.get(href)

        # Aspetto finché non ho raggiunto la pagina nuova
        try:
            self.wait.until(lambda d: d.current_url != current_url) # driver passato automaticamente da Selenium
        except Exception:
            self.log.exception("Failed while waiting for next page to load")
            return False

        return True

    def collect_job_links(self):
        # prendo gli href RELATIVI delle card e li trasformo in assoluti
        anchors = self.driver.find_elements(By.CSS_SELECTOR, "#job-list [data-testid='searchSerpJobTitle'] a[href*='/job/']")
        base = "https://www.simplyhired.com/"
        links = []

        for a in anchors:
            href = a.get_attribute("href") or a.get_attribute("data-mdref")
            if href:
                # molti href sono relativi ("/job/..."): faccio join col dominio corrente
                links.append(urljoin(base, href))

        # tolgo duplicati preservando l'ordine
        seen, unique = set(), []
        for u in links:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    def scrape_job(self, job_url):
        # Apro il dettaglio in una nuova scheda per non perdere lo stato della SERP
        original = self.driver.current_window_handle
        handles_before = list(self.driver.window_handles)
        self.driver.execute_script("window.open(arguments[0], '_blank');", job_url)

        handles_after = list(self.driver.window_handles)
        opened_new_tab = False

        for handle in handles_after:
            if handle not in handles_before:
                self.driver.switch_to.window(handle)
                opened_new_tab = True
                break

        if not opened_new_tab:
            # Non è stata aperta una nuova scheda (es. popup bloccato), resto dove sono
            self.driver.switch_to.window(handles_after[-1])

        # Sezione parsing Titolo Descrizione e Qualifiche
        try:
            # Selettori
            title_sel = "[data-testid='viewJobTitle']"
            description_sel = "[data-testid='viewJobBodyJobFullDescriptionContent']"
            qualifications_sel = "[data-testid='viewJobQualificationItem']"

            # Aspetto caricamento, non aspetto qualifiche perché potrebbero essere non presenti
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, title_sel)))
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, description_sel)))

            # Elementi DOM
            title_el = self.driver.find_element(By.CSS_SELECTOR, title_sel)
            description_el = self.driver.find_element(By.CSS_SELECTOR, description_sel)
            qualifications_el = self.driver.find_elements(By.CSS_SELECTOR, qualifications_sel)

            # Estraggo testo
            title = title_el.text.strip()
            description_text = description_el.get_attribute("innerText")
            description = description_text.strip() if description_text else ""
            raw_qualifications = [q.text.strip() for q in qualifications_el]
            qualifications = uniq_clean_qualifications(raw_qualifications)

            return {
                "Title": title,
                "Description": description,
                "Qualifications": qualifications,
                "Link": job_url
            }
        except Exception as e:
            self.log.exception("Failed to parse job detail: %s", job_url)
        finally:
            # Chiudo la scheda del job e torno ai risultati
            handles_now = list(self.driver.window_handles)

            if opened_new_tab and len(handles_now) > 1:
                try:
                    self.driver.close()
                except Exception:
                    self.log.warning("Unable to close job tab for %s", job_url)
                handles_now = list(self.driver.window_handles)

            if original in handles_now:
                self.driver.switch_to.window(original)
            elif handles_now:
                self.driver.switch_to.window(handles_now[0])
            else:
                self.log.error("All browser windows were closed while scraping %s", job_url)

    def run_search(self, search_term: str, max_pages: int = 2) -> Tuple[List[JobRow], SearchStats]:

        # Vado pagina iniziale, da li' vedo pagine successive
        url = self._build_search_url(search_term)
        self.driver.get(url)
        try:
            self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="searchSerpJob"]')))
        except Exception:
            self.log.exception("Initial results not loaded for '%s'", search_term)
            return [], {
                "search_term": search_term,
                "pages_attempted": 1,
                "pages_visited": 0,
                "jobs_collected": 0,
                "job_parse_failures": 0,
                "links_discovered": 0,
            }

        jobs = []
        job_failures = 0
        links_discovered = 0
        pages_attempted = 0
        pages_visited = 0

        for page in range(1, max_pages+1):
            pages_attempted += 1
            self.log.info("Reading page %s for '%s'", page, search_term)
            if page > 1: # la prima pagina non cambia link
                if not self.goto_next_page():
                    self.log.warning("Next page not reached for '%s' (target page %s)", search_term, page)
                    break

            # Colleziono i link di tutti i lavori che ho trovato
            try:
                self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="searchSerpJob"]')))
            except TimeoutException:
                self.log.warning(
                    "Timeout while waiting for job cards on '%s' (page %s); stopping pagination",
                    search_term,
                    page,
                )
                break
            except Exception:
                self.log.exception("Failed to parse job links for '%s' (page %s)", search_term, page)
                raise

            links = self.collect_job_links()
            pages_visited += 1
            links_discovered += len(links)
            self.log.info("Found %s job links on page %s for '%s'", len(links), page, search_term)
            for href in links:
                try:
                    row = self.scrape_job(href)
                    if row:
                        jobs.append({
                            "SearchTerm": search_term,
                            "Title": row["Title"],
                            "Description": row["Description"],
                            "Qualifications": row["Qualifications"],
                            "Link": row["Link"],
                        })

                # se qualcosa va storto su un annuncio, continuo
                except Exception:
                    job_failures += 1
                    self.log.exception("Impossibile leggere %s", href)

        stats: SearchStats = {
            "search_term": search_term,
            "pages_attempted": pages_attempted,
            "pages_visited": pages_visited,
            "jobs_collected": len(jobs),
            "job_parse_failures": job_failures,
            "links_discovered": links_discovered,
        }

        return jobs, stats
