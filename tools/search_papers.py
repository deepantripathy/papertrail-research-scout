"""
search_papers.py - the Searcher step of PaperTrail Research Scout.

Plain Python, no LLM: takes search queries, calls the Semantic Scholar API,
merges and deduplicates the results, and saves candidates to a JSON file
that the Screener subagent reads next.

Usage (from the project root):

  # Quick manual test with one or more queries
  python tools/search_papers.py -q "graph neural networks drug target interaction" -q "GNN explainability molecules"

  # Normal pipeline use: read queries written by the Planner subagent
  python tools/search_papers.py --queries-file output/queries.json

queries.json format (written by the Planner):
  {"queries": ["...", "..."], "year_min": 2018, "year_max": null}

Optional: set a free Semantic Scholar API key to avoid rate-limit errors.
  PowerShell:  $env:SEMANTIC_SCHOLAR_API_KEY = "your-key"
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = ",".join([
    "paperId", "title", "abstract", "year", "venue", "authors",
    "citationCount", "influentialCitationCount", "url",
    "externalIds", "openAccessPdf", "publicationTypes",
])
MAX_RETRIES = 5


def s2_search(query, limit, year_min=None, year_max=None, api_key=None):
    """Run one Semantic Scholar search, retrying on rate limits (HTTP 429)."""
    params = {"query": query, "limit": min(limit, 100), "fields": S2_FIELDS}
    if year_min or year_max:
        params["year"] = f"{year_min or ''}-{year_max or ''}"
    headers = {"x-api-key": api_key} if api_key else {}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            print(f"  network error ({e}); retrying...", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt + 1
            print(f"  HTTP {resp.status_code}; waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json().get("data", []) or []

    print(f"  gave up on query after {MAX_RETRIES} attempts: {query!r}", file=sys.stderr)
    return []


def normalize_title(title):
    """Lowercase, strip punctuation - used to catch duplicates with different IDs."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def to_candidate(p):
    """Convert a raw Semantic Scholar record into our simple candidate format."""
    ext = p.get("externalIds") or {}
    pdf = p.get("openAccessPdf") or {}
    authors = [a.get("name") for a in (p.get("authors") or []) if a.get("name")]
    return {
        "id": p.get("paperId"),
        "title": (p.get("title") or "").strip(),
        "abstract": (p.get("abstract") or "").strip(),
        "year": p.get("year"),
        "venue": p.get("venue") or "",
        "authors": authors[:6] + (["et al."] if len(authors) > 6 else []),
        "citation_count": p.get("citationCount") or 0,
        "influential_citation_count": p.get("influentialCitationCount") or 0,
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
        "url": p.get("url"),
        "pdf_url": pdf.get("url"),
        "is_survey": "Review" in (p.get("publicationTypes") or []),
        "matched_queries": [],
        "source": "semantic_scholar",
    }


def search_all(queries, per_query, year_min, year_max, api_key, require_abstract):
    merged = {}          # key -> candidate
    title_index = {}     # normalized title -> key
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q}", file=sys.stderr)
        for raw in s2_search(q, per_query, year_min, year_max, api_key):
            c = to_candidate(raw)
            if not c["title"]:
                continue
            key = c["doi"] or c["id"]
            norm = normalize_title(c["title"])
            key = title_index.get(norm, key)   # same title seen under another ID
            if key in merged:
                if q not in merged[key]["matched_queries"]:
                    merged[key]["matched_queries"].append(q)
                continue
            c["matched_queries"].append(q)
            merged[key] = c
            title_index[norm] = key
        time.sleep(1.1 if not api_key else 0.2)  # be polite to the free API

    candidates = list(merged.values())
    dropped = 0
    if require_abstract:
        before = len(candidates)
        candidates = [c for c in candidates if c["abstract"]]
        dropped = before - len(candidates)

    # Papers matched by several queries first, then by citations - a sensible
    # default order before the Screener and Ranker do the real work.
    candidates.sort(key=lambda c: (len(c["matched_queries"]), c["citation_count"]), reverse=True)
    return candidates, dropped


def main():
    ap = argparse.ArgumentParser(description="Search Semantic Scholar and save deduplicated candidates.")
    ap.add_argument("-q", "--query", action="append", default=[], help="A search query (repeatable).")
    ap.add_argument("--queries-file", help="JSON file from the Planner with a 'queries' list.")
    ap.add_argument("--per-query", type=int, default=30, help="Results per query (max 100).")
    ap.add_argument("--max-candidates", type=int, default=50, help="Cap on total candidates kept.")
    ap.add_argument("--year-min", type=int)
    ap.add_argument("--year-max", type=int)
    ap.add_argument("--keep-no-abstract", action="store_true", help="Keep papers with no abstract.")
    ap.add_argument("--out", default="output/candidates.json")
    args = ap.parse_args()

    queries = list(args.query)
    year_min, year_max = args.year_min, args.year_max
    if args.queries_file:
        with open(args.queries_file, encoding="utf-8") as f:
            plan = json.load(f)
        queries += plan.get("queries", [])
        year_min = year_min or plan.get("year_min")
        year_max = year_max or plan.get("year_max")
    if not queries:
        ap.error("give at least one --query or a --queries-file")

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print("Note: no SEMANTIC_SCHOLAR_API_KEY set - using the shared free tier (slower).", file=sys.stderr)

    candidates, dropped = search_all(
        queries, args.per_query, year_min, year_max, api_key,
        require_abstract=not args.keep_no_abstract,
    )
    candidates = candidates[: args.max_candidates]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:   # utf-8 matters on Windows
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(candidates)} candidates to {out}"
          f"{f' ({dropped} without abstracts skipped)' if dropped else ''}", file=sys.stderr)
    for c in candidates[:5]:
        print(f"  - {c['title']} ({c['year']}, {c['citation_count']} citations)", file=sys.stderr)


if __name__ == "__main__":
    main()