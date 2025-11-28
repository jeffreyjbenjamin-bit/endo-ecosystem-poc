import argparse
from src.connectors import (
    pubmed,
    ctgov,
    openalex,
    nih_reporter,
    regulators,
    preprints,
    web_search,
    semantic_scholar,
    crossref,
)
from src.common.storage import save_raw_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="endometriosis")

    # Option 2 — set core limit to 500
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    # ----- PubMed (fully scalable) -----
    ids = pubmed.search_ids(args.term, retmax=args.limit)
    pubmed_payload = pubmed.fetch_summaries(ids)
    uri_pubmed = save_raw_json("pubmed", "batch", pubmed_payload)

    # ----- ClinicalTrials.gov -----
    ctgov_payload = ctgov.search(condition=args.term, pagesize=args.limit)
    uri_ctgov = save_raw_json("ctgov", "batch", ctgov_payload)

    # ----- OpenAlex (API max per_page=50) -----
    oa_limit = min(args.limit, 50)
    oa_payload = openalex.search(query=args.term, per_page=oa_limit)
    uri_oa = save_raw_json("openalex", "batch", oa_payload)

    # ----- NIH RePORTER (connector limit=100) -----
    nih_limit = min(args.limit, 100)
    nih_payload = nih_reporter.search(query=args.term, limit=nih_limit)
    uri_nih = save_raw_json("nih_reporter", "batch", nih_payload)

    # ----- EMA & FDA -----
    ema_entries = list(regulators.ema_items())
    fda_entries = list(regulators.fda_items())
    uri_ema = save_raw_json("ema", "batch", {"entries": ema_entries})
    uri_fda = save_raw_json("fda", "batch", {"entries": fda_entries})

    # ----- Preprints (no safe limit exposed) -----
    biorxiv_entries = list(preprints.biorxiv_items())
    medrxiv_entries = list(preprints.medrxiv_items())
    uri_bio = save_raw_json("biorxiv", "batch", {"entries": biorxiv_entries})
    uri_med = save_raw_json("medrxiv", "batch", {"entries": medrxiv_entries})

    # ----- Google CSE (hard limit 10) -----
    try:
        web_payload = web_search.search_google_cse(query=args.term, num=10)
        uri_web = save_raw_json("web_search", "batch", web_payload)
        count_web = len(web_payload.get("items", []) or [])
        print("  Web Search  ->", f"{uri_web} ({count_web} items)")
    except Exception as e:
        print("  Web Search  -> skipped:", e)

    print("Saved raw:")
    print("  PubMed       ->", uri_pubmed)
    print("  CT.gov       ->", uri_ctgov)
    print("  OpenAlex     ->", uri_oa)
    print("  NIH RePORTER ->", uri_nih)
    print(f"  EMA          -> {uri_ema} ({len(ema_entries)} items)")
    print(f"  FDA          -> {uri_fda} ({len(fda_entries)} items)")
    print("  bioRxiv      ->", f"{uri_bio} ({len(biorxiv_entries)} items)")
    print("  medRxiv      ->", f"{uri_med} ({len(medrxiv_entries)} items)")

    # ----- Semantic Scholar (safe limit=30) -----
    try:
        s2_limit = min(args.limit, 30)
        s2_payload = semantic_scholar.search(query=args.term, limit=s2_limit)
        uri_s2 = save_raw_json("semantic_scholar", "batch", s2_payload)
        print(
            "  Semantic Scholar ->",
            f"{uri_s2} ({len(s2_payload.get('data', []) or [])} items)",
        )
    except Exception as e:
        print("  Semantic Scholar -> skipped:", e)

    # ----- Crossref (safe ~40) -----
    cr_limit = min(args.limit, 40)
    cr_payload = crossref.search(query=args.term, limit=cr_limit)
    uri_cr = save_raw_json("crossref", "batch", cr_payload)
    print(
        "  Crossref     ->",
        f"{uri_cr} ({len(cr_payload.get('message', {}).get('items', []) or [])} items)",
    )


if __name__ == "__main__":
    main()
