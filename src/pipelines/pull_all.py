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
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    # PubMed
    ids = pubmed.search_ids(args.term, retmax=args.limit)
    pubmed_payload = pubmed.fetch_summaries(ids)
    uri_pubmed = save_raw_json("pubmed", "batch", pubmed_payload)

    # ClinicalTrials.gov
    ctgov_payload = ctgov.search(condition=args.term, pagesize=args.limit)
    uri_ctgov = save_raw_json("ctgov", "batch", ctgov_payload)

    # OpenAlex
    oa_payload = openalex.search(query=args.term, per_page=min(args.limit, 50))
    uri_oa = save_raw_json("openalex", "batch", oa_payload)

    # NIH RePORTER
    nih_payload = nih_reporter.search(query=args.term, limit=min(args.limit, 100))
    uri_nih = save_raw_json("nih_reporter", "batch", nih_payload)

    # EMA & FDA: collect normalized entries then stash raw as a simple envelope
    ema_entries = list(regulators.ema_items())
    fda_entries = list(regulators.fda_items())
    uri_ema = save_raw_json("ema", "batch", {"entries": ema_entries})
    uri_fda = save_raw_json("fda", "batch", {"entries": fda_entries})

    # Preprints (bioRxiv + medRxiv)
    biorxiv_entries = list(preprints.biorxiv_items())
    medrxiv_entries = list(preprints.medrxiv_items())
    uri_bio = save_raw_json("biorxiv", "batch", {"entries": biorxiv_entries})
    uri_med = save_raw_json("medrxiv", "batch", {"entries": medrxiv_entries})

    # Web Search (Google CSE)
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
    print("  bioRxiv     ->", f"{uri_bio} ({len(biorxiv_entries)} items)")
    print("  medRxiv     ->", f"{uri_med} ({len(medrxiv_entries)} items)")

    # Semantic Scholar
    try:
        s2_payload = semantic_scholar.search(query=args.term, limit=min(args.limit, 30))
        uri_s2 = save_raw_json("semantic_scholar", "batch", s2_payload)
        print(
            "  Semantic Scholar ->",
            f"{uri_s2} ({len(s2_payload.get('data', []) or [])} items)",
        )
    except Exception as e:
        print("  Semantic Scholar -> skipped:", e)

    # Crossref
    cr_payload = crossref.search(query=args.term, limit=min(args.limit, 40))
    uri_cr = save_raw_json("crossref", "batch", cr_payload)
    print(
        "  Crossref     ->",
        f"{uri_cr} ({len(cr_payload.get('message', {}).get('items', []) or [])} items)",
    )


if __name__ == "__main__":
    main()
