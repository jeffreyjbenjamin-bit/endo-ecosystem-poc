import argparse
from src.connectors import pubmed, ctgov, openalex, nih_reporter, regulators
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

    print("Saved raw:")
    print("  PubMed       ->", uri_pubmed)
    print("  CT.gov       ->", uri_ctgov)
    print("  OpenAlex     ->", uri_oa)
    print("  NIH RePORTER ->", uri_nih)
    print(f"  EMA          -> {uri_ema} ({len(ema_entries)} items)")
    print(f"  FDA          -> {uri_fda} ({len(fda_entries)} items)")


if __name__ == "__main__":
    main()
