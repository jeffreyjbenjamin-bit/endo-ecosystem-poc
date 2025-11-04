import argparse
from src.connectors import pubmed, ctgov, openalex, nih_reporter
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

    # NIH RePORTER  <-- THIS BLOCK IS THE NEW ONE
    nih_payload = nih_reporter.search(query=args.term, limit=min(args.limit, 100))
    uri_nih = save_raw_json("nih_reporter", "batch", nih_payload)

    print("Saved raw:")
    print("  PubMed      ->", uri_pubmed)
    print("  CT.gov      ->", uri_ctgov)
    print("  OpenAlex    ->", uri_oa)
    print("  NIH RePORTER->", uri_nih)


if __name__ == "__main__":
    main()
