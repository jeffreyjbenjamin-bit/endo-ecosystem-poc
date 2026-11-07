# Changelog

All notable changes to Evidence Gap, newest first.

<!-- Maintainers: add new rows at the top of the table. The app shows the version in
the first row in its footer. Type is one of: Added, Changed, Removed, Fixed. -->

| Version | Date | Type | Change |
|---|---|---|---|
| 1.6.0 | 2026-11-07 | Changed | "Newsletter Builder v2" renamed to **Build Newsletter from Intel Briefing**. |
| 1.6.0 | 2026-11-07 | Changed | Briefing articles shown as tiles, like AI Article Search & Summary, to cut scrolling. |
| 1.6.0 | 2026-11-07 | Changed | **Find similar articles** button and search options moved to the top of the tab. |
| 1.6.0 | 2026-11-07 | Added | Similar articles open in a panel on the right, each with a short explanation of why it relates to the featured article. |
| 1.6.0 | 2026-11-07 | Removed | Output Comparison tab. |
| 1.6.0 | 2026-11-07 | Removed | Q & A tab. |
| 1.6.0 | 2026-11-07 | Changed | Changelog shown as a table. |
| 1.5.0 | 2026-11-07 | Added | Daily email send runs automatically on the hosted app at the time set in Report Config: a fresh AI Article Search, then the Intelligence Briefing and any queued Newsletter. |
| 1.5.0 | 2026-11-07 | Added | Send time is in US Eastern time; each day's send runs only once, with a two-hour catch-up window after restarts. |
| 1.5.0 | 2026-11-07 | Changed | Report Config shows the time zone, that sending is automatic, and the last send date. |
| 1.5.0 | 2026-11-07 | Added | Footer on every page with © BoldWave, Inc., the app version, and a link to this changelog. |
| 1.5.0 | 2026-11-07 | Added | Changelog page in the app. |
| 1.4.0 | 2026-11-07 | Added | Automatic deployment: every update to the main code branch is released to the hosted app, with a health check. |
| 1.4.0 | 2026-11-07 | Changed | Hosted app keeps its saved data (briefings, newsletters, mailing list, history) in persistent storage, so releases and restarts no longer reset it. |
| 1.3.0 | 2026-11-07 | Added | Newsletter Builder v2: start a newsletter from a past Intelligence Briefing, feature one or two of its articles, find similar articles in the evidence library and on the web, choose supporting articles, then generate, edit, preview and save. |
| 1.3.0 | 2026-11-07 | Changed | "Newsletter Config" tab renamed to **Report Config**. |
| 1.3.0 | 2026-11-07 | Fixed | Hosted app build failure caused by newer versions of a third-party library. |
| 1.2.0 | 2026-07-01 | Added | Diagnostics, Therapeutics and Epidemiology search categories and matching briefing sections. |
| 1.2.0 | 2026-07-01 | Added | Publication dates noted in the briefing where available. |
| 1.2.0 | 2026-07-01 | Changed | Intelligence Briefing in the app uses the same card layout as the emailed version. |
| 1.2.0 | 2026-07-01 | Changed | AI Article Search defaults to the last 7 days (was 90). |
| 1.1.0 | 2026-06-27 | Added | Newsletter Builder split into Select Highlights and Newsletter Builder views. |
| 1.1.0 | 2026-06-27 | Added | AI Topic Intelligence with clickable topic filters. |
| 1.1.0 | 2026-06-27 | Added | Recipient subscription types: briefing, newsletter, or both. |
| 1.1.0 | 2026-06-27 | Added | Send a newsletter immediately, or queue it for the next daily briefing. |
| 1.1.0 | 2026-06-27 | Added | Quick-send bar and Markdown formatting toolbar in the newsletter editor. |
| 1.1.0 | 2026-06-27 | Added | Publish-date filter and maximum publication age in Select Highlights. |
| 1.1.0 | 2026-06-27 | Changed | Model selection moved to the Q & A tab, dataset controls to RAG Article Presentation; sidebar removed. |
| 1.1.0 | 2026-06-27 | Changed | Daily send runs a real AI Article Search when asked, sends by subscription type, and sends queued newsletters separately. |
| 1.0.0 | 2026-06-20 | Added | AI Article Search & Summary: scored web search written up as an Intelligence Briefing. |
| 1.0.0 | 2026-06-20 | Added | Output Comparison of AI Search and RAG briefings. |
| 1.0.0 | 2026-06-20 | Added | Newsletter Builder for composing newsletters from briefing highlights. |
| 1.0.0 | 2026-06-20 | Added | Email delivery of briefings and newsletters, with a mailing list. |
| 1.0.0 | 2026-06-21 | Changed | Evidence library refreshed. |
| 0.3.0 | 2025-11-28 | Added | GPT-5.1 chat model alongside GPT-4o-mini; larger data pulls from sources. |
| 0.3.0 | 2025-11-29 | Added | Password protection for the app. |
| 0.3.0 | 2025-11-30 | Added | Hosted deployment of the Streamlit app. |
| 0.3.0 | 2025-12-08 | Changed | Evidence library refreshed (also 2025-11-16 and 2025-11-27). |
| 0.2.0 | 2025-11-05 | Added | Evidence library ingestion from PubMed, ClinicalTrials.gov, FDA, EMA, bioRxiv, medRxiv, OpenAlex, Semantic Scholar, Crossref, NIH RePORTER and web search. |
| 0.2.0 | 2025-11-05 | Added | Normalization, de-duplication, unique IDs and Parquet/SQLite storage. |
| 0.2.0 | 2025-11-05 | Added | Azure OpenAI embeddings, FAISS vector index, and question answering over the library (RAG). |
| 0.2.0 | 2025-11-05 | Added | Health checks, query logging, and the first Streamlit interface (Ask and History). |
| 0.1.0 | 2025-10-27 | Added | Project foundation: repository structure, architecture overview, development environment, configuration, CI, tests and code formatting. |
| 0.1.0 | 2025-10-28 | Added | First ingestion helpers (HTTP and ClinicalTrials.gov). |
