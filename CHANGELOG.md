# Changelog

All notable changes to Evidence Gap, newest first. Each version lists the day it was
released.

<!-- Maintainers: add new versions at the top. The app reads the first
"## [x.y.z]" heading for the version number shown in its footer. -->



## [1.5.0] — 2026-11-07

### Added
- Daily email send now runs automatically on the hosted app every day at the time set
  in Report Config: a fresh AI Article Search, then the Intelligence Briefing to
  briefing subscribers and any queued Newsletter to newsletter subscribers.
- Send time is in US Eastern time (daylight saving handled), and each day's send runs
  only once, with a two-hour catch-up window after restarts.
- Report Config shows the time zone, that sending is automatic, and the last send date.
- Persistent footer on every page: © BoldWave, Inc., the app version, and a link to
  this changelog.
- Changelog page in the app.

## [1.4.0] — 2026-11-07

### Added
- Automatic deployment: every update to the main code branch is built and released to
  the hosted app without manual steps, with a health check after each release.

### Changed
- The hosted app keeps its saved data (briefings, newsletters, mailing list,
  comparisons, Q&A history) in persistent storage, so releases and restarts no longer
  reset it.

## [1.3.0] — 2026-11-07

### Added
- **Newsletter Builder v2** tab (right of Q & A): start a newsletter from a past
  Intelligence Briefing (defaults to the most recent), feature one or two of its
  articles, find similar articles in the evidence library and on the web, choose
  supporting articles, then generate, edit, preview and save the newsletter.
  Drafts also appear in the original Newsletter Builder for sending or scheduling.

### Changed
- "Newsletter Config" tab renamed to **Report Config**.

### Fixed
- Hosted app build failure caused by newer versions of a third-party library.

## [1.2.0] — 2026-07-01

### Added
- Diagnostics, Therapeutics and Epidemiology search categories and matching
  Intelligence Briefing sections; new categories selectable in the focus area picker.
- Publication dates noted in the briefing where available.

### Changed
- Intelligence Briefing in the app now uses the same layout as the emailed version,
  with card-style sections throughout.
- AI Article Search defaults to the last 7 days (was 90).

## [1.1.0] — 2026-06-27

### Added
- Newsletter Builder split into **Select Highlights** and **Newsletter Builder** views.
- AI Topic Intelligence with clickable topic filters.
- Recipient subscription types: briefing, newsletter, or both.
- Send a newsletter immediately, or queue it to go out with the next daily briefing.
- Quick-send bar on the draft newsletter, and a Markdown formatting toolbar in the editor.
- Publish-date filter and maximum publication age in Select Highlights.

### Changed
- Model selection and New Q&A Session moved to the Q & A tab; dataset controls moved
  to the RAG Article Presentation tab; sidebar removed.
- Daily send script runs a real AI Article Search when asked, sends only to the right
  subscription types, and sends a queued newsletter as a separate email.

## [1.0.0] — 2026-06-20

### Added
- **AI Article Search & Summary**: web search across endometriosis research, news and
  regulatory sources, scored for recency, credibility, relevance and importance, and
  written up as an Intelligence Briefing.
- **Output Comparison** of AI Search and RAG briefings using a quality rubric.
- **Newsletter Builder** for composing newsletters from briefing highlights.
- Email delivery of briefings and newsletters, with a mailing list.
- Dataset refreshed (2026-06-21).

## [0.3.0] — 2025-11-28 to 2025-12-08

### Added
- GPT-5.1 chat model alongside GPT-4o-mini.
- Password protection for the app.
- Hosted deployment of the Streamlit app.

### Changed
- Larger data pulls from sources; refreshed evidence library (2025-11-16 through
  2025-12-08).

## [0.2.0] — 2025-11-04 to 2025-11-05

### Added
- Evidence library ingestion from ten sources: PubMed, ClinicalTrials.gov, FDA, EMA,
  bioRxiv, medRxiv, OpenAlex, Semantic Scholar, Crossref and NIH RePORTER, plus web search.
- Normalization, de-duplication, unique IDs and Parquet/SQLite storage.
- Azure OpenAI embeddings with a FAISS vector index; retrieval and question answering
  over the library (RAG).
- Health checks and logging of every question and answer.
- First Streamlit interface: Ask and History.

## [0.1.0] — 2025-10-27 to 2025-10-29

### Added
- Project foundation: repository structure, architecture overview, development
  environment, configuration via environment files, automated checks (CI, tests,
  code formatting), and the first ingestion helpers.
