# Changelog

All notable changes to Evidence Gap, by day, newest first.

<!-- Maintainers: add a new row at the top for each day (or a new version block at the
top of today's row). The app shows the first version number in this table in its
footer. Keep each version's changes in its own <ul> list. -->

<table>
<thead>
<tr><th>Date</th><th>Change</th></tr>
</thead>
<tbody>

<tr><td>2026-11-07</td><td>

<b>v1.10.0</b>
<ul>
<li>New <b>Clinical Trials</b> tab (before Report Config): open endometriosis trials with at least one US site from ClinicalTrials.gov, with ID (linked to the study, source shown below), Title, Lead Sponsor, Study / Phase, Status, Primary Completion, Enrollment and Site Country.</li>
<li><b>Refresh</b> runs a new search; the last results are kept between visits.</li>
<li><b>Companies only</b> filter hides academic, hospital and government sponsors.</li>
<li><b>Confirmed endometriosis only</b> filter keeps trials whose inclusion criteria require a surgically or pathologically confirmed diagnosis.</li>
<li>"Search provider" status message removed from Build Intelligence Briefing (problems with the search connection are still shown).</li>
</ul>

<b>v1.9.0</b>
<ul>
<li>Build Intelligence Briefing now searches company press releases: a new <b>Industry &amp; Press Releases</b> focus area (Business Wire, PR Newswire, GlobeNewswire, and Phase 1–3 / diagnostics announcements), included automatically when no focus area is chosen. All existing searches are unchanged.</li>
<li>Press releases are tagged, shown with a <b>Press Release</b> badge, and up to three on-topic ones are kept among the sources each briefing is written from.</li>
<li>New <b>Industry &amp; Company News</b> briefing section: company, product, modality, development stage and the news, attributed to the company.</li>
</ul>

<b>v1.8.0</b>
<ul>
<li>RAG Article Presentation and the original Newsletter Builder tabs hidden.</li>
<li><b>Recent newsletters</b> menu on Build Newsletter from Intel Briefing to reopen any recent newsletter for editing.</li>
<li>Visual (WYSIWYG) newsletter editor with a formatting toolbar; a Markdown view is still available.</li>
<li><b>Send</b> button on the editing page emails the newsletter to newsletter subscribers after a confirmation, and <b>Send with daily briefing</b> queues it for the next daily send.</li>
</ul>

<b>v1.7.0</b>
<ul>
<li>"AI Article Search &amp; Summary" tab renamed to <b>Build Intelligence Briefing</b>.</li>
<li><b>Build Newsletter from Intel Briefing</b> moved to the second tab.</li>
<li>Supporting articles and the newsletter title, note and Generate button moved to the top of that tab.</li>
<li>Generate Newsletter opens a separate editing page with the editor and live preview side by side; Back returns to the tab with selections kept.</li>
<li>Changelog shown as a Date and Change table, one row per day.</li>
</ul>

<b>v1.6.0</b>
<ul>
<li>"Newsletter Builder v2" renamed to <b>Build Newsletter from Intel Briefing</b>.</li>
<li>Briefing articles shown as tiles to cut scrolling.</li>
<li>Find similar articles button and search options moved to the top of the tab.</li>
<li>Similar articles open in a panel on the right, each with a short explanation of why it relates to the featured article.</li>
<li>Output Comparison and Q &amp; A tabs removed.</li>
</ul>

<b>v1.5.0</b>
<ul>
<li>Daily email send runs automatically on the hosted app: a fresh search, then the Intelligence Briefing and any queued Newsletter.</li>
<li>Send time is in US Eastern time; each day's send runs only once, with a two-hour catch-up window after restarts.</li>
<li>Report Config shows the time zone, that sending is automatic, and the last send date.</li>
<li>Footer on every page with © BoldWave, Inc., the app version, and a link to this changelog.</li>
</ul>

<b>v1.4.0</b>
<ul>
<li>Automatic deployment: every update to the main code branch is released to the hosted app, with a health check.</li>
<li>Hosted app keeps its saved data (briefings, newsletters, mailing list, history) in persistent storage, so releases and restarts no longer reset it.</li>
</ul>

<b>v1.3.0</b>
<ul>
<li>Newsletter Builder v2: start a newsletter from a past Intelligence Briefing, feature one or two of its articles, find similar articles, choose supporting ones, then generate, edit and save.</li>
<li>"Newsletter Config" tab renamed to <b>Report Config</b>.</li>
<li>Fixed a hosted app build failure caused by newer versions of a third-party library.</li>
</ul>

</td></tr>

<tr><td>2026-07-01</td><td>

<b>v1.2.0</b>
<ul>
<li>Diagnostics, Therapeutics and Epidemiology search categories and matching briefing sections.</li>
<li>Publication dates noted in the briefing where available.</li>
<li>Intelligence Briefing in the app uses the same card layout as the emailed version.</li>
<li>Article search defaults to the last 7 days (was 90).</li>
</ul>

</td></tr>

<tr><td>2026-06-27</td><td>

<b>v1.1.0</b>
<ul>
<li>Newsletter Builder split into Select Highlights and Newsletter Builder views.</li>
<li>AI Topic Intelligence with clickable topic filters.</li>
<li>Recipient subscription types: briefing, newsletter, or both.</li>
<li>Send a newsletter immediately, or queue it for the next daily briefing.</li>
<li>Quick-send bar and Markdown formatting toolbar in the newsletter editor.</li>
<li>Publish-date filter and maximum publication age in Select Highlights.</li>
<li>Model selection and dataset controls moved into their tabs; sidebar removed.</li>
<li>Daily send runs a real search when asked, sends by subscription type, and sends queued newsletters separately.</li>
</ul>

</td></tr>

<tr><td>2026-06-21</td><td>

<b>v1.0.0</b>
<ul>
<li>Evidence library refreshed.</li>
</ul>

</td></tr>

<tr><td>2026-06-20</td><td>

<b>v1.0.0</b>
<ul>
<li>AI Article Search &amp; Summary: scored web search written up as an Intelligence Briefing.</li>
<li>Output Comparison of AI Search and RAG briefings.</li>
<li>Newsletter Builder for composing newsletters from briefing highlights.</li>
<li>Email delivery of briefings and newsletters, with a mailing list.</li>
</ul>

</td></tr>

<tr><td>2025-12-08</td><td>

<b>v0.3.0</b>
<ul>
<li>Evidence library refreshed.</li>
</ul>

</td></tr>

<tr><td>2025-11-30</td><td>

<b>v0.3.0</b>
<ul>
<li>Hosted deployment of the Streamlit app.</li>
</ul>

</td></tr>

<tr><td>2025-11-29</td><td>

<b>v0.3.0</b>
<ul>
<li>Password protection for the app.</li>
</ul>

</td></tr>

<tr><td>2025-11-28</td><td>

<b>v0.3.0</b>
<ul>
<li>GPT-5.1 chat model alongside GPT-4o-mini.</li>
<li>Larger data pulls from sources.</li>
</ul>

</td></tr>

<tr><td>2025-11-27</td><td>

<b>v0.2.0</b>
<ul>
<li>Evidence library refreshed (also 2025-11-16 and 2025-11-19).</li>
</ul>

</td></tr>

<tr><td>2025-11-05</td><td>

<b>v0.2.0</b>
<ul>
<li>Evidence library ingestion from PubMed, ClinicalTrials.gov, FDA, EMA, bioRxiv, medRxiv, OpenAlex, Semantic Scholar, Crossref, NIH RePORTER and web search.</li>
<li>Normalization, de-duplication, unique IDs and Parquet/SQLite storage.</li>
<li>Azure OpenAI embeddings, FAISS vector index, and question answering over the library (RAG).</li>
<li>Health checks, query logging, and the first Streamlit interface (Ask and History).</li>
</ul>

</td></tr>

<tr><td>2025-11-04</td><td>

<b>v0.2.0</b>
<ul>
<li>First evidence library sources, including FDA and EMA, with normalization.</li>
</ul>

</td></tr>

<tr><td>2025-10-28</td><td>

<b>v0.1.0</b>
<ul>
<li>First ingestion helpers (HTTP and ClinicalTrials.gov).</li>
</ul>

</td></tr>

<tr><td>2025-10-27</td><td>

<b>v0.1.0</b>
<ul>
<li>Project foundation: repository structure, architecture overview, development environment, configuration, CI, tests and code formatting.</li>
</ul>

</td></tr>

</tbody>
</table>
