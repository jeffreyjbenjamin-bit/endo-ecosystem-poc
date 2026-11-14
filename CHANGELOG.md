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

<tr><td>2026-11-14</td><td>

<b>v1.18.1</b>
<ul>
<li>Ask AI (newsletter editor) now answers in a warmer, friendlier, more conversational tone: it notes what already works, offers suggestions as options, and suggests a helpful next step, while staying accurate and to the point.</li>
<li>Friendlier welcome message when the Ask AI panel opens.</li>
</ul>

</td></tr>

<tr><td>2026-11-12</td><td>

<b>v1.18.0</b>
<ul>
<li><b>Ask AI can search for more articles</b> on the newsletter editing page: Claude searches the app's evidence library and past Intelligence Briefing results, and the live web, then recommends articles (title link, source, date, why it fits) with edits you can apply to add them.</li>
<li>Search steps are shown as they happen (searching the library or web, and how many results).</li>
<li>New starter question: "Find recent articles to add to this newsletter".</li>
<li>Web searches are counted and priced in Settings → AI models &amp; usage. If web search isn't enabled for the Anthropic organization, the chat searches the library only and says so.</li>
</ul>

</td></tr>

<tr><td>2026-11-08</td><td>

<b>v1.17.0</b>
<ul>
<li>Home <b>AI trend analysis</b> now uses <b>Claude Sonnet 5.5</b> (GPT-5.1 if Claude isn't set up; switch with TRENDS_AI_PROVIDER).</li>
<li>Each trend lists the collected sources that support it, shown as a source count with a small bar.</li>
<li>New charts: <b>Past-year trends by month</b> (how many supporting sources each trend had each month) and <b>What's being published</b> (items collected per month: articles, press releases, clinical trials), with a Chart data table.</li>
<li>The analysis covers more of the past year (up to 450 most recent items).</li>
</ul>

<b>v1.16.0</b>
<ul>
<li><b>Claude Sonnet 5.5</b> (Anthropic) now writes newsletters (Generate Newsletter) and powers the newsletter editor's <b>Ask AI</b> chat. Everything else stays on Azure OpenAI GPT-5.1.</li>
<li>New shared AI service for Claude calls: server-side only, automatic retries for rate limits and temporary errors, clear error messages, and a fallback model when a request is declined.</li>
<li>Every Claude call logs its input, output and cached tokens and an estimated cost; <b>Settings → AI models &amp; usage</b> shows which model each feature uses and the totals.</li>
<li>Models are chosen with server settings (ANTHROPIC_MODEL, NEWSLETTER_AI_PROVIDER); without an Anthropic API key the newsletter features keep using GPT-5.1.</li>
</ul>

<b>v1.15.0</b>
<ul>
<li><b>Users</b>: a user list with types Admin, Physician, Payer, Patient and Researcher, started from the mailing list (everyone as Admin). Add, edit, deactivate and delete users, and import new mailing-list people, in Settings. Sign-in and permissions by type are not switched on yet.</li>
<li>"Report Config" renamed to <b>Settings</b> (⚙️), with Users at the top and Email &amp; Schedule below.</li>
<li>Ask AI now also gets the newsletter's <b>source articles</b> (titles, links and summaries), so it can check claims against them.</li>
<li>Ask AI can <b>suggest edits</b>: each appears as a before/after card with an <b>Apply</b> button (and Apply all) that changes the newsletter in the editor, ready to review and save.</li>
<li>The Ask AI chat now fills the whole panel height.</li>
<li>Build Newsletter from Intel Briefing moved to the right of Clinical Trials.</li>
</ul>

<b>v1.14.0</b>
<ul>
<li><b>💬 Ask AI</b> on the newsletter editing page opens a chat panel on the right. The AI answers questions using the newsletter being edited, including unsaved changes, and can suggest rewrites to paste in. Starter questions, Clear and close are included, and the conversation is kept while you edit.</li>
<li>Fixed <b>Edit this newsletter</b> in the Home pop-up, which did not open the editor.</li>
</ul>

</td></tr>

<tr><td>2026-11-07</td><td>

<b>v1.13.0</b>
<ul>
<li>Home: Key Developments shown as cards in the briefing's style, with the article's image where the source provides one.</li>
<li>Home: most recent clinical trials moved to the right-hand column under the newsletter, showing ID, Date and Title.</li>
<li>Home: <b>Open latest Intelligence Briefing</b> and <b>Open latest newsletter</b> show the published version in a pop-up over the page; the newsletter pop-up has an Edit button.</li>
</ul>

<b>v1.12.0</b>
<ul>
<li>New <b>Home</b> tab (first tab), a summary of the whole app: the latest briefing's Intelligence Summary and Key Developments, an at-a-glance panel (latest briefing, open trials, latest newsletter with an Open button), recent search results as cards with preview images, and the 5 most recently posted clinical trials.</li>
<li><b>AI trend analysis</b> on Home: top trends of the past week, month and year from briefing results, the evidence library and clinical trials. Updated after each daily briefing, or on demand with Refresh analysis.</li>
<li>Build Intelligence Briefing opens on the most recent briefing.</li>
<li>Fixed <b>View</b> in Past Runs, which failed to load earlier briefings.</li>
<li>On Build Newsletter from Intel Briefing, a clinical trial can be <b>featured</b> like an article, so <b>Find similar articles</b> works from a selected trial.</li>
<li>"SMTP ready" banner removed from Report Config (a warning still shows if email is not set up).</li>
</ul>

<b>v1.11.0</b>
<ul>
<li>Report Config moved from a tab to a <b>⚙️</b> button in the upper right of the top banner; <b>← Back</b> returns to the tab you were on, with your selections kept.</li>
<li>Clinical Trials list refreshes automatically once a day, and again just before the daily email.</li>
<li>Newly listed trials get a <b>NEW</b> badge and are added to the Intelligence Briefing in a <b>New Clinical Trials</b> section until a daily email has carried them.</li>
<li><b>Newsletter</b> checkbox on each clinical trial; marked trials appear on Build Newsletter from Intel Briefing, where they can be added to the newsletter alongside the selected articles.</li>
</ul>

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
