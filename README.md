# endo-ecosystem-poc
Bold Wave Productions Endo Ecosystem PoC
flowchart LR
  A[Feedly Pro+ \n RSS Collections] --> B[Aggregator / ETL]
  A2[Controlled Web Search \n (Allow-Listed)] --> B
  A3[APIs e.g., NIH RePORTER] --> B
  B --> C[Normalization Layer \n (Provenance, Dedupe)]
  C --> D[AI Insight Engine \n (Summarize, Tag, Cluster)]
  D --> E[Authentication \n Email/Password + Reset]
  E --> F[Actionable Insights Output]
  D --> F
