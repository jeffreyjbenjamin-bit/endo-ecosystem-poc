import os
import json
import pathlib
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("STORAGE_BACKEND", "none").lower()
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/documents.db")


def save_raw_json(source: str, source_id: str, payload: dict) -> str:
    key = f"raw/{source}/{datetime.utcnow():%Y%m%d}/{source_id or 'batch'}-{int(datetime.utcnow().timestamp())}.json"
    body = json.dumps(payload, ensure_ascii=False).encode()

    if BACKEND == "s3":
        import boto3

        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        bucket = os.getenv("S3_BUCKET")
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        return f"s3://{bucket}/{key}"
    elif BACKEND == "azureblob":
        from azure.storage.blob import BlobServiceClient

        svc = BlobServiceClient(
            account_url=f"https://{os.getenv('AZURE_BLOB_ACCOUNT')}.blob.core.windows.net",
            credential=os.getenv("AZURE_BLOB_KEY"),
        )
        container = os.getenv("AZURE_BLOB_CONTAINER")
        svc.get_container_client(container).upload_blob(key, body, overwrite=True)
        return f"azblob://{container}/{key}"
    else:
        local_path = os.path.join(".", key)
        pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(body)
        return local_path


SQLITE_SCHEMA = """
create table if not exists documents (
  id integer primary key,
  source text not null,
  source_id text,
  title text,
  abstract text,
  url text,
  published_date text,
  authors text,
  journal_or_venue text,
  doi text,
  disease text,
  topics text,
  trial_info text,
  geos text,
  mesh_terms text,
  license text,
  quality_score real default 0,
  hash_sha256 text not null,
  raw_blob_uri text,
  ingested_at text default (datetime('now')),
  lang text
);
create unique index if not exists ux_source_sourceid on documents (source, source_id);
create index if not exists ix_pubdate on documents (published_date);
"""


def sqlite_conn():
    pathlib.Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SQLITE_SCHEMA)
    return conn


def upsert_document(conn, doc: dict):
    def j(x):
        return None if x is None else json.dumps(x, ensure_ascii=False)

    params = (
        doc["source"],
        doc.get("source_id"),
        doc.get("title"),
        doc.get("abstract"),
        doc.get("url"),
        doc.get("published_date"),
        j(doc.get("authors")),
        doc.get("journal_or_venue"),
        doc.get("doi"),
        j(doc.get("disease")),
        j(doc.get("topics")),
        j(doc.get("trial_info")),
        j(doc.get("geos")),
        j(doc.get("mesh_terms")),
        doc.get("license"),
        doc.get("quality_score", 0.0),
        doc["hash_sha256"],
        doc.get("raw_blob_uri"),
        doc.get("lang"),
    )
    conn.execute(
        """
      insert into documents (source,source_id,title,abstract,url,published_date,authors,journal_or_venue,doi,
                             disease,topics,trial_info,geos,mesh_terms,license,quality_score,hash_sha256,raw_blob_uri,lang)
      values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
      on conflict(source,source_id) do update set
        title=excluded.title, abstract=excluded.abstract, url=excluded.url, published_date=excluded.published_date,
        authors=excluded.authors, journal_or_venue=excluded.journal_or_venue, doi=excluded.doi,
        disease=excluded.disease, topics=excluded.topics, trial_info=excluded.trial_info, geos=excluded.geos,
        mesh_terms=excluded.mesh_terms, license=excluded.license, quality_score=excluded.quality_score,
        hash_sha256=excluded.hash_sha256, raw_blob_uri=excluded.raw_blob_uri, lang=excluded.lang
    """,
        params,
    )
