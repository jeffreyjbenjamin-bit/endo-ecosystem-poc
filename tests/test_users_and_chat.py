"""Users table and the editing page's chat (sources, suggested edits)."""

import json
import sqlite3

import pytest

from src.common import users as U
from src.ui.newsletter_from_briefing import (
    apply_edit,
    chat_system_prompt,
    newsletter_sources,
    parse_edits,
)


@pytest.fixture
def dbs(tmp_path):
    ml = tmp_path / "mailing_list.sqlite"
    conn = sqlite3.connect(ml)
    conn.execute(
        "CREATE TABLE recipients (id INTEGER PRIMARY KEY, name TEXT, email TEXT, "
        "added_ts TEXT, last_sent_ts TEXT, subscription_type TEXT)"
    )
    conn.executemany(
        "INSERT INTO recipients (name, email) VALUES (?, ?)",
        [("Jeff B", "Jeff@Example.com"), ("Ann", "ann@example.com")],
    )
    conn.commit()
    conn.close()
    return {"db_path": str(tmp_path / "users.sqlite"), "mailing_db": str(ml)}


def test_users_seeded_from_mailing_list_as_admins_once(dbs):
    assert U.ensure_users(**dbs) == 2
    users = U.list_users(dbs["db_path"])
    assert {u["email"] for u in users} == {"jeff@example.com", "ann@example.com"}
    assert {u["user_type"] for u in users} == {"Admin"}
    assert all(u["source"] == "mailing list" and u["active"] for u in users)
    # deleting everyone does not re-seed
    for u in users:
        U.delete_user(u["id"], dbs["db_path"])
    assert U.ensure_users(**dbs) == 0
    assert U.list_users(dbs["db_path"]) == []


def test_add_update_validate_and_import(dbs):
    U.ensure_users(**dbs)
    uid = U.add_user("Dr Lee", "lee@clinic.org", "Physician", db_path=dbs["db_path"])
    with pytest.raises(U.UserError):
        U.add_user("Dup", "LEE@clinic.org", "Patient", db_path=dbs["db_path"])
    with pytest.raises(U.UserError):
        U.add_user("Bad", "not-an-email", "Patient", db_path=dbs["db_path"])
    with pytest.raises(U.UserError):
        U.add_user("Bad", "x@y.org", "Doctor", db_path=dbs["db_path"])
    U.update_user(uid, "Dr Lee", "lee@clinic.org", "Researcher", False, dbs["db_path"])
    lee = next(u for u in U.list_users(dbs["db_path"]) if u["id"] == uid)
    assert lee["user_type"] == "Researcher" and lee["active"] is False
    # import adds only people who are not users yet
    conn = sqlite3.connect(dbs["mailing_db"])
    conn.execute("INSERT INTO recipients (name, email) VALUES ('Pat', 'pat@x.org')")
    conn.commit()
    conn.close()
    assert U.import_mailing_list("Patient", **dbs) == 1
    pat = next(u for u in U.list_users(dbs["db_path"]) if u["email"] == "pat@x.org")
    assert pat["user_type"] == "Patient"


def test_chat_gets_sources_with_summaries():
    nl = {
        "sources_json": json.dumps(
            [
                {
                    "role": "featured",
                    "title": "Drug X trial",
                    "url": "https://a.com/x",
                    "source": "a.com",
                    "published_date": "2026-06-01",
                    "reason": "",
                },
                {
                    "role": "supporting",
                    "title": "Test Y",
                    "url": "https://b.com/y",
                    "source": "b.com",
                    "snippet": "Blood test Y has 90% sensitivity.",
                },
            ]
        ),
        "config_json": json.dumps({"briefing_run_id": 7}),
    }
    runs = [
        {
            "id": 7,
            "sources_json": json.dumps(
                [{"url": "https://a.com/x/", "snippet": "Phase 3 met its endpoint."}]
            ),
        }
    ]
    srcs = newsletter_sources(nl, runs)
    assert [s["summary"] for s in srcs] == [
        "Phase 3 met its endpoint.",  # filled in from the briefing
        "Blood test Y has 90% sensitivity.",
    ]
    prompt = chat_system_prompt("T", "June", "Drug X works [1].", srcs)
    assert "[1] Drug X trial (featured · a.com · 2026-06-01)" in prompt
    assert "Phase 3 met its endpoint." in prompt and "<<<EDIT" in prompt
    # original Newsletter Builder sources
    legacy = {
        "sources_json": json.dumps(
            [
                {
                    "content": "Title line\nBody text",
                    "source_type": "rag",
                    "section": "lead",
                }
            ]
        )
    }
    assert newsletter_sources(legacy)[0]["title"] == "Title line"


def test_parse_and_apply_suggested_edits():
    doc = "# News\n\nDrug X **works** in   most patients [1].\n\nOther text."
    answer = (
        "Soften the claim.\n<<<EDIT\nFIND:\nDrug X **works** in most patients [1].\n"
        "REPLACE:\nDrug X showed benefit in a Phase 3 trial [1].\n>>>\n"
        "And drop the filler.\n<<<EDIT\nFIND:\nOther text.\nREPLACE:\n>>>"
    )
    text, edits = parse_edits(answer)
    assert "<<<EDIT" not in text and "(suggested edit 2 below)" in text
    assert edits[1] == {"find": "Other text.", "replace": ""}
    # whitespace differences are tolerated
    doc2 = apply_edit(doc, edits[0]["find"], edits[0]["replace"])
    assert "showed benefit in a Phase 3 trial [1]." in doc2
    doc3 = apply_edit(doc2, edits[1]["find"], edits[1]["replace"])
    assert doc3.rstrip().endswith("[1].")
    assert apply_edit(doc, "Not in the document", "x") is None


def test_library_search_merges_library_and_briefings():
    import numpy as np
    import pandas as pd

    from src.ui.newsletter_from_briefing import library_search

    df = pd.DataFrame(
        [
            {
                "uid": "u1",
                "title": "Dienogest trial",
                "url": "https://pubmed/1",
                "source": "pubmed",
                "published_date": "2026-05-01",
                "abstract": "Long text " * 5,
            },
        ]
    )
    runs = [
        {
            "sources_json": json.dumps(
                [
                    {
                        "title": "Dienogest vs GnRH",
                        "url": "https://news/2",
                        "source_name": "news",
                        "snippet": "dienogest comparison",
                        "published_date": "2026-10-01",
                    },
                    {
                        "title": "Dienogest trial",
                        "url": "https://pubmed/1/",
                        "snippet": "dup",
                    },
                    {"title": "Unrelated", "url": "https://x/3", "snippet": "nothing"},
                ]
            )
        }
    ]
    deps = {
        "load_docs_df": lambda: df,
        "load_sqlite_vectors": lambda: (["u1"], np.ones((1, 3))),
        "embed_text": lambda q: np.ones(3),
        "cosine_search": lambda vec, uids, mat, k: [("u1", 0.91)],
        "load_ai_runs": lambda n: runs,
    }
    res = library_search("new dienogest studies", deps, 5)
    titles = [(r["title"], r["found_in"]) for r in res]
    assert titles == [
        ("Dienogest trial", "evidence library"),
        ("Dienogest vs GnRH", "Intelligence Briefing results"),
    ]
    assert len(res[0]["summary"]) <= 400

    # library unavailable: briefings still searched, with a note
    deps["load_sqlite_vectors"] = lambda: (_ for _ in ()).throw(
        RuntimeError("no vectors")
    )
    res = library_search("dienogest", deps, 5)
    assert isinstance(res[0], str) and "unavailable" in res[0]
