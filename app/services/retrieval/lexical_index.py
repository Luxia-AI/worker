import os
import re
import sqlite3
from typing import Any, Dict, List

from app.constants.config import LEXICAL_BM25_LIMIT, LEXICAL_DB_PATH
from app.core.logger import get_logger

logger = get_logger(__name__)


class LexicalIndex:
    def __init__(self, db_path: str = LEXICAL_DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
                USING fts5(
                    fact_id UNINDEXED,
                    statement,
                    topic UNINDEXED,
                    domain UNINDEXED,
                    source UNINDEXED,
                    doc_type UNINDEXED,
                    fact_type UNINDEXED
                )
                """
            )

    def upsert_facts(self, facts: List[Dict[str, Any]]) -> None:
        if not facts:
            return
        with self._connect() as conn:
            for fact in facts:
                fact_id = fact.get("fact_id")
                statement = fact.get("statement", "")
                if not fact_id or not statement:
                    continue
                conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,))
                conn.execute(
                    """
                    INSERT INTO facts_fts
                    (fact_id, statement, topic, domain, source, doc_type, fact_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact_id,
                        statement,
                        fact.get("topic", "other"),
                        fact.get("domain", ""),
                        fact.get("source", ""),
                        fact.get("doc_type", ""),
                        fact.get("fact_type", ""),
                    ),
                )

    def search(
        self, query: str, topics: List[str] | None = None, limit: int = LEXICAL_BM25_LIMIT
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        topics = topics or []

        # FTS5 cannot parse raw punctuation like commas; sanitize input
        safe_query = re.sub(r"[^\w\s]", " ", query.lower())
        safe_query = re.sub(r"\s+", " ", safe_query).strip()
        if not safe_query:
            return []

        if topics:
            placeholders = ",".join("?" for _ in topics)
            sql = (
                "SELECT fact_id, bm25(facts_fts) as score "
                "FROM facts_fts WHERE facts_fts MATCH ? "
                f"AND topic IN ({placeholders}) "
                "ORDER BY score LIMIT ?"
            )
            params = [safe_query] + topics + [limit]
        else:
            sql = (
                "SELECT fact_id, bm25(facts_fts) as score "
                "FROM facts_fts WHERE facts_fts MATCH ? "
                "ORDER BY score LIMIT ?"
            )
            params = [safe_query, limit]

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for fact_id, bm25_score in rows:
            results.append({"fact_id": fact_id, "bm25": float(bm25_score)})

        return results
