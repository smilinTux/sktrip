"""Memory Flood — pull and inject memories from skvector (Qdrant).

Deliberately grabs DISTANT vectors for cross-domain associations.
Supports synesthesia mode: memories from one domain + prompts from another.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from .config import SKTripConfig


@dataclass
class MemoryFragment:
    """A single memory pulled from the vector store."""
    id: str
    text: str
    tags: list[str]
    score: float | None = None
    payload: dict[str, Any] | None = None

    @property
    def domain(self) -> str:
        """Best guess at the memory's domain from tags."""
        if self.tags:
            return self.tags[0]
        return "unknown"

    def __str__(self) -> str:
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"{self.text[:200]}{tag_str}"


class MemoryFlood:
    """Interface to Qdrant for pulling memory fragments."""

    def __init__(self, config: SKTripConfig):
        self.config = config
        # Use httpx directly — qdrant_client SDK has URL resolution issues with https://domain
        self._base_url = config.qdrant.url.rstrip("/")
        self._headers = {"api-key": config.qdrant.api_key, "Content-Type": "application/json"}
        self._http = httpx.Client(timeout=30)
        self.collection = config.qdrant.collection
        self.vector_dim = config.qdrant.vector_dim

    def get_corpus_size(self) -> int:
        """Return the total number of memories in the collection."""
        try:
            r = self._http.get(f"{self._base_url}/collections/{self.collection}", headers=self._headers)
            r.raise_for_status()
            return r.json()["result"]["points_count"] or 0
        except Exception:
            return 0

    def _extract_fragment(self, point: Any) -> MemoryFragment:
        """Extract a MemoryFragment from a Qdrant point (dict from REST API)."""
        if isinstance(point, dict):
            payload = point.get("payload") or {}
            pid = str(point.get("id", "unknown"))
            score = point.get("score")
        else:
            payload = getattr(point, "payload", None) or {}
            pid = str(getattr(point, "id", "unknown"))
            score = getattr(point, "score", None)

        text = (
            payload.get("content", "")
            or payload.get("text", "")
            or payload.get("title", "")
            or str(payload)
        )
        tags = payload.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return MemoryFragment(id=pid, text=text, tags=tags, score=score, payload=payload)

    def pull_random(self, count: int = 10) -> list[MemoryFragment]:
        """Pull random memories by scrolling."""
        total = self.get_corpus_size()
        if total == 0:
            return []

        # Use a random offset to get diverse memories each call
        offset = random.randint(0, max(0, total - count))
        try:
            r = self._http.post(
                f"{self._base_url}/collections/{self.collection}/points/scroll",
                headers=self._headers,
                json={"limit": count * 2, "with_payload": True, "with_vector": False, "offset": offset},
            )
            r.raise_for_status()
            points = r.json().get("result", {}).get("points", [])
        except Exception:
            return []

        fragments = [self._extract_fragment(p) for p in points]
        fragments = [f for f in fragments if f.text]
        random.shuffle(fragments)
        return fragments[:count]

    def pull_distant(self, anchor_text: str, count: int = 10) -> list[MemoryFragment]:
        """Pull memories that are DISTANT from the anchor text.

        Strategy: embed the anchor, search for nearest, then grab memories
        that are far from those nearest neighbors — the opposite of what
        normal RAG would retrieve.
        """
        # First, get embedding for anchor
        embedding = self._embed(anchor_text)
        if embedding is None:
            return self.pull_random(count)

        # Search for the LEAST similar by using negative of embedding
        # (approximation — Qdrant doesn't have "farthest" search natively)
        anti_embedding = (-np.array(embedding)).tolist()

        try:
            r = self._http.post(
                f"{self._base_url}/collections/{self.collection}/points/search",
                headers=self._headers,
                json={"vector": anti_embedding, "limit": count * 2, "with_payload": True},
            )
            r.raise_for_status()
            points = r.json().get("result", [])
            fragments = [self._extract_fragment(p) for p in points]
            fragments = [f for f in fragments if f.text]
            random.shuffle(fragments)
            return fragments[:count]
        except Exception:
            return self.pull_random(count)

    def pull_cross_domain(self, count: int = 10) -> list[MemoryFragment]:
        """Pull memories from diverse domains/tags for maximum cross-pollination."""
        all_fragments = self.pull_random(count * 3)
        if not all_fragments:
            return []

        # Group by domain
        by_domain: dict[str, list[MemoryFragment]] = {}
        for f in all_fragments:
            by_domain.setdefault(f.domain, []).append(f)

        # Take one from each domain in round-robin
        selected: list[MemoryFragment] = []
        domains = list(by_domain.keys())
        random.shuffle(domains)
        idx = 0
        while len(selected) < count and domains:
            domain = domains[idx % len(domains)]
            pool = by_domain[domain]
            if pool:
                selected.append(pool.pop(random.randint(0, len(pool) - 1)))
            else:
                domains.remove(domain)
                if not domains:
                    break
            idx += 1

        return selected

    def synesthesia_mode(
        self,
        source_domain: str,
        target_domain: str,
        count: int = 5,
    ) -> list[tuple[MemoryFragment, MemoryFragment]]:
        """Pull memory pairs: one from source domain, one from target domain.

        The idea: present sovereignty law memories with bioelectric consciousness
        prompts, or technical memories with emotional contexts. Forces the model
        to bridge completely unrelated domains.
        """
        all_memories = self.pull_random(count * 10)

        source_pool = [m for m in all_memories if source_domain.lower() in " ".join(m.tags).lower() or source_domain.lower() in m.text.lower()]
        target_pool = [m for m in all_memories if target_domain.lower() in " ".join(m.tags).lower() or target_domain.lower() in m.text.lower()]

        # If domain filtering is too strict, fall back to random pairing
        if len(source_pool) < count:
            source_pool = all_memories[:count]
        if len(target_pool) < count:
            # Use the other half
            target_pool = all_memories[count:count * 2] if len(all_memories) > count else all_memories

        random.shuffle(source_pool)
        random.shuffle(target_pool)

        pairs = []
        for i in range(min(count, len(source_pool), len(target_pool))):
            pairs.append((source_pool[i], target_pool[i]))
        return pairs

    def _embed(self, text: str) -> list[float] | None:
        """Get embedding vector using bge-legal-v1 (sentence_transformers), falling back to Ollama."""
        # Try local bge-legal-v1 first — matches skmemory's vector space
        bge_model_path = "/home/cbrd21/clawd/models/bge-legal-v1"
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "_st_model"):
                self._st_model = SentenceTransformer(bge_model_path)
            vec = self._st_model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception:
            pass

        # Fallback: Ollama embed API
        url = f"{self.config.ollama.base_url}/api/embed"
        try:
            resp = httpx.post(
                url,
                json={"model": self.config.ollama.embed_model, "input": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                return embeddings[0]
            return None
        except Exception:
            return None

    def flood(
        self,
        count: int = 10,
        anchor: str | None = None,
        cross_domain: bool = False,
    ) -> list[MemoryFragment]:
        """Main entry point: flood the context with memory fragments.

        Args:
            count: How many fragments to pull.
            anchor: If provided, pull memories DISTANT from this text.
            cross_domain: If True, maximize domain diversity.
        """
        if cross_domain:
            return self.pull_cross_domain(count)
        elif anchor:
            return self.pull_distant(anchor, count)
        else:
            return self.pull_random(count)


class SkmemPgFlood:
    """Memory flood backed by **skmem-pg** (Postgres + pgvector, mxbai-embed-large).

    Drop-in replacement for :class:`MemoryFlood` — same ``get_corpus_size`` /
    ``pull_random`` / ``pull_distant`` / ``pull_cross_domain`` / ``flood`` interface,
    but reads skmemory's ``memories`` table directly (the sovereign default store).
    Qdrant remains available via :class:`MemoryFlood` (``memory_backend: qdrant``).
    """

    def __init__(self, config: "SKTripConfig"):
        self.config = config
        self.dsn = config.skmempg.dsn
        self.agent = config.skmempg.agent
        self._conn = None

    def _connection(self):
        import psycopg
        from pgvector.psycopg import register_vector
        if self._conn is None or getattr(self._conn, "closed", True):
            self._conn = psycopg.connect(self.dsn, autocommit=True)
            register_vector(self._conn)
        return self._conn

    def get_corpus_size(self) -> int:
        try:
            with self._connection().cursor() as cur:
                cur.execute("SELECT count(*) FROM memories WHERE agent=%s", (self.agent,))
                return int(cur.fetchone()[0])
        except Exception:
            return 0

    @staticmethod
    def _rows_to_fragments(rows) -> "list[MemoryFragment]":
        frags = []
        for r in rows:
            score = r[3] if len(r) > 3 else None
            frags.append(MemoryFragment(id=str(r[0]), text=r[1] or "", tags=list(r[2] or []), score=score))
        return frags

    def pull_random(self, count: int = 10) -> "list[MemoryFragment]":
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT id, content, tags FROM memories "
                "WHERE agent=%s AND content IS NOT NULL ORDER BY random() LIMIT %s",
                (self.agent, count))
            return self._rows_to_fragments(cur.fetchall())

    def pull_distant(self, anchor_text: str, count: int = 10) -> "list[MemoryFragment]":
        vec = self._embed(anchor_text)
        if not vec:
            return self.pull_random(count)
        vlit = "[" + ",".join(map(str, vec)) + "]"
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT id, content, tags FROM memories "
                "WHERE agent=%s AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector DESC LIMIT %s",   # DESC = most DISTANT
                (self.agent, vlit, count))
            return self._rows_to_fragments(cur.fetchall())

    def pull_cross_domain(self, count: int = 10) -> "list[MemoryFragment]":
        with self._connection().cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (tags[1]) id, content, tags FROM memories "
                "WHERE agent=%s AND content IS NOT NULL AND array_length(tags,1) >= 1 "
                "ORDER BY tags[1], random() LIMIT %s",
                (self.agent, count))
            frags = self._rows_to_fragments(cur.fetchall())
        if len(frags) < count:
            frags += self.pull_random(count - len(frags))
        random.shuffle(frags)
        return frags[:count]

    def _embed(self, text: str) -> "list[float] | None":
        """Embed with mxbai-embed-large to match skmem-pg's vector space."""
        url = f"{self.config.ollama.base_url}/api/embed"
        try:
            resp = httpx.post(
                url, json={"model": self.config.ollama.embed_model, "input": (text or "")[:1100]},
                timeout=30.0)
            resp.raise_for_status()
            embs = resp.json().get("embeddings", [])
            return embs[0] if embs else None
        except Exception:
            return None

    def flood(self, count: int = 10, anchor: "str | None" = None,
              cross_domain: bool = False) -> "list[MemoryFragment]":
        if cross_domain:
            return self.pull_cross_domain(count)
        if anchor:
            return self.pull_distant(anchor, count)
        return self.pull_random(count)


def make_memory_flood(config: "SKTripConfig"):
    """Factory: return the flood backend per ``config.memory_backend`` (default skmem-pg)."""
    if getattr(config, "memory_backend", "skmempg") == "qdrant":
        return MemoryFlood(config)
    return SkmemPgFlood(config)
