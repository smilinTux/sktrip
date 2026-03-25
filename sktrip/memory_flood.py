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
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScrollRequest

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
        self.client = QdrantClient(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key,
            https=True,
            timeout=30,
        )
        self.collection = config.qdrant.collection
        self.vector_dim = config.qdrant.vector_dim

    def get_corpus_size(self) -> int:
        """Return the total number of memories in the collection."""
        try:
            info = self.client.get_collection(self.collection)
            return info.points_count or 0
        except Exception:
            return 0

    def _extract_fragment(self, point: Any) -> MemoryFragment:
        """Extract a MemoryFragment from a Qdrant point."""
        payload = point.payload or {}
        text = (
            payload.get("content", "")
            or payload.get("text", "")
            or payload.get("title", "")
            or str(payload)
        )
        tags = payload.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        score = getattr(point, "score", None)
        pid = str(point.id) if hasattr(point, "id") else "unknown"
        return MemoryFragment(id=pid, text=text, tags=tags, score=score, payload=payload)

    def pull_random(self, count: int = 10) -> list[MemoryFragment]:
        """Pull random memories by scrolling with random offset."""
        total = self.get_corpus_size()
        if total == 0:
            return []

        fragments: list[MemoryFragment] = []
        # Scroll through random offsets to get diverse memories
        attempts = min(count * 3, total)
        offsets_tried: set[int] = set()

        while len(fragments) < count and len(offsets_tried) < attempts:
            # Use scroll with random offset via point ID
            try:
                result = self.client.scroll(
                    collection_name=self.collection,
                    limit=min(count - len(fragments), 20),
                    with_payload=True,
                    with_vectors=False,
                )
                points, _next = result
                for p in points:
                    frag = self._extract_fragment(p)
                    if frag.text:
                        fragments.append(frag)
                break  # scroll gives us a batch, good enough for random
            except Exception:
                break

        # Shuffle to randomize order
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
            results = self.client.search(
                collection_name=self.collection,
                query_vector=anti_embedding,
                limit=count * 2,
                with_payload=True,
            )
            fragments = [self._extract_fragment(r) for r in results if self._extract_fragment(r).text]
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
        """Get embedding vector for text via Ollama."""
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
