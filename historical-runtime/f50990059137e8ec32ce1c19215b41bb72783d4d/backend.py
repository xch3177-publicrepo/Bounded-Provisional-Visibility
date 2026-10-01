"""
Backend-independent vector-store interface (§5.1). The control plane
(functional_slice.py) talks only to this interface; the in-memory and Milvus
backends implement it identically so the analysis never depends on the backend.

Two eligibility-enforcement modes are supported at the query layer:
  - post-retrieval authoritative check (query service filters raw ANN candidates
    by the control store)  -> uses search(..., use_index_filter=False)
  - in-index metadata filter (backend filters by a `visible` scalar)
    -> uses search(..., use_index_filter=True) + set_visible(...)
"""

import math
from abc import ABC, abstractmethod


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


class VectorBackend(ABC):
    @abstractmethod
    def insert(self, id, vector, visible=True): ...

    @abstractmethod
    def set_visible(self, id, visible): ...      # in-index mode: update `visible` scalar

    @abstractmethod
    def delete(self, id): ...

    @abstractmethod
    def search(self, qvec, k, use_index_filter=False, consistency=None):
        """Return up to k ids by descending similarity. If use_index_filter,
        only entities with visible==True are eligible (in-index enforcement).
        `consistency` sends a per-request consistency level where the backend
        has one; an in-memory backend is trivially strongly consistent and
        ignores it."""
        ...


class InMemoryBackend(VectorBackend):
    """Correctness/logic backend (brute-force cosine). NOT a performance baseline."""
    def __init__(self):
        self.v = {}
        self.vis = {}

    def insert(self, id, vector, visible=True):
        self.v[id] = list(vector)
        self.vis[id] = visible

    def set_visible(self, id, visible):
        self.vis[id] = visible

    def delete(self, id):
        self.v.pop(id, None)
        self.vis.pop(id, None)

    def search(self, qvec, k, use_index_filter=False, consistency=None):
        # Snapshot first. A real vector store serves reads while writes land;
        # iterating the live dict raises "changed size during iteration" the
        # moment a query runs concurrently with ingestion, which is the only
        # interesting case. A stale-but-consistent view is the right semantics
        # here, and it is what the Milvus backends give too.
        cand = [(cosine(qvec, vec), i) for i, vec in list(self.v.items())
                if (not use_index_filter or self.vis.get(i, False))]
        cand.sort(reverse=True)
        return [i for _, i in cand[:k]]
