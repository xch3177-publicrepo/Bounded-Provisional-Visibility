"""
Gate 2 -- eligibility-mode comparison (RQ-elig, §6.6).

Compares postfilter vs infilter visibility enforcement, and now measures result
QUALITY (Eligible Recall@K vs an exact brute-force ground truth), not just result
COUNT (under-fill). It also varies the HIDDEN DISTRIBUTION, because under-fill is
driven by the LOCAL hidden ratio within a query's retrieved neighborhood, not the
global hidden ratio:
  - random:      hidden items spread uniformly (benign)
  - clustered:   hidden items concentrated in semantic clusters (lineage burst)
  - rank-biased: the items that top the query set are hidden (worst case)

Status: functional validation + local Milvus Lite latency characterization. Real
P99 / throughput / concurrency / delta_query-vs-consistency remain PENDING a full
standalone deployment (Lite latency is a microbenchmark, not production).

Run:  python3 gate2.py --backend inmemory
      ./.venv312/bin/python gate2.py --backend milvus --uri /tmp/g2.db      # Lite
"""

import argparse
import random
import time

from backend import InMemoryBackend, cosine


def pctl(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def make_backend(name, uri, dim):
    if name == "inmemory":
        return InMemoryBackend()
    if name == "milvus":
        from milvus_backend import MilvusBackend
        return MilvusBackend(dim=dim, uri=uri)
    raise SystemExit(f"unknown backend {name}")


def gen_vecs(N, dim):
    return {i: [random.gauss(0, 1) for _ in range(dim)] for i in range(N)}


def topk_eligible(vecs, q, eligible, K):
    return [i for i in sorted(eligible, key=lambda j: cosine(vecs[j], q), reverse=True)[:K]]


def hidden_mask(mode, vecs, N, hf, qvecs, seed=2):
    n = int(hf * N)
    if mode == "random":
        return set(random.Random(seed).sample(range(N), n))
    if mode == "clustered":
        rng = random.Random(seed)
        seeds = [vecs[s] for s in rng.sample(range(N), 5)]
        score = {i: max(cosine(vecs[i], sv) for sv in seeds) for i in range(N)}
        return set(sorted(range(N), key=lambda i: score[i], reverse=True)[:n])
    if mode == "rankbiased":                # deterministic given the query set (worst case)
        score = {i: max(cosine(vecs[i], q) for q in qvecs) for i in range(N)}
        return set(sorted(range(N), key=lambda i: score[i], reverse=True)[:n])
    raise SystemExit(f"unknown hidden mode {mode}")


def evaluate(backend, vecs, qvecs, eligible, hidden, k, over_fetch):
    thr = 1.0 - 1.0 / over_fetch            # under-fill iff pool hidden occupancy > thr
    pf_under, pf_rec, if_rec, hM, fill, cond_rec = [], [], [], [], [], []
    for q in qvecs:
        gt = set(topk_eligible(vecs, q, eligible, k))
        raw = backend.search(q, over_fetch * k, use_index_filter=False)
        h = sum(1 for i in raw if i in hidden) / max(1, len(raw))
        hM.append(h)
        pf = [i for i in raw if i in eligible][:k]
        fill.append(len(pf) / k)
        pf_under.append(1 if len(pf) < k else 0)
        r = len(set(pf) & gt) / k
        pf_rec.append(r)
        if len(pf) == k:                    # recall GIVEN full-k -> isolates ANN error
            cond_rec.append(r)
        inf = backend.search(q, k, use_index_filter=True)
        if_rec.append(len(set(inf) & gt) / k)
    return {"underfill": sum(pf_under) / len(pf_under),
            "exceed": sum(1 for h in hM if h > thr) / len(hM),   # == underfill (exact)
            "hM50": pctl(hM, 0.50), "hM90": pctl(hM, 0.90), "hM95": pctl(hM, 0.95),
            "fill": sum(fill) / len(fill),
            "pf_recall": sum(pf_rec) / len(pf_rec),
            "cond_recall": (sum(cond_rec) / len(cond_rec)) if cond_rec else float("nan"),
            "if_recall": sum(if_rec) / len(if_rec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="inmemory", choices=["inmemory", "milvus"])
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()
    random.seed(0)
    vecs = gen_vecs(args.N, args.dim)
    qvecs = [[random.gauss(0, 1) for _ in range(args.dim)] for _ in range(args.queries)]

    def fresh(hidden):
        b = make_backend(args.backend, args.uri, args.dim)       # fresh collection
        for i in range(args.N):
            b.insert(i, vecs[i], visible=(i not in hidden))
        return b

    print(f"Gate 2 -- eligibility comparison  (backend={args.backend}, N={args.N}, "
          f"dim={args.dim}, k={args.k}, queries={args.queries})")

    print("\n[A] Hidden distribution (30% global, over-fetch 2x, 3 seeds). Under-fill is set")
    print("    by the UPPER TAIL of per-query pool hidden-occupancy h_M crossing thr=50%,")
    print("    not the mean: random & clustered share P50 but clustered's P90/P95 is heavier.")
    print("    exceed==under-fill is a self-consistency check of the threshold mechanism.")
    print(f"  {'mode':>11} {'hM_P50':>7} {'hM_P90':>7} {'hM_P95':>7} {'underfill':>11} "
          f"{'pf_rec':>7} {'cond_rec':>8} {'if_rec':>7}")
    for mode in ["random", "clustered", "rankbiased"]:
        acc = []
        for sd in (2, 3, 4):
            hidden = hidden_mask(mode, vecs, args.N, 0.30, qvecs, seed=sd)
            eligible = set(range(args.N)) - hidden
            acc.append(evaluate(fresh(hidden), vecs, qvecs, eligible, hidden, args.k, over_fetch=2))
        m = {kk: sum(a[kk] for a in acc) / len(acc) for kk in acc[0]}
        rng = (max(a["underfill"] for a in acc) - min(a["underfill"] for a in acc)) / 2
        print(f"  {mode:>11} {m['hM50']*100:6.0f}% {m['hM90']*100:6.0f}% {m['hM95']*100:6.0f}% "
              f"{m['underfill']*100:7.1f}%±{rng*100:2.0f} {m['pf_recall']:7.3f} "
              f"{m['cond_recall']:8.3f} {m['if_recall']:7.3f}")

    print("\n[B] Over-fetch vs rank-biased worst-case (30% hidden). cond_rec (recall | full-k)")
    print("    ~1.0 in-memory (exact) confirms recall loss comes from under-fill, not")
    print("    selection; on Lite, cond_rec<1 would be pure ANN approximation error.")
    print(f"  {'overfetch':>9} {'underfill':>10} {'fill':>6} {'pf_rec':>7} {'cond_rec':>8} {'if_rec':>7}")
    hidden = hidden_mask("rankbiased", vecs, args.N, 0.30, qvecs)
    eligible = set(range(args.N)) - hidden
    for of in [1, 2, 4, 8]:
        r = evaluate(fresh(hidden), vecs, qvecs, eligible, hidden, args.k, over_fetch=of)
        print(f"  {of:8}x {r['underfill']*100:9.1f}% {r['fill']:6.3f} {r['pf_recall']:7.3f} "
              f"{r['cond_recall']:8.3f} {r['if_recall']:7.3f}")

    print("\nRank-biased hiding is an ADVERSARIAL worst-case containment workload (revoked")
    print("items concentrated at high ranks), not an estimate of typical production hidden")
    print("rates. Under-fill/recall are logic properties (in-memory exact + Lite ANN, same")
    print("trend); real P99/throughput/concurrency/delta_query-vs-consistency need standalone.")


if __name__ == "__main__":
    main()
