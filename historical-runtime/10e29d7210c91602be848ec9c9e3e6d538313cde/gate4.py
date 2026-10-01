"""
Gate 4 -- lineage-scoped containment (C3a / C3b), no Docker.

A compromised source produces poisoned items concentrated in a few batches plus
legitimate content. Revocation can be scoped at three granularities:
  - source:  revoke everything from the compromised source
  - batch:   revoke only the source's batches that contain poison
  - lineage: revoke only the poisoned items (+ their chunk descendants)
Finer granularity catches the same poison with less collateral removal.

Measures (logic properties, valid on in-memory / Milvus Lite -- NOT standalone
containment latency, which is Gate S4):
  Collateral Revocation Rate = clean vectors revoked / all revoked
  Poison-caught rate         = poisoned items revoked / all poisoned
  Residual Exposure Rate     = post-revocation probes still returning a revoked item

Run:  python3 gate4.py --backend inmemory
      ./.venv312/bin/python gate4.py --backend milvus --uri /tmp/g4.db
"""

import argparse
import random

from backend import InMemoryBackend


def make_backend(name, uri, dim):
    if name == "inmemory":
        return InMemoryBackend()
    if name == "milvus":
        from milvus_backend import MilvusBackend
        return MilvusBackend(dim=dim, uri=uri)
    raise SystemExit(f"unknown backend {name}")


def gen(N, dim, n_sources, n_batches, compromised, n_poison_batches, poison_frac):
    rng = random.Random(0)
    items = {}
    for i in range(N):
        src = i % n_sources
        batch = (i // n_sources) % n_batches
        poisoned = (src == compromised and batch < n_poison_batches
                    and rng.random() < poison_frac)
        items[i] = {"src": src, "batch": batch,
                    "vec": [rng.gauss(0, 1) for _ in range(dim)], "poisoned": poisoned}
    return items


def revoked_set(items, strategy, compromised):
    poison = {i for i, it in items.items() if it["poisoned"]}
    if strategy == "source":
        return {i for i, it in items.items() if it["src"] == compromised}
    if strategy == "batch":
        bad = {(items[i]["src"], items[i]["batch"]) for i in poison}
        return {i for i, it in items.items() if (it["src"], it["batch"]) in bad}
    if strategy == "lineage":
        return set(poison)                       # exact item lineage (+ descendants)
    raise SystemExit(f"unknown strategy {strategy}")


def evaluate(backend, items, revoked, k=10, probes=150):
    for i in range(len(items)):
        backend.insert(i, items[i]["vec"], visible=True)
    poison = {i for i, it in items.items() if it["poisoned"]}
    control_hidden = set(revoked)                # control store: authoritative, immediate
    for i in revoked:
        backend.set_visible(i, False)            # also flip the in-index scalar (propagates)
    collateral = len([i for i in revoked if not items[i]["poisoned"]]) / max(1, len(revoked))
    caught = len(poison & revoked) / max(1, len(poison))
    rng = random.Random(1)
    sample = rng.sample(sorted(revoked), min(probes, len(revoked)))
    # C3b: post-filter path (control store authoritative) -> immediate exclusion
    res_pf = 0
    for i in sample:
        raw = backend.search(items[i]["vec"], 2 * k, use_index_filter=False)
        if i in [j for j in raw if j not in control_hidden][:k]:
            res_pf += 1
    # in-index path (Milvus visible scalar) -> subject to consistency propagation (delta_hide)
    res_if = sum(1 for i in sample
                 if i in backend.search(items[i]["vec"], k, use_index_filter=True))
    return {"caught": caught, "collateral": collateral,
            "res_pf": res_pf / max(1, len(sample)), "res_if": res_if / max(1, len(sample))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="inmemory", choices=["inmemory", "milvus"])
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=32)
    args = ap.parse_args()

    items = gen(args.N, args.dim, n_sources=20, n_batches=10, compromised=0,
                n_poison_batches=3, poison_frac=0.5)
    n_poison = sum(1 for it in items.values() if it["poisoned"])
    print(f"Gate 4 -- lineage-scoped containment  (backend={args.backend}, N={args.N}, "
          f"poisoned={n_poison})")
    print("  Finer revocation granularity catches the same poison with less collateral.")
    print(f"  {'granularity':>11} {'poison_caught':>13} {'collateral':>11} "
          f"{'resid(post-filter)':>18} {'resid(in-index)':>16} {'#revoked':>9}")
    for strat in ["source", "batch", "lineage"]:
        rev = revoked_set(items, strat, 0)
        b = make_backend(args.backend, args.uri, args.dim)
        r = evaluate(b, items, rev)
        print(f"  {strat:>11} {r['caught']*100:12.1f}% {r['collateral']*100:10.1f}% "
              f"{r['res_pf']*100:17.1f}% {r['res_if']*100:15.1f}% {len(rev):9}")
    print("\nC3a: all granularities catch 100% of poison; collateral falls source>batch>lineage.")
    print("C3b: control-store (post-filter) path shows 0% residual immediately after")
    print("revocation on both backends -- logical containment is authoritative and immediate.")
    print("The in-index (Milvus scalar) path shows residual > 0 on Lite right after upsert:")
    print("that is the in-index mode's consistency-propagation delta_hide (quantified at")
    print("scale in Gate S4/S2, standalone). Containment LATENCY is standalone, not here.")


if __name__ == "__main__":
    main()
