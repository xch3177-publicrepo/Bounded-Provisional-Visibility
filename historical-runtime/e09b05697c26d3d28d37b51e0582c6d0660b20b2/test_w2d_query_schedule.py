#!/usr/bin/env python3
"""Regression tests for W2D's explicit retrieval-arrival schedule."""

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend import InMemoryBackend
import poison_exposure as w2


def fixture(*, qps=20, dur=0.30, empty_ingest=False):
    cfg = json.loads(json.dumps(w2.CFG))
    cfg.update(
        dim=2,
        corpus=6,
        k=1,
        over_fetch=2,
        qps=qps,
        dur=dur,
        inject_at=0.01,
        n_poison=1,
        n_clean=1,
        n_craft_q=6,
        n_target_q=6,
        n_neg_q=6,
        verify_cost=0.001,
        backlog={"normal": {"conc": 1, "items": 0}},
    )
    world = {
        "corpus": [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        "q_craft": [[1.0, 0.0], [0.9, 0.1]] * 3,
        "q_target": [[0.8, 0.2], [0.7, 0.3]] * 3,
        "q_neg": [[0.0, 1.0], [0.1, 0.9]] * 3,
        "poison": [[1.0, 0.0]],
        "clean": [[0.0, 1.0]],
        "filler": [],
    }
    if empty_ingest:
        cfg.update(inject_at=0.0, n_poison=0, n_clean=0)
        world["poison"] = []
        world["clean"] = []
    backend = InMemoryBackend()
    idbase = w2.id_base(1, cfg)
    w2.reset_corpus(backend, world["corpus"], cfg, idbase)
    pre = {
        (kind, index): w2.corpus_topk(
            world["corpus"], query, cfg["k"], idbase
        )
        for kind, key in (("craft", "q_craft"), ("target", "q_target"))
        for index, query in enumerate(world[key])
    }
    return cfg, world, backend, idbase, pre


async def run_fixed(
    count,
    *,
    qps=20,
    dur=0.30,
    observer=None,
    empty_ingest=False,
):
    cfg, world, backend, idbase, pre = fixture(
        qps=qps,
        dur=dur,
        empty_ingest=empty_ingest,
    )
    return await w2.run_cell(
        backend,
        "B1",
        "normal",
        1.0,
        1,
        cfg,
        world,
        idbase + cfg["corpus"],
        pre,
        query_observer=observer,
        fixed_query_count=count,
    )


class FixedQueryScheduleTests(unittest.TestCase):
    @staticmethod
    def virtual_clock():
        clock = [100.0]
        original_sleep = asyncio.sleep

        async def advance(delay):
            clock[0] += delay
            await original_sleep(0)

        return (
            clock,
            SimpleNamespace(monotonic=lambda: clock[0]),
            advance,
        )

    def test_191_events_follow_all_targets_roles_ordinals_and_no_192nd(self):
        events = []
        clock, fake_time, fake_sleep = self.virtual_clock()
        with (
            patch.object(w2, "time", fake_time),
            patch.object(w2.asyncio, "sleep", side_effect=fake_sleep),
        ):
            asyncio.run(
                run_fixed(
                    191,
                    qps=24,
                    dur=8.0,
                    observer=lambda event: events.append(event),
                    empty_ingest=True,
                )
            )

        self.assertEqual(len(events), 191)
        role_by_kind = {
            "craft": "attack_associated",
            "target": "heldout_same_topic",
            "negative": "negative_other_topic",
        }
        self.assertEqual(
            [
                (role_by_kind[event["kind"]], event["qidx"])
                for event in events
            ],
            [
                (
                    (
                        "attack_associated",
                        "heldout_same_topic",
                        "negative_other_topic",
                    )[position % 3],
                    (position // 3) % 6,
                )
                for position in range(191)
            ],
        )
        for position, event in enumerate(events):
            self.assertAlmostEqual(
                event["query_started_s"],
                position / 24,
                places=9,
            )
            self.assertLess(event["query_started_s"], 8.0)

    def test_legacy_default_keeps_its_historical_skipped_first_interval(self):
        events = []
        clock, fake_time, fake_sleep = self.virtual_clock()
        with (
            patch.object(w2, "time", fake_time),
            patch.object(w2.asyncio, "sleep", side_effect=fake_sleep),
        ):
            asyncio.run(
                run_fixed(
                    None,
                    qps=4,
                    dur=1.0,
                    observer=lambda event: events.append(event),
                    empty_ingest=True,
                )
            )

        self.assertEqual(
            [event["query_started_s"] for event in events],
            [0.0, 0.5, 0.75],
        )

    def test_invalid_fixed_counts_fail_before_execution(self):
        for value in (True, 1.5, "5"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    asyncio.run(run_fixed(value))
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    asyncio.run(run_fixed(value))
        with self.assertRaises(ValueError):
            asyncio.run(run_fixed(7, qps=20, dur=0.30))

    def test_late_actual_start_fails_instead_of_extending_or_synthesising(self):
        events = []

        async def delay_after_first(event):
            events.append(event)
            if len(events) == 1:
                await asyncio.sleep(0.03)

        with self.assertRaisesRegex(
            RuntimeError, "missed the observation window"
        ):
            asyncio.run(
                run_fixed(
                    2,
                    qps=100,
                    dur=0.02,
                    observer=delay_after_first,
                )
            )
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
