from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import migrate_experiment6_retriever_runtime_profiles as migration


class RetrieverRuntimeProfileMigrationTests(unittest.TestCase):
    def test_consistent_batch_evidence_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            log.write_text("command=python infer.py --batch-size 4 --max-new-tokens 128\n", encoding="utf-8")
            batch, evidence = migration.infer_batch_size(
                {"batch_size": 4, "log": str(log)},
                [{"batchSize": 4}],
                1,
            )
        self.assertEqual(batch, 4)
        self.assertEqual(len(evidence), 3)

    def test_config_default_is_used_only_without_artifact_evidence(self) -> None:
        batch, evidence = migration.infer_batch_size({}, [{}], 1)
        self.assertEqual(batch, 1)
        self.assertEqual(evidence, ["formal config default"])

    def test_disagreeing_batch_evidence_is_rejected(self) -> None:
        with self.assertRaises(migration.MigrationError):
            migration.infer_batch_size(
                {"batch_size": 1},
                [{"batchSize": 4}],
                1,
            )


    def test_missing_converter_run_is_backfilled(self) -> None:
        candidates = [{
            "index": 0,
            "source": "Econ_002",
            "seed": 2026073101,
            "candidateSha256": "abc",
        }]
        converters = [{
            "index": 0,
            "source": "Econ_002",
            "seed": 2026073101,
            "candidateSha256": "abc",
        }]
        migration.backfill_converter_runs(converters, candidates, 1)
        self.assertEqual(converters[0]["run"], 1)

    def test_wrong_converter_run_is_rejected(self) -> None:
        candidates = [{
            "index": 0,
            "source": "Econ_002",
            "seed": 2026073101,
            "candidateSha256": "abc",
        }]
        converters = [{
            "index": 0,
            "source": "Econ_002",
            "seed": 2026073101,
            "candidateSha256": "abc",
            "run": 2,
        }]
        with self.assertRaises(migration.MigrationError):
            migration.backfill_converter_runs(converters, candidates, 1)


    def test_converter_seed_uses_run_seed_plus_row_index(self) -> None:
        candidates = [
            {"index": 0, "source": "A", "seed": 100, "candidateSha256": "a"},
            {"index": 1, "source": "B", "seed": 100, "candidateSha256": "b"},
        ]
        converters = [
            {"index": 0, "source": "A", "seed": 100, "candidateSha256": "a"},
            {"index": 1, "source": "B", "seed": 101, "candidateSha256": "b"},
        ]
        migration.backfill_converter_runs(converters, candidates, 1)
        self.assertEqual([item["run"] for item in converters], [1, 1])


if __name__ == "__main__":
    unittest.main()
