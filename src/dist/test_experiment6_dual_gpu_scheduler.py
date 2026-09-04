#!/usr/bin/env python3
"""Regression tests for the corrected-12 dual-GPU scheduler."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import experiment6_dual_gpu_scheduler as scheduler


class DualGpuSchedulerTests(unittest.TestCase):
    def test_lpt_balances_jobs_and_retry_uses_alternate_device(self) -> None:
        jobs = [
            scheduler.Job("m1", "mistral", 1, 100.0),
            scheduler.Job("m2", "mistral", 1, 90.0),
            scheduler.Job("f1", "flan", 1, 10.0),
            scheduler.Job("f2", "flan", 1, 9.0),
        ]
        assignments = scheduler.partition_lpt(jobs, ["0", "1"])
        self.assertEqual({job.output_id for job in assignments["0"] + assignments["1"]}, {"m1", "m2", "f1", "f2"})
        self.assertTrue(assignments["0"])
        self.assertTrue(assignments["1"])

        retry = scheduler.Job("m1", "mistral", 1, 100.0, previous_device="0")
        retry_assignment = scheduler.partition_lpt([retry], ["0", "1"])
        self.assertEqual(retry_assignment["0"], [])
        self.assertEqual(retry_assignment["1"], [retry])

    def test_job_command_records_device_and_fresh_flag(self) -> None:
        job = scheduler.Job("case", "flan", 3, 1.0)
        command = scheduler.job_command(Path("/tmp/root"), job, "1", True)
        self.assertEqual(command[command.index("--run") + 1], "3")
        self.assertEqual(command[command.index("--cuda-visible-devices") + 1], "1")
        self.assertIn("--no-resume", command)
        resumed = scheduler.job_command(Path("/tmp/root"), job, "0", False)
        self.assertNotIn("--no-resume", resumed)

    def test_one_worker_per_device_never_overlaps_on_same_device(self) -> None:
        jobs = {
            "0": [scheduler.Job("a", "flan", 1, 1.0), scheduler.Job("b", "flan", 1, 1.0)],
            "1": [scheduler.Job("c", "flan", 1, 1.0), scheduler.Job("d", "flan", 1, 1.0)],
        }
        counts = {"0": 0, "1": 0}
        maxima = {"0": 0, "1": 0}
        lock = threading.Lock()

        def fake_execute(root, job, device, fresh, attempt, active, active_lock):
            del root, fresh, attempt, active, active_lock
            with lock:
                counts[device] += 1
                maxima[device] = max(maxima[device], counts[device])
            time.sleep(0.02)
            with lock:
                counts[device] -= 1
            return {"job": job, "device": device, "valid": True, "runtimeSeconds": 0.02}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            scheduler, "execute_job", side_effect=fake_execute
        ):
            results = scheduler.run_assignments(
                Path(directory),
                jobs,
                fresh=True,
                attempt=1,
                heartbeat_seconds=1,
                stall_seconds=60,
            )
        self.assertEqual(len(results), 4)
        self.assertEqual(maxima, {"0": 1, "1": 1})

    def test_missing_manifest_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(
                scheduler.valid_completed_manifest(Path(directory), "case", 1)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
