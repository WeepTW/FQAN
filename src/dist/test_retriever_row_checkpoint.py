from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import retriever_row_checkpoint as checkpoint


class RetrieverRowCheckpointTests(unittest.TestCase):
    def test_atomic_round_trip_preserves_sparse_rows(self) -> None:
        hashes = [checkpoint.text_sha256("a"), checkpoint.text_sha256("b")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.checkpoint.jsonl"
            checkpoint.write_checkpoint(
                path,
                [{"index": 0, "inputSha256": hashes[0], "value": "x"}, None],
            )
            loaded = checkpoint.load_checkpoint(path, hashes)
        self.assertEqual(loaded[0]["value"], "x")
        self.assertIsNone(loaded[1])

    def test_input_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.checkpoint.jsonl"
            checkpoint.write_checkpoint(
                path,
                [{"index": 0, "inputSha256": checkpoint.text_sha256("old")}],
            )
            with self.assertRaises(RuntimeError):
                checkpoint.load_checkpoint(path, [checkpoint.text_sha256("new")])

    def test_duplicate_index_is_rejected(self) -> None:
        digest = checkpoint.text_sha256("a")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.checkpoint.jsonl"
            path.write_text(
                '{"index":0,"inputSha256":"' + digest + '"}\n'
                '{"index":0,"inputSha256":"' + digest + '"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                checkpoint.load_checkpoint(path, [digest])


if __name__ == "__main__":
    unittest.main()
