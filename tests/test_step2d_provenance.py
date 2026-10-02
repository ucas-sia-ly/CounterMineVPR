"""Commit metadata may drift; scientific bytes and unapproved code may not."""
from pathlib import Path
import unittest

from countermine.mining.full_structural_provenance import (
    PROVENANCE_SOURCE, READER_REPAIRS, code_provenance_matches,
    scientific_provenance_matches,
)
from countermine.mining.structural_analysis import sha256_file


class ProvenanceTests(unittest.TestCase):
    def test_only_producer_commit_is_descriptive(self):
        saved = {"git_commit": "producer", "manifest_sha256": "manifest",
                 "input_file_sha256": {"source.py": "source"}, "real_rgb_only": True}
        current = dict(saved, git_commit="reader")
        self.assertTrue(scientific_provenance_matches(saved, current))
        self.assertEqual(saved["git_commit"], "producer")
        for changed in (dict(current, manifest_sha256="drift"),
                        dict(current, input_file_sha256={"source.py": "drift"}),
                        dict(current, real_rgb_only=False),
                        dict(current, unexpected="new field")):
            with self.subTest(changed=changed):
                self.assertFalse(scientific_provenance_matches(saved, changed))

    def test_known_reader_repairs_only_and_actual_sources_match_approval(self):
        root = Path(__file__).resolve().parents[1]
        saved = {path: pair[0] for path, pair in READER_REPAIRS.items()}
        current = {path: sha256_file(root / path) for path in READER_REPAIRS}
        self.assertEqual(current, {path: pair[1] for path, pair in READER_REPAIRS.items()})
        current[PROVENANCE_SOURCE] = sha256_file(root / PROVENANCE_SOURCE)
        saved["unchanged.py"] = current["unchanged.py"] = "unchanged"
        self.assertTrue(code_provenance_matches(saved, current))
        self.assertTrue(code_provenance_matches(current, current))
        for path in (*READER_REPAIRS, "unchanged.py"):
            with self.subTest(path=path):
                self.assertFalse(code_provenance_matches(saved, dict(current, **{path: "unapproved"})))
                self.assertFalse(code_provenance_matches(dict(saved, **{path: "unknown producer"}), current))
        self.assertFalse(code_provenance_matches(saved, dict(current, added="unknown")))
        self.assertFalse(code_provenance_matches(saved, {key: value for key, value in current.items()
                                                       if key != "unchanged.py"}))
        modified_helper = dict(current, **{PROVENANCE_SOURCE: "modified"})
        self.assertFalse(code_provenance_matches(current, modified_helper))


if __name__ == "__main__":
    unittest.main()
