"""Fixture-only full exporter exercises real contact sheets and publication."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_full_graph_visuals as visual_fixtures
from countermine.mining.full_graph_analysis import ANALYSIS_SOURCE_FILES, export_full_audit
from countermine.mining.structural_analysis import sha256_file


class FullExporterTests(unittest.TestCase):
    def fixture(self, root):
        tables, manifest, dataset, bank = visual_fixtures.FullVisualTests().prepare(root)
        runtime = bank.parent
        for name in ("full_population.csv", "full_candidate_structural_metrics.csv", "full_calibration_summary.json",
                     "full_population_summary.json", "full_measurement_summary.json", "full_graph_summary.json",
                     "image_nodes.csv", "image_edges.csv", "place_nodes.csv", "place_edges.csv"):
            (runtime / name).write_text("project scalar evidence", encoding="utf-8")
        (bank / "summary.json").write_text('{"image_count":5}', encoding="utf-8")
        raw = root / "cache/gsv_mini/rgb_candidates_raw.csv"
        raw.write_text("frozen SALAD candidates", encoding="utf-8")
        for source in ANALYSIS_SOURCE_FILES:
            path = root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("project analysis source", encoding="utf-8")
        audit_dir = root / "docs/audits"
        audit_dir.mkdir(parents=True)
        random = {"count": 4999, "q95_fraction": .022, "q99_fraction": .0038}
        (audit_dir / "step2c_null_topology_metrics.json").write_text(json.dumps({"joint_null": {"rates": {
            name: {"random": random} for name in ("all", "same_city", "cross_city")}}}))
        replay = {"passed": True, "pair_count": 5000, "maximum_metric_differences": {},
                  "q95_count": 352, "q99_count": 77}
        calibration = {"pilot_reproduction": replay, "provenance": {"code_sha256": {"calibration.py": "hash"}}}
        measurement = {"shard_count": 1, "completed_shard_count": 1, "runtime_throughput_summary": {
            "pairs_processed": 4, "pairs_per_second": 1}, "configuration": {
                "feature_bank_directory": bank.relative_to(root).as_posix()}}
        graph = {"graph_code_hashes": {"graph.py": "hash"}, "artifacts": {
            name: {"sha256": sha256_file(runtime / name), "count": len(table)}
            for name, table in zip(("image_nodes.csv", "image_edges.csv", "place_nodes.csv", "place_edges.csv"), tables)}}
        source_hashes = {path: sha256_file(path) for path in (manifest, raw, audit_dir / "step2c_null_topology_metrics.json")}
        return tables, runtime, dataset, replay, calibration, measurement, graph, source_hashes

    def mocks(self, stack, fixture, *, replay_error=False):
        tables, runtime, dataset, replay, calibration, measurement, graph, _ = fixture
        stack.enter_context(patch("countermine.mining.full_structural_population.load_full_population", return_value=(tables[1], {"population": {"full_canonical_edge_count": 4}})))
        stack.enter_context(patch("countermine.mining.full_structural_measurement.load_full_metrics", return_value=(tables[1], measurement)))
        stack.enter_context(patch("countermine.mining.full_structural_calibration.load_full_calibrated", return_value=(tables[1], calibration)))
        stack.enter_context(patch("countermine.mining.full_structural_calibration.validate_pilot_reproduction",
                                  side_effect=ValueError("pilot metrics differ") if replay_error else None, return_value=replay))
        stack.enter_context(patch("countermine.mining.full_countermine_graph.load_full_graph", return_value=(*tables, graph)))
        stack.enter_context(patch("countermine.mining.full_graph_analysis.frozen_provenance", return_value={
            "step2a_snapshot_sha256": "hash-a", "step2b_snapshot_sha256": "hash-b", "step2c_snapshot_sha256": "hash-c"}))
        stack.enter_context(patch("countermine.mining.full_graph_analysis.current_git_commit", return_value="commit"))

    def test_full_export_publishes_finite_snapshot_ten_hashed_figures_and_audits(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            fixture = self.fixture(root)
            self.mocks(stack, fixture)
            snapshot = export_full_audit(fixture[1], dataset_root=fixture[2], repo_root=root)
            target = root / "docs/audits/step2d_full_countermine_metrics.json"
            self.assertEqual(snapshot, json.loads(target.read_text()))
            self.assertEqual(snapshot["pilot_reproduction"]["pair_count"], 5000)
            self.assertEqual(snapshot["population"]["canonical_edge_count"], 4)
            hashes = snapshot["visualization"]["figure_sha256"]
            self.assertEqual(len(hashes), 10)
            self.assertTrue(all(sha256_file(root / "docs/audits" / name) == digest for name, digest in hashes.items()))
            self.assertTrue((fixture[1] / "image_node_audit.csv").is_file())
            self.assertTrue((fixture[1] / "full_place_components.json").is_file())
            self.assertTrue(all(sha256_file(path) == digest for path, digest in fixture[-1].items()))
            self.assertNotIn(str(root), json.dumps(snapshot, allow_nan=False))

    def test_actual_pilot_drift_stops_before_any_plot_or_publication(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            fixture = self.fixture(root)
            self.mocks(stack, fixture, replay_error=True)
            with self.assertRaisesRegex(ValueError, "pilot metrics differ"):
                export_full_audit(fixture[1], dataset_root=fixture[2], repo_root=root)
            self.assertFalse((root / "outputs/step2d").exists())
            self.assertFalse((root / "docs/audits/step2d_full_countermine_metrics.json").exists())

    def test_historical_snapshot_destination_rejected_before_input_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "step2d_"):
                export_full_audit(root / "cache/countermine_rgb/step2d",
                    snapshot=root / "docs/audits/step2c_null_topology_metrics.json", repo_root=root)
            self.assertFalse((root / "outputs/step2d").exists())


if __name__ == "__main__":
    unittest.main()
