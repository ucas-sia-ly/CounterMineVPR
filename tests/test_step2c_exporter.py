"""Synthetic exporter integration: protected destinations, hashes and four plots."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from countermine.mining.joint_null import (
    SOURCE_FILES, analyze_joint_null, code_hashes, publish_stage, validate_step2c_snapshot,
)
from countermine.mining.structural_analysis import sha256_file
from countermine.mining.topology_null import FixedGraph, analyze_topology_null
from test_joint_null import joint_fixture
from test_topology_null import frozen_fixture, topology_fixture

SPEC = importlib.util.spec_from_file_location('step2c_exporter', Path(__file__).resolve().parents[1] / 'tools/16_export_step2c_audit.py')
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class ExporterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in SOURCE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('pass\n')
        source = self.root / 'frozen.csv'
        source.write_text('immutable')
        self.provenance = {'real_rgb_only': True, 'synthetic_images_used': False, 'no_new_local_matching': True,
                           'step2a_snapshot_sha256': 'a' * 64, 'step2b_snapshot_sha256': 'b' * 64,
                           'input_file_sha256': {'frozen.csv': sha256_file(source)}, 'step2c_code_sha256': code_hashes(self.root)}
        candidates, controls = joint_fixture()
        endpoints = topology_fixture()
        for name in ('image_id_a', 'image_id_b', 'place_uid_a', 'place_uid_b', 'geo_distance_m'):
            candidates[name] = endpoints[name].tolist() + [({'image_id_a': 'g1', 'image_id_b': 'h1', 'place_uid_a': 'V', 'place_uid_b': 'W', 'geo_distance_m': 1200})[name]]
        observed = FixedGraph(candidates).statistics()
        self.inputs = {'image_edges': candidates, 'joint_candidates': candidates.copy(), 'random': controls,
                       'step2b_snapshot': frozen_fixture(observed), 'provenance': self.provenance}
        self.runtime = self.root / 'cache/countermine_rgb/step2c'
        calibrated, paired, joint_report = analyze_joint_null(candidates, controls)
        publish_stage(self.runtime, 'joint_null', {'random_joint_metrics.csv': calibrated, 'paired_joint_evidence.csv': paired},
                      joint_report, self.provenance, {'seed': 42, 'candidate_count': 5000, 'random_count': 4999,
                                                     'quantiles': [.95, .99], 'primary_self_inclusive': True}, repo_root=self.root)
        distribution, topo_report = analyze_topology_null(candidates, permutations=13, seed=42)
        publish_stage(self.runtime, 'topology_null', {'topology_permutations.csv': distribution}, topo_report,
                      self.provenance, {'seed': 42, 'permutations': 13, 'candidate_count': 5000}, repo_root=self.root)
        self.args = argparse.Namespace(runtime_dir=self.runtime, output_dir=self.root / 'outputs/step2c',
                                       snapshot=self.root / 'docs/audits/step2c_null_topology_metrics.json',
                                       step2b_dir=self.root / 'cache/countermine_rgb/step2b',
                                       step2b_snapshot=self.root / 'docs/audits/step2b_countermine_graph_metrics.json')
        self.loader = patch.object(EXPORTER, 'load_frozen_inputs', return_value=self.inputs)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_complete_export_and_frozen_provenance(self):
        from matplotlib.figure import Figure
        captured = []
        original_save = Figure.savefig
        def capture(figure, *args, **kwargs):
            captured.append(figure)
            return original_save(figure, *args, **kwargs)
        with patch.object(Figure, 'savefig', capture):
            snapshot = EXPORTER.export_audit(self.args, repo_root=self.root)
        # The fixture's cross-city rate exceeds the first panel's rate. Every
        # bar must remain within the shared axis, rather than being clipped.
        for axis in captured[0].axes:
            for rectangle in axis.patches:
                self.assertLessEqual(rectangle.get_y() + rectangle.get_height(), axis.get_ylim()[1] + 1e-12)
        validate_step2c_snapshot(snapshot)
        self.assertEqual(snapshot['provenance']['step2a_snapshot_sha256'], 'a' * 64)
        self.assertEqual(snapshot['provenance']['step2b_snapshot_sha256'], 'b' * 64)
        self.assertEqual(snapshot['provenance']['step2c_code_sha256'], self.provenance['step2c_code_sha256'])
        self.assertEqual(snapshot['provenance']['permutations'], 13)
        self.assertEqual(self.args.snapshot.read_bytes(), (self.runtime / 'null_topology_metrics.json').read_bytes())
        self.assertEqual(len(snapshot['figure_sha256']), 4)
        for name in EXPORTER.ARTIFACT_NAMES:
            source = self.args.output_dir / name
            curated = self.args.snapshot.parent / ('step2c_' + name)
            self.assertEqual(sha256_file(source), sha256_file(curated))
            with Image.open(source) as image:
                image.verify()
        self.assertEqual((self.root / 'frozen.csv').read_text(), 'immutable')

    def test_rejects_modified_permutation_summary_before_figures(self):
        path = self.runtime / 'topology_null_summary.json'
        summary = json.loads(path.read_text())
        summary['report']['topology_null']['comparisons']['core_q95']['repeated_support_2']['empirical_exceedance_fraction'] = -1
        path.write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            EXPORTER.export_audit(self.args, repo_root=self.root)
        self.assertFalse(self.args.output_dir.exists())
        self.assertFalse(self.args.snapshot.exists())

    def test_rejects_historical_snapshot_destination(self):
        self.args.snapshot = self.args.step2b_snapshot
        self.args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        self.args.snapshot.write_text('frozen snapshot')
        with self.assertRaises(ValueError):
            EXPORTER.export_audit(self.args, repo_root=self.root)
        self.assertEqual(self.args.snapshot.read_text(), 'frozen snapshot')
        self.assertFalse(self.args.output_dir.exists())

    def test_rejects_mixed_source_provenance(self):
        path = self.runtime / 'joint_null_summary.json'
        summary = json.loads(path.read_text())
        summary['provenance']['step2b_snapshot_sha256'] = 'changed'
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            EXPORTER.export_audit(self.args, repo_root=self.root)


if __name__ == '__main__':
    unittest.main()
