"""Step 2D input identity and narrowly scoped legacy reader compatibility.

Producer Git revisions describe a run; input and implementation hashes bind it.
The only legacy code pairs accepted here are the explicit reader-only repair
below. Arbitrary code edits, including future edits to these files, still fail.
Existing scientific files and their recorded producer metadata stay unchanged.
"""

PROVENANCE_SOURCE = "countermine/mining/full_structural_provenance.py"
READER_REPAIRS = {
    "countermine/mining/local_feature_bank.py": (
        "5b51830635441d769cfcd2e91dea9ab7f3d0633ecc7f40295eb6f09140a89d49",
        "382d7f9d73d963edad2f847747fb7c92702ee799d7a55f9a7c6597a9cccacc23",
    ),
    "countermine/mining/full_structural_measurement.py": (
        "a9bff8df63d87cff65380b1162f60e7ad2d13b8a1777751ab22dc93342bbcec1",
        "4bcbdaca7eee3c879376c719ba6452b81099fde6bf06e8cf1aca189b41a477aa",
    ),
}


def scientific_provenance_matches(saved, current):
    """Compare every scientific field; retain producer Git commit as metadata."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    return ({key: value for key, value in saved.items() if key != "git_commit"}
            == {key: value for key, value in current.items() if key != "git_commit"})


def code_provenance_matches(saved, current):
    """Exact source binding, with just the known legacy-to-repaired reader pairs."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    if saved == current:
        return True
    # The legacy reader did not include this new validation-only module.
    comparison = dict(current)
    if PROVENANCE_SOURCE not in saved:
        comparison.pop(PROVENANCE_SOURCE, None)
    if set(saved) != set(comparison):
        return False
    changed = [key for key in saved if saved[key] != comparison[key]]
    return bool(changed) and all(
        READER_REPAIRS.get(key) == (saved[key], comparison[key]) for key in changed)
