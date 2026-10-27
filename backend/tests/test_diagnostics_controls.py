import importlib.util
from pathlib import Path


def test_frozen_synthetic_controls_include_parser_paths_unknowns_and_semantic_limit():
    path = Path(__file__).resolve().parents[2] / "evaluation/diagnostics_v1/run_controls.py"
    spec = importlib.util.spec_from_file_location("diagnostic_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.evaluate()
    assert report["n_cases"] == 17
    assert report["n_contract_mismatches"] == 0
    assert report["roles"]["negative_control"] == 4
    assert report["roles"]["unsupported_semantics"] == 1
    blind = next(row for row in report["rows"] if row["role"] == "unsupported_semantics")
    assert blind["mutation"] == "same_length_replacement"
