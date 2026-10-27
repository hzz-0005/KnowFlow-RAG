from __future__ import annotations

from pathlib import Path

from .common import ReplayError, digest, identity, inside, read_json


def load_manifest(path: Path):
    path = path.resolve()
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != "knowflow.m1.manifest.v1":
        raise ReplayError("Unknown manifest schema")
    root = (path.parent / manifest.get("root", ".")).resolve()
    if manifest.get("completion_views") != ["any", "final"]:
        raise ReplayError("Manifest must require both any and final completion views")
    inputs = [{"path": str(path), "sha256": digest(path), "role": "manifest"}]
    for key in ("datasets", "runs", "artifacts", "comparisons"):
        entries = manifest.get(key, [])
        if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
            raise ReplayError(f"Manifest {key} must contain objects")

    def verified(spec, role):
        source = inside(root, spec["path"])
        actual = digest(source) if source.is_file() else None
        if not actual or actual != spec.get("sha256"):
            raise ReplayError(f"Missing input or SHA256 mismatch: {spec['path']}")
        inputs.append({"path": spec["path"], "sha256": actual, "role": role})
        return source

    if not isinstance(manifest.get("contracts"), dict):
        raise ReplayError("Missing metric contracts reference")
    contracts_file = verified(manifest["contracts"], "contracts")
    contracts_data = read_json(contracts_file)
    if (
        not isinstance(contracts_data, dict)
        or contracts_data.get("schema") != "knowflow.metric_contracts.v1"
    ):
        raise ReplayError("Unknown metric contracts schema")
    contracts = contracts_data.get("metrics", [])
    if (
        not isinstance(contracts, list)
        or not contracts
        or any(
            not isinstance(m, dict) or not isinstance(m.get("id"), str) or not m["id"]
            for m in contracts
        )
        or len({m["id"] for m in contracts}) != len(contracts)
    ):
        raise ReplayError("Metric contracts must have unique nonempty IDs")
    for c in contracts:
        if c.get("value_type", "ratio") not in {"ratio", "binary"}:
            raise ReplayError("Metric value_type must be ratio or binary")
        if c.get("eligibility") not in {"all", "evidence", "semantic"}:
            raise ReplayError(f"Unknown eligibility rule: {c['id']}")
        if c.get("interval_method", "none") not in {
            "none",
            "paired_bootstrap",
            "binomial_wilson",
            "clopper_pearson",
            "cluster_bootstrap",
        }:
            raise ReplayError(f"Unknown interval method: {c['id']}")
        if c.get("aggregation_view", "final") != "final":
            raise ReplayError(
                "Metric summary view must be explicitly final; attempts stay separate"
            )
    datasets = {}
    for spec in manifest.get("datasets", []):
        if spec["id"] in datasets:
            raise ReplayError("Duplicate dataset identity")
        source = verified(spec, "dataset")
        data = read_json(source)
        if not isinstance(data, dict):
            raise ReplayError("Dataset must be an object")
        cases = data.get("cases", [])
        if not isinstance(cases, list) or not cases:
            raise ReplayError(f"Empty dataset: {spec['id']}")
        if any(not isinstance(row, dict) for row in cases):
            raise ReplayError("Dataset cases must contain objects")
        ids = [row.get("id") for row in cases]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ReplayError(f"Duplicate/invalid sample IDs: {spec['id']}")
        if any(not isinstance(c.get("question"), str) for c in cases):
            raise ReplayError(f"Missing question: {spec['id']}")
        datasets[spec["id"]] = {
            "spec": spec,
            "data": data,
            "cases": {row["id"]: row for row in cases},
        }
    runs, logical_ids = [], set()
    for spec in manifest.get("runs", []):
        if spec.get("input_granularity") != "per_sample":
            raise ReplayError("Run adapters require explicit per_sample input granularity")
        if spec["id"] in logical_ids:
            raise ReplayError("Duplicate logical run ID")
        logical_ids.add(spec["id"])
        if spec["dataset_id"] not in datasets:
            raise ReplayError(f"Run refers to unknown dataset: {spec['id']}")
        data = read_json(verified(spec, "run"))
        if not isinstance(data, dict):
            raise ReplayError("Run must be an object")
        ds = datasets[spec["dataset_id"]]
        adapter = spec.get("adapter")
        if adapter == "rag_quality":
            if (
                data.get("kind") != "rag_quality"
                or not isinstance(data.get("variants"), dict)
                or data.get("run_id") != spec["id"]
            ):
                raise ReplayError(f"Invalid rag_quality schema: {spec['path']}")
            # The historical writer hashes json.dumps(sort_keys=True), not file bytes.
            if data.get("dataset_sha256") != identity(ds["data"]):
                raise ReplayError(f"Historical canonical dataset hash mismatch: {spec['path']}")
            if data.get("dataset_version") != ds["data"].get("version"):
                raise ReplayError("Dataset version mismatch")
        elif adapter == "retrieval_run":
            if (
                data.get("run_id") != spec["id"]
                or not isinstance(data.get("cases"), list)
                or data.get("dataset_version") != ds["data"].get("version")
            ):
                raise ReplayError("Invalid retrieval-only run schema")
        elif adapter == "ledger_v1":
            if data.get("schema") != "knowflow.replay_trace.v1" or data.get("run_id") != spec["id"]:
                raise ReplayError("Unknown normalized trace schema")
        elif adapter != "planned_only":
            raise ReplayError(f"Unknown run adapter: {adapter}")
        selected = spec.get("selected_sample_ids", list(ds["cases"]))
        if not isinstance(selected, list) or any(not isinstance(i, str) for i in selected):
            raise ReplayError("Selected sample IDs must be a list of strings")
        if len(set(selected)) != len(selected) or any(i not in ds["cases"] for i in selected):
            raise ReplayError("Selected sample IDs must be a unique subset of the dataset")
        strategies = spec.get(
            "strategies",
            [data["strategy"]] if adapter == "retrieval_run" else list(data.get("variants", {})),
        )
        if not strategies or len(set(strategies)) != len(strategies):
            raise ReplayError("Run must declare unique strategies")
        runs.append(
            {
                "spec": spec,
                "data": data,
                "dataset": ds,
                "selected": selected,
                "strategies": strategies,
            }
        )
    artifacts = []
    for spec in manifest.get("artifacts", []):
        source = verified(spec, "aggregate_artifact")
        if spec.get("input_granularity") not in {"aggregate_only", "per_sample"}:
            raise ReplayError("Artifact must explicitly declare its input granularity")
        adapter = spec.get("adapter")
        if adapter not in {"aggregate_json", "aggregate_md", "inventory_only"}:
            raise ReplayError("Unknown aggregate adapter")
        if spec["input_granularity"] == "per_sample" and adapter != "inventory_only":
            raise ReplayError(
                "Unadapted per-sample files are inventory_only, never aggregate inputs"
            )
        item = {
            **spec,
            "recoverability": (
                "aggregate_only"
                if spec["input_granularity"] == "aggregate_only"
                else "per_sample_unadapted"
            ),
            "per_sample_reconstruction": False,
        }
        if adapter == "aggregate_json":
            data = read_json(source)
            # Keep source aggregates as source aggregates. Never distribute their mean to IDs.
            if isinstance(data, dict):
                item["source_metrics"] = data.get("metrics")
                item["source_denominators"] = data.get("denominators")
                item["source_run_id"] = data.get("run_id")
                item["aggregate_fields"] = _aggregate_fields(data)
        artifacts.append(item)
    return {
        "manifest": manifest,
        "manifest_path": path,
        "root": root,
        "inputs": inputs,
        "contracts": contracts,
        "datasets": datasets,
        "runs": runs,
        "artifacts": artifacts,
    }


def _aggregate_fields(value, path=()):
    """Transcribe aggregate containers, never expand case IDs into invented observations."""
    result = []
    if isinstance(value, dict):
        metrics = value.get("metrics")
        if isinstance(metrics, dict):
            result.append(
                {
                    "source_path": list(path),
                    "metrics": [
                        {
                            "metric_id": key,
                            "value": val,
                            "denominator": value.get("denominators", {}).get(key),
                        }
                        for key, val in metrics.items()
                    ],
                }
            )
        for key, child in value.items():
            if key not in {"metrics", "denominators"}:
                result.extend(_aggregate_fields(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_aggregate_fields(child, (*path, index)))
    return result


def verify_inputs(context):
    for entry in context["inputs"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = context["root"] / path
        if digest(path) != entry["sha256"]:
            raise ReplayError(f"Input changed while replaying: {entry['path']}")
