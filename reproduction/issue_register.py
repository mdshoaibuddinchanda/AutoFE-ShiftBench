"""Create/update the explicit audit register; never silently remove an item."""
import argparse
import json
from pathlib import Path

REGISTER = Path("provenance/critical_repair/issue_register.json")
GROUPS = {
    "B": [
        "Probability metric class alignment", "Dataset/task/cache compatibility", "Precompute retry dependencies", "Durable results and crash recovery", "Provenance validation", "Stop-prefix information boundary", "Metric ranges and complete bounds", "Missing comparison sides", "Publication statistics and figures", "Distribution measurement coordinates", "Feature counts", "Class-prior condition semantics", "Categorical memory safety", "Dependency-ready dispatch", "Deadlines stop limits and run status", "Concurrent artifact publication", "Worker/writer errors and retention", "SQLite resource lifetime"],
    "C": [
        "One run and full analysis compatibility", "Authoritative snapshot identity", "Read-only verification", "Actual artifact lineage", "Stability history schema", "MI discrete-feature provenance", "Exact downloaded source identity", "Bounded non-overwriting acquisition", "Supported P12 launcher", "Actual GPU fit and declared device policy", "Unused SHAP import and output axes", "F1 averaging and legacy schema", "Missing table cells", "Sensitivity regime scope", "Duplicate payloads and attempt chronology", "Reporting flags notebooks and empty helper", "Depth duplicates zero severity and cap validation", "Timeout queue drain", "Rare-class cap feasibility"],
    "E": [
        "Requested prepared inputs only", "Compute and reuse complete diagnostics once", "Selected pipeline precompute", "Native thread budget", "Concurrent killable timed tasks", "Reuse Windows workers", "Common candidate artifact sharing", "Select before held-out materialization", "Stream exact candidate scoring", "Batch NumPy and memoize expressions", "Value-only perturbation evaluation", "Stream sparse Jacobian summaries", "Dependency-aware finite differences", "Exact sparse one-hot representation", "Final matrices and read-only memmaps", "Dynamic dependency-ready scheduling", "Memory/VRAM and queue backpressure", "Reference-aware cache lifetime", "Batched manifest queries", "Durable persistence batching", "Lazy unused imports", "Equivalent prediction reuse", "RNG-preserving perturbation vectorization", "Bounded acquisition and metadata reuse"],
    "R": [
        "Shared ledger and analysis bundle", "Indexed joins", "Normalized sensitivity memberships", "Identical regime calculation reuse", "Exact sign enumeration and batched resampling", "Equivalent legacy Cliff delta", "Scoped artifact hash reuse", "Figure aggregate reuse with required formats"],
    "M": ["Existing invalid escape warnings", "Shared P12 unrelated package conflicts", "Historical manuscript grid is separate", "Code fingerprint covers untracked contents", "External artifact paths are unambiguous"],
}
B_FILES = ["evaluation.py,pipeline_runner.py", "task_manifest.py,pipeline_runner.py", "pipeline_runner.py,task_manifest.py", "pipeline_runner.py,task_manifest.py", "provenance.py", "sensitivity_analysis.py", "dataset_statistics.py,sensitivity_analysis.py", "dataset_statistics.py", "plotting_q1.py,generate_tables.py,notebooks", "evaluation.py,pipeline_runner.py", "pipeline_runner.py,feature_engineering.py", "shift_generator.py,paper/manuscript.tex", "preprocessing.py", "pipeline_runner.py", "pipeline_runner.py", "pipeline_runner.py", "pipeline_runner.py", "task_manifest.py"]


def create():
    if REGISTER.exists():
        raise FileExistsError(REGISTER)
    rows = []
    for group, titles in GROUPS.items():
        for i, title in enumerate(titles, 1):
            rows.append({"id": f"{group}{i:02}", "title": title, "evidence_level": "supplied_audit; pending checkout regression", "path": "active_or_latent_to_verify", "supporting_locations": B_FILES[i-1] if group == "B" else "supplied audit and named modules; precise test evidence pending", "original_reproducer": None, "required_correction": title, "compatibility_impact": "scientific correction or equivalent optimization to adjudicate", "tests": [], "status": "pending", "limitations": []})
    REGISTER.write_text(json.dumps({"schema": "critical_repair_issue_register_v1", "baseline": "55cf47307c86ad68e827326d862469ec89193a7d", "issues": rows}, indent=2)+"\n", encoding="utf-8")


def update(ids, fields):
    value = json.loads(REGISTER.read_text(encoding="utf-8"))
    wanted = set(ids.split(","))
    found = set()
    for row in value["issues"]:
        if row["id"] in wanted:
            row.update(fields)
            found.add(row["id"])
    if found != wanted:
        raise ValueError(f"Unknown IDs: {wanted-found}")
    REGISTER.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--ids")
    parser.add_argument("--fields", type=Path)
    args = parser.parse_args()
    if args.create:
        create()
    else:
        update(args.ids, json.loads(args.fields.read_text(encoding="utf-8")))
