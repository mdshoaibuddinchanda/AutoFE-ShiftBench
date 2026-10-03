from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.provenance import build_dataset_registry, build_provenance_package, canonical_sha256, verify_provenance
from src.protocol import EVALUATION_PROTOCOL_VERSION
from src.seeding import SEED_SCHEME_VERSION
from src.task_manifest import ManifestStore, build_task_records


class ProvenanceTests(unittest.TestCase):
    def _repo(self) -> Path:
        root = Path(tempfile.mkdtemp())
        (root / "config").mkdir()
        (root / "data" / "raw").mkdir(parents=True)
        (root / "config" / "dataset_list.yaml").write_text("datasets:\n  - name: present\n  - name: missing\n", encoding="utf-8")
        (root / "data" / "raw" / "present.csv").write_text("feature,target_label\n0,0\n1,1\n", encoding="utf-8")
        return root

    def test_registry_labels_missing_data_without_inventing_hashes(self) -> None:
        root = self._repo()
        registry = build_dataset_registry(root / "config" / "dataset_list.yaml", repo_root=root)
        by_name = {item["name"]: item for item in registry["datasets"]}
        self.assertEqual(by_name["present"]["status"], "verified")
        self.assertEqual(by_name["missing"]["status"], "unavailable")
        self.assertNotIn("sha256", by_name["missing"])
        self.assertTrue(all(not str(item["path"]).startswith(("C:", "D:", "/")) for item in registry["datasets"]))
        self.assertEqual(canonical_sha256({"b": 1, "a": 2}), canonical_sha256({"a": 2, "b": 1}))

    def test_package_and_compatibility_verification_are_read_only(self) -> None:
        root = self._repo()
        records = build_task_records(["present"], [1], [0], [("clean", 0.0, "clean")], ["Raw"], ["logistic_regression"], include_precompute=False)
        db = root / "manifest.db"
        store = ManifestStore(db)
        run_id = "provenance_test"
        store.create_run(run_id, {"protocol_version": EVALUATION_PROTOCOL_VERSION, "seed_scheme_version": SEED_SCHEME_VERSION}, records)
        attempt = store.claim_task(run_id, records[0]["scientific_task_id"], worker_id="test")
        payload = {"status": "success", "roc_auc": 0.75}
        store.commit_result(run_id, records[0]["scientific_task_id"], attempt, payload)
        ledger = root / "results.jsonl"
        ledger.write_text(json.dumps({"run_id": run_id, "scientific_task_id": records[0]["scientific_task_id"], **payload}) + "\n", encoding="utf-8")
        output = root / "reports" / "provenance"
        build_provenance_package(output_dir=output, repo_root=root, dataset_list_path=root / "config" / "dataset_list.yaml", manifest_db=db, ledger_path=ledger, run_id=run_id)
        result = verify_provenance(repo_root=root, dataset_list_path=root / "config" / "dataset_list.yaml", manifest_db=db, ledger_path=ledger, run_id=run_id, package_dir=output)
        self.assertEqual(result["overall_status"], "valid")
        names = {check["name"] for check in result["checks"]}
        self.assertIn("protocol_compatibility", names)
        self.assertIn("seed_scheme_compatibility", names)
        self.assertIn("task_protocol_compatibility", names)
        self.assertIn("task_seed_scheme_compatibility", names)
        package_text = "\n".join(path.read_text(encoding="utf-8") for path in output.glob("*.json"))
        self.assertNotIn(str(root), package_text)

        (root / "data" / "raw" / "present.csv").write_text("feature,target_label\n0,0\n1,0\n", encoding="utf-8")
        tampered = verify_provenance(repo_root=root, dataset_list_path=root / "config" / "dataset_list.yaml", manifest_db=db, ledger_path=ledger, run_id=run_id, package_dir=output)
        self.assertEqual(tampered["overall_status"], "attention_required")
        self.assertTrue(any(check["status"] == "conflict" for check in tampered["checks"]))


if __name__ == "__main__":
    unittest.main()
