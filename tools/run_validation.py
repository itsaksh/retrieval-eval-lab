"""Run the real suite and write a compact evidence record. Exit nonzero on failure."""
import json
import os
from pathlib import Path
import platform
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"python": platform.python_version(), "platform": platform.system(), "tests_run": result.testsRun,
              "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": [{"test": str(test), "reason": reason} for test,reason in result.skipped],
              "real_model_integration_enabled": bool(os.environ.get("RETRIEVAL_MODEL_DIR")),
              "successful": result.wasSuccessful()}
    path = ROOT / "reports/test-results.json"
    path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
