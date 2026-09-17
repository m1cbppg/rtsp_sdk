import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_demo_bundle import _redact_content


class ExportDemoBundleTests(unittest.TestCase):
    def test_structured_json_secrets_and_rtsp_credentials_are_redacted(self):
        content = json.dumps({
            "api_key": "SYNTHETIC_KEY",
            "nested": {"password": "SYNTHETIC_PASSWORD", "lease_token": "SYNTHETIC_TOKEN"},
            "url": "rtsp://user:pass@example.invalid/live",
        })
        redacted = _redact_content(content, ".json")
        self.assertNotIn("SYNTHETIC_KEY", redacted)
        self.assertNotIn("SYNTHETIC_PASSWORD", redacted)
        self.assertNotIn("SYNTHETIC_TOKEN", redacted)
        self.assertNotIn("user:pass@", redacted)
        parsed = json.loads(redacted)
        self.assertEqual(parsed["api_key"], "[REDACTED]")
        self.assertEqual(parsed["nested"]["password"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
