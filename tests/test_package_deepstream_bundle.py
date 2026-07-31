from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.package_deepstream_bundle import FILES, package


class FakeProcess:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.stdout = io.BytesIO(b"docker-image-tar")

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def poll(self) -> int:
        return 0


class PackageDeepStreamBundleTests(unittest.TestCase):
    def test_package_streams_image_and_deployment_files_into_one_zip(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            for source in FILES:
                path = project / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            output = root / "bundle.zip"
            with patch(
                "scripts.package_deepstream_bundle.subprocess.Popen",
                FakeProcess,
            ):
                package(project, output, ["image:one", "image:two"])
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                image_bytes = archive.read("images.tar")

        self.assertEqual(image_bytes, b"docker-image-tar")
        self.assertIn("engines/.keep", names)
        self.assertTrue(set(FILES.values()).issubset(names))


if __name__ == "__main__":
    unittest.main()
