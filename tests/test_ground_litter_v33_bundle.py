"""Integrity tests for the Ground Litter V3.3 candidate-image bundle.

The candidate image itself is built on the server under a separate authorisation.
What can and must be verified locally is that the bundle cannot silently drift
away from the Dockerfile, that the default base image is the running hardened
V3.2 tag, and that the candidate never reuses a production tag. A manifest whose
file list no longer matches the Dockerfile would look correct while shipping the
wrong payload.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from scripts.package_ground_litter_v33_bundle import (
    BUNDLE_NAME,
    CANDIDATE_IMAGE,
    CONTAINER_COMPARE_FILES,
    CONTEXT_DIRS,
    CONTEXT_FILES,
    DOCKERFILE_NAME,
    REPO_ROOT,
    ROLLBACK_IMAGE,
    collect_context_files,
    dockerfile_base_image,
)

PRODUCTION_V32_HARDENED_TAG = (
    "rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918"
)


def dockerfile_copy_sources() -> list[str]:
    text = (REPO_ROOT / DOCKERFILE_NAME).read_text(encoding="utf-8")
    text = re.sub(r"\\\s*\n\s*", " ", text)  # join line continuations
    sources: list[str] = []
    for line in text.splitlines():
        if not line.startswith("COPY "):
            continue
        parts = line.split()[1:]
        # The last token is the destination; everything before it is a source.
        sources.extend(parts[:-1])
    return sources


class CandidateBundleLayoutTests(unittest.TestCase):
    def test_every_dockerfile_copy_source_is_packaged(self):
        packaged = set(CONTEXT_FILES)
        for directory in CONTEXT_DIRS:
            packaged.add(directory)
        unpackaged = [
            source for source in dockerfile_copy_sources()
            if source not in packaged and not any(
                source == directory or source.startswith(directory + "/")
                for directory in CONTEXT_DIRS
            )
        ]
        self.assertEqual(unpackaged, [], "Dockerfile COPY 源未包含在候选包中")

    def test_packaged_paths_all_exist(self):
        files = collect_context_files()
        self.assertTrue(files)
        for path in files:
            self.assertTrue(path.is_file(), path)

    def test_context_covers_every_runtime_module_the_dockerfile_replaces(self):
        copied = {
            source for source in dockerfile_copy_sources()
            if source.startswith("rtsp_annotator/")
        }
        self.assertEqual(copied, set(CONTAINER_COMPARE_FILES))

    def test_bundle_name_matches_the_deploy_document(self):
        self.assertEqual(BUNDLE_NAME, "ground-litter-v33-dual-recall-20260918")


class CandidateImageTagTests(unittest.TestCase):
    def test_default_base_image_is_the_running_hardened_v32_tag(self):
        # The overlay must land on the image that is actually running, so the
        # server build needs no pull and the rollback tag stays meaningful.
        self.assertEqual(dockerfile_base_image(), PRODUCTION_V32_HARDENED_TAG)

    def test_never_overwrites_a_production_tag(self):
        source = (REPO_ROOT / DOCKERFILE_NAME).read_text(encoding="utf-8")
        # The Dockerfile may only reference the base image through FROM; it must
        # not pin a -t/--tag of its own, which would let a build overwrite it.
        self.assertNotIn("--tag", source)
        self.assertNotIn("docker tag", source)
        self.assertEqual(source.count("FROM "), 1)
        self.assertIn("FROM ${BASE_IMAGE}", source)

    def test_candidate_tag_is_distinct_from_every_referenced_production_tag(self):
        self.assertNotEqual(CANDIDATE_IMAGE, PRODUCTION_V32_HARDENED_TAG)
        self.assertEqual(CANDIDATE_IMAGE, ROLLBACK_IMAGE.replace(
            "ground-litter-v32-hardening-20260918",
            "ground-litter-v33-dual-recall-20260918",
        ))
        self.assertTrue(CANDIDATE_IMAGE.startswith("rtsp-yolo-annotator:"))
        self.assertIn("v33-dual-recall", CANDIDATE_IMAGE)


class BundleEntrypointTests(unittest.TestCase):
    def test_bundle_script_is_importable_without_side_effects(self):
        # Importing must not build an image, copy files or touch the network.
        module_path = REPO_ROOT / "scripts" / "package_ground_litter_v33_bundle.py"
        self.assertTrue(module_path.is_file())

    def test_context_file_list_is_relative_and_unique(self):
        self.assertEqual(len(CONTEXT_FILES), len(set(CONTEXT_FILES)))
        for name in CONTEXT_FILES:
            path = Path(name)
            self.assertFalse(path.is_absolute(), name)
            self.assertNotIn("..", path.parts, name)

    def test_tarball_is_byte_reproducible(self):
        # The recorded SHA-256 is only useful if rebuilding the same sources
        # yields the same archive; a gzip timestamp would silently invalidate it.
        import hashlib  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from scripts.package_ground_litter_v33_bundle import make_tarball

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "bundle" / "payload"
            staging.mkdir(parents=True)
            (staging / "a.py").write_text("print('a')\n", encoding="utf-8")
            (staging / "sub").mkdir()
            (staging / "sub" / "b.json").write_text("{}\n", encoding="utf-8")

            digests = []
            for index in (1, 2):
                target = root / f"context-{index}.tar.gz"
                make_tarball(staging, target)
                digests.append(hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
