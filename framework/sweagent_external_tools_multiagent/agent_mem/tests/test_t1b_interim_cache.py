"""Level 0 unit tests for T1-B: InterimCache."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_mem.storage.interim_cache import InterimCache


class TestInterimCache(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = InterimCache(cache_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


    def test_write_and_read_roundtrip(self):
        """Written card can be read back with correct fields."""
        ok = self.cache.write_interim_card(
            instance_id="inst-001",
            attempt_id="attempt-1",
            card_type="InterimLocalizationCard",
            localization={"file": "foo/bar.py", "function": "baz", "line_range": "42-42", "confidence": 0.7},
            source_step=5,
        )
        self.assertTrue(ok)
        cards = self.cache.read_interim_cards("inst-001")
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["card_type"], "InterimLocalizationCard")
        self.assertEqual(cards[0]["localization"]["file"], "foo/bar.py")
        self.assertEqual(cards[0]["source_step"], 5)

    def test_multiple_cards_for_same_instance(self):
        """Multiple cards can be appended and all are read back."""
        for i in range(3):
            self.cache.write_interim_card(
                instance_id="inst-multi",
                attempt_id="attempt-1",
                card_type="InterimLocalizationCard",
                localization={"file": f"src/f{i}.py", "function": f"fn{i}", "line_range": "1-1", "confidence": 0.6},
                source_step=i,
            )
        cards = self.cache.read_interim_cards("inst-multi")
        self.assertEqual(len(cards), 3)

    def test_empty_read_on_missing_instance(self):
        """Reading a non-existent instance returns empty list."""
        cards = self.cache.read_interim_cards("nonexistent-999")
        self.assertEqual(cards, [])


    def test_concurrent_write_no_corruption(self):
        """Two threads writing concurrently produce intact JSON."""
        errors: list[str] = []

        def write_cards(attempt_id: str):
            for _ in range(5):
                ok = self.cache.write_interim_card(
                    instance_id="inst-concurrent",
                    attempt_id=attempt_id,
                    card_type="InterimLocalizationCard",
                    localization={"file": "c.py", "function": "f", "line_range": "1-1", "confidence": 0.5},
                    source_step=0,
                )
                if not ok:
                    errors.append(f"{attempt_id}: write returned False")

        t1 = threading.Thread(target=write_cards, args=("a-1",))
        t2 = threading.Thread(target=write_cards, args=("a-2",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(errors, [])
        cards = self.cache.read_interim_cards("inst-concurrent")
        # 10 writes total, but capped at 20, so all 10 should be there
        self.assertEqual(len(cards), 10)


    def test_build_hint_items_format(self):
        """build_hint_items returns correctly shaped hint dicts."""
        self.cache.write_interim_card(
            instance_id="inst-hint",
            attempt_id="attempt-1",
            card_type="InterimLocalizationCard",
            localization={"file": "src/module.py", "function": "process", "line_range": "100-105", "confidence": 0.75},
            source_step=12,
        )
        items = self.cache.build_hint_items("inst-hint")
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertIn("hint", item)
        self.assertIn("src/module.py", item["hint"])
        self.assertEqual(item["card_type"], "InterimLocalizationCard")
        self.assertTrue(item["family_id"].startswith("interim:"))
        self.assertIn("item_confidence", item)

    def test_build_hint_items_deduplication(self):
        """Two cards with same file+function produce only one hint."""
        for i in range(2):
            self.cache.write_interim_card(
                instance_id="inst-dedup",
                attempt_id=f"attempt-{i}",
                card_type="InterimLocalizationCard",
                localization={"file": "same.py", "function": "same_fn", "line_range": "10-10", "confidence": 0.7},
                source_step=i,
            )
        items = self.cache.build_hint_items("inst-dedup")
        self.assertEqual(len(items), 1)

    def test_build_hint_items_skips_no_file(self):
        """Cards without a file path produce no hint."""
        self.cache.write_interim_card(
            instance_id="inst-nofile",
            attempt_id="attempt-1",
            card_type="InterimLocalizationCard",
            localization={"function": "foo", "line_range": "1-1", "confidence": 0.6},
            source_step=1,
        )
        items = self.cache.build_hint_items("inst-nofile")
        self.assertEqual(items, [])


    def test_cap_at_max_cards(self):
        """Writing more than _MAX_CARDS_PER_INSTANCE keeps only the latest ones."""
        from agent_mem.storage.interim_cache import _MAX_CARDS_PER_INSTANCE
        for i in range(_MAX_CARDS_PER_INSTANCE + 5):
            self.cache.write_interim_card(
                instance_id="inst-cap",
                attempt_id="a-1",
                card_type="InterimLocalizationCard",
                localization={"file": f"f{i}.py", "function": "fn", "line_range": "1-1", "confidence": 0.5},
                source_step=i,
            )
        cards = self.cache.read_interim_cards("inst-cap")
        self.assertLessEqual(len(cards), _MAX_CARDS_PER_INSTANCE)
        # Should keep the latest ones: last file should be present
        files = [c["localization"]["file"] for c in cards]
        self.assertIn(f"f{_MAX_CARDS_PER_INSTANCE + 4}.py", files)


    def test_archive_moves_file(self):
        """archive() moves the cache file to archived/ subdirectory."""
        self.cache.write_interim_card(
            instance_id="inst-archive",
            attempt_id="a-1",
            card_type="InterimLocalizationCard",
            localization={"file": "x.py", "function": "f", "line_range": "1-1", "confidence": 0.5},
            source_step=1,
        )
        active_path = os.path.join(self.tmpdir, "inst-archive.json")
        self.assertTrue(os.path.exists(active_path))
        self.cache.archive("inst-archive")
        self.assertFalse(os.path.exists(active_path))
        archived_path = os.path.join(self.tmpdir, "archived", "inst-archive.json")
        self.assertTrue(os.path.exists(archived_path))

    def test_archive_nonexistent_is_noop(self):
        """archive() on a missing instance does not raise."""
        try:
            self.cache.archive("no-such-instance")
        except Exception as e:
            self.fail(f"archive() raised unexpectedly: {e}")


    def test_isolation_from_other_instance(self):
        """Writing to inst-A does not affect inst-B."""
        self.cache.write_interim_card(
            instance_id="inst-A",
            attempt_id="a-1",
            card_type="InterimLocalizationCard",
            localization={"file": "a.py", "function": "fa", "line_range": "1-1", "confidence": 0.6},
            source_step=1,
        )
        cards_b = self.cache.read_interim_cards("inst-B")
        self.assertEqual(cards_b, [])

    def test_from_env_with_custom_dir(self):
        """from_env() picks up SWE_AGENT_T1B_CACHE_DIR."""
        import os
        with unittest.mock.patch.dict(os.environ, {"SWE_AGENT_T1B_CACHE_DIR": "/tmp/test_cache_xyz"}):
            cache = InterimCache.from_env()
        self.assertEqual(cache._cache_dir, "/tmp/test_cache_xyz")


# Need to import mock for the last test
import unittest.mock

if __name__ == "__main__":
    unittest.main()
