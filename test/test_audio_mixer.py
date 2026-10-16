"""Check the shortened mixer demos without installing the analysis pipeline.

Run with ``python -m unittest test.test_audio_mixer`` (requires FFmpeg).
The metadata describes the original dataset excerpts, not the demo MP3 length.
"""

import json
from pathlib import Path
import shutil
import subprocess
import unittest


ASSETS = Path(__file__).resolve().parents[1] / "docs" / "_static" / "audio-mixer"
TRACKS = ("jamal", "mance")
STEMS = ("piano", "bass", "drums", "beats")


@unittest.skipUnless(shutil.which("ffprobe"), "FFmpeg's ffprobe is required")
class ShortenedMixerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.durations = {}
        for track in TRACKS:
            for stem in STEMS:
                path = ASSETS / track / (stem + ".mp3")
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "json", str(path)],
                    check=True, capture_output=True, text=True,
                )
                cls.durations[track, stem] = float(
                    json.loads(result.stdout)["format"]["duration"]
                )

    def test_each_stem_is_under_thirty_seconds(self):
        for (track, stem), duration in self.durations.items():
            with self.subTest(track=track, stem=stem):
                self.assertGreater(duration, 29.0)
                self.assertLess(duration, 30.0)

    def test_stems_keep_matching_lengths(self):
        for track in TRACKS:
            with self.subTest(track=track):
                durations = [self.durations[track, stem] for stem in STEMS]
                self.assertAlmostEqual(max(durations), min(durations), places=5)


if __name__ == "__main__":
    unittest.main()
