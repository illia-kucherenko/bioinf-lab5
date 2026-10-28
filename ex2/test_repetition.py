"""
Unit Tests for DNA Repetition Detection Engine (Bioinformatics Lab 5 - Exercise 2)
"""

import os
import unittest
import matplotlib.pyplot as plt

from repetition_detector import (
    BruteForceRepetitionEngine,
    RepetitionResult,
    draw_repetition_analysis,
    load_dna_sequence,
)


class TestRepetitionDetector(unittest.TestCase):
    """Test suite verifying all requirements of Task 2.md."""

    @classmethod
    def setUpClass(cls):
        # Load test sequence (Milk_vetch_chlorotic_dwarf_virus.fasta)
        cls.header, cls.sequence = load_dna_sequence("Milk_vetch_chlorotic_dwarf_virus.fasta")

    def test_sequence_length(self):
        """Sequence must be around 1000 - 3000 nt."""
        seq_len = len(self.sequence)
        self.assertGreaterEqual(seq_len, 900)
        self.assertLessEqual(seq_len, 3500)
        # Ensure only valid nucleotides
        self.assertTrue(set(self.sequence).issubset(set("ATGCN")))

    def test_combinatorial_generation(self):
        """Combinations between 3 and 6 letters must be correctly generated."""
        for k, expected_count in [(3, 64), (4, 256), (5, 1024), (6, 4096)]:
            combs = BruteForceRepetitionEngine.generate_all_combinations(k)
            self.assertEqual(len(combs), expected_count)
            self.assertEqual(len(combs[0]), k)
            self.assertEqual(len(set(combs)), expected_count)

    def test_tandem_repeat_synthetic(self):
        """Verify brute force detection on controlled synthetic tandem repeats."""
        # Synthetic sequence with known tandem repeats:
        # 1. CAGCAG -> unit 'CAG' (len 3), repeats 2, total 6
        # 2. AGAAGAAGA -> unit 'AGA' (len 3), repeats 3, total 9
        # 3. GAAGGAAG -> unit 'GAAG' (len 4), repeats 2, total 8
        # 4. TTACTTTACT -> unit 'TTACT' (len 5), repeats 2, total 10
        test_seq = "NNNN" + "CAGCAG" + "NNNN" + "AGAAGAAGA" + "NNNN" + "GAAGGAAG" + "NNNN" + "TTACTTTACT" + "NNNN"
        engine = BruteForceRepetitionEngine(test_seq)
        results = engine.detect_tandem_repeats()

        patterns = {r.pattern: r for r in results}
        self.assertIn("CAGCAG", patterns)
        self.assertEqual(patterns["CAGCAG"].motif, "CAG")
        self.assertEqual(patterns["CAGCAG"].length, 6)

        self.assertIn("AGAAGAAGA", patterns)
        self.assertEqual(patterns["AGAAGAAGA"].motif, "AGA")
        self.assertEqual(patterns["AGAAGAAGA"].length, 9)

        self.assertIn("GAAGGAAG", patterns)
        self.assertEqual(patterns["GAAGGAAG"].motif, "GAAG")
        self.assertEqual(patterns["GAAGGAAG"].length, 8)

        self.assertIn("TTACTTTACT", patterns)
        self.assertEqual(patterns["TTACTTTACT"].motif, "TTACT")
        self.assertEqual(patterns["TTACTTTACT"].length, 10)

    def test_repetition_length_bounds(self):
        """All detected repetitions must be between 6 and 10 letters."""
        engine = BruteForceRepetitionEngine(self.sequence)
        tandem = engine.detect_tandem_repeats()
        dispersed = engine.detect_dispersed_repeats()

        for r in tandem:
            self.assertGreaterEqual(r.length, 6, f"Tandem repeat {r.pattern} is shorter than 6 letters")
            self.assertLessEqual(r.length, 10, f"Tandem repeat {r.pattern} is longer than 10 letters")
            self.assertGreaterEqual(r.motif_length, 3, f"Motif {r.motif} is shorter than 3 letters")
            self.assertLessEqual(r.motif_length, 6, f"Motif {r.motif} is longer than 6 letters")

        for r in dispersed:
            self.assertGreaterEqual(r.length, 6, f"Dispersed repeat {r.pattern} is shorter than 6 letters")
            self.assertLessEqual(r.length, 10, f"Dispersed repeat {r.pattern} is longer than 10 letters")
            self.assertGreaterEqual(r.occurrences, 2, f"Dispersed repeat {r.pattern} does not repeat")

    def test_visual_diagram_generation(self):
        """Verify that repetition analysis diagram generates cleanly."""
        engine = BruteForceRepetitionEngine(self.sequence)
        res = engine.run_all()
        test_out = "test_repetition_diagram.png"

        fig = draw_repetition_analysis(
            sequence=self.sequence,
            tandem_repeats=res["tandem_repeats"],
            dispersed_repeats=res["dispersed_repeats"],
            output_path=test_out,
            show_window=False
        )
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(test_out))
        self.assertGreater(os.path.getsize(test_out), 20000)

        plt.close(fig)
        if os.path.exists(test_out):
            os.remove(test_out)


if __name__ == "__main__":
    unittest.main()
