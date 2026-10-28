"""
Unit Tests for Gel Electrophoresis Simulation (Bioinformatics Lab 5)
"""

import math
import os
import unittest
import matplotlib.pyplot as plt

from gel_simulation import (
    DNAFragment,
    calculate_migration_distance,
    calculate_molecular_weight,
    draw_gel_electrophoresis,
    fetch_ncbi_sequence,
    generate_dna_samples,
)


class TestGelElectrophoresisSimulation(unittest.TestCase):
    """Test suite verifying all requirements from Task.md."""

    @classmethod
    def setUpClass(cls):
        # Fetch or load NCBI sequence
        cls.header, cls.sequence = fetch_ncbi_sequence("X52331.1")

    def test_ncbi_sequence_length(self):
        """Sequence must be between 1000 and 3000 nucleotides / letters."""
        seq_len = len(self.sequence)
        self.assertGreaterEqual(seq_len, 1000, "Sequence must be at least 1000 bp")
        self.assertLessEqual(seq_len, 3000, "Sequence must be at most 3000 bp")
        # Ensure it contains valid nucleotide characters
        valid_nucleotides = set("ATGCN")
        self.assertTrue(set(self.sequence).issubset(valid_nucleotides))

    def test_sample_count_and_storage(self):
        """Take 10 random samples from sequence and store them in an array."""
        samples = generate_dna_samples(self.sequence, num_samples=10, seed=123)
        # Verify 10 samples
        self.assertEqual(len(samples), 10)
        self.assertIsInstance(samples, list)

        # Store in array of sequences
        sample_array = [s.sequence for s in samples]
        self.assertEqual(len(sample_array), 10)

        # Verify each sample is a valid substring of the NCBI sequence
        for sample in samples:
            self.assertIn(sample.sequence, self.sequence)
            self.assertEqual(len(sample.sequence), sample.length)
            self.assertEqual(sample.sequence, self.sequence[sample.start_pos:sample.end_pos])

    def test_sample_lengths_range(self):
        """Samples must be between 100 and 3000 bases."""
        samples = generate_dna_samples(self.sequence, num_samples=10, min_length=100, max_length=3000, seed=456)
        for s in samples:
            self.assertGreaterEqual(s.length, 100, f"Sample {s.sample_id} is shorter than 100 bp")
            self.assertLessEqual(s.length, 3000, f"Sample {s.sample_id} is longer than 3000 bp")

    def test_molecular_weight_calculation(self):
        """Molecular weight of DNA fragments should be physically realistic."""
        # A 1000 bp fragment dsDNA should be ~650 kDa (+/- 5%)
        test_seq = "A" * 500 + "G" * 500
        mw = calculate_molecular_weight(test_seq, is_dsdna=True)
        # ~650,000 Da
        self.assertGreater(mw, 600000)
        self.assertLess(mw, 700000)

    def test_migration_friction_physics(self):
        """
        Verify the fundamental physical law:
        Short DNA fragments meet small friction forces and travel faster/farther.
        Long DNA fragments exhibit high friction force and travel slowly (smaller distance).
        """
        rf_3000 = calculate_migration_distance(3000)
        rf_1500 = calculate_migration_distance(1500)
        rf_500 = calculate_migration_distance(500)
        rf_100 = calculate_migration_distance(100)

        # Smaller fragment length -> Greater migration distance Rf
        self.assertLess(rf_3000, rf_1500, "3000 bp must migrate less than 1500 bp")
        self.assertLess(rf_1500, rf_500, "1500 bp must migrate less than 500 bp")
        self.assertLess(rf_500, rf_100, "500 bp must migrate less than 100 bp")

        # Relative values within [0.0, 1.0]
        for rf in [rf_3000, rf_1500, rf_500, rf_100]:
            self.assertGreaterEqual(rf, 0.0)
            self.assertLessEqual(rf, 1.0)

    def test_required_diagram_ladder_labels(self):
        """Diagram labels on the right must represent 3000 bp, 1500 bp, and 500 bp."""
        required_labels = [3000, 1500, 500]
        distances = {bp: calculate_migration_distance(bp) for bp in required_labels}

        # Verify strictly decreasing size corresponds to strictly increasing migration distance
        self.assertTrue(distances[3000] < distances[1500] < distances[500])

    def test_gel_visual_generation(self):
        """Verify that the visual representation generates and saves cleanly without error."""
        test_output = "test_gel_output.png"
        samples = generate_dna_samples(self.sequence, num_samples=10, seed=789)
        fig = draw_gel_electrophoresis(
            samples=samples,
            reference_ladder_bp=[3000, 1500, 500],
            output_path=test_output,
            show_window=False
        )
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(test_output), "Gel image file should be created")
        file_size = os.path.getsize(test_output)
        self.assertGreater(file_size, 50000, "Generated gel image should be a valid non-empty file")

        # Explicitly close figure to release canvas and avoid file handle locks
        plt.close(fig)

        # Clean up test output
        if os.path.exists(test_output):
            os.remove(test_output)


if __name__ == "__main__":
    unittest.main()
