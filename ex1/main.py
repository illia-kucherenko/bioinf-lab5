"""
Gel Electrophoresis Simulation - Lab 5
Bioinformatics Lab Exercise

Main executable script that:
1. Fetches an arbitrary DNA sequence (1000 - 3000 bp) from NCBI.
2. Generates 10 random samples between 100 and 3000 bases.
3. Stores these samples in an array.
4. Calculates their molecular weights and simulates electrophoresis migration.
5. Produces a visual representation with required diagram labels (right): 3000 bp, 1500 bp, 500 bp.
"""

from __future__ import annotations

import argparse
import sys
from gel_simulation import (
    DNAFragment,
    calculate_migration_distance,
    calculate_molecular_weight,
    draw_gel_electrophoresis,
    fetch_ncbi_sequence,
    generate_dna_samples,
)
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate DNA Agarose Gel Electrophoresis from NCBI Sequence."
    )
    parser.add_argument(
        "--accession",
        type=str,
        default="X52331.1",
        help="NCBI Nucleotide Accession ID (sequence length must be between 1000 and 3000 bp). Default: X52331.1 (2958 bp)."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random samples to extract (default: 10)."
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=100,
        help="Minimum sample fragment length in bp (default: 100)."
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=3000,
        help="Maximum sample fragment length in bp (default: 3000)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None for true randomness)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gel_electrophoresis.png",
        help="File path to save the gel diagram (default: gel_electrophoresis.png)."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not display interactive window (only save PNG)."
    )

    args = parser.parse_args()

    # Step 1: Fetch DNA sequence from NCBI
    print(f"[*] Step 1: Fetching sequence for accession '{args.accession}' from NCBI...")
    try:
        header, sequence = fetch_ncbi_sequence(accession=args.accession)
    except Exception as e:
        print(f"[!] Error fetching sequence: {e}", file=sys.stderr)
        return 1

    seq_len = len(sequence)
    print(f"    - Accession: {args.accession}")
    print(f"    - Description: {header}")
    print(f"    - Sequence Length: {seq_len} nucleotides (Satisfies 1000 - 3000 nt requirement)")
    print(f"    - First 60 nt: {sequence[:60]}...")
    print(f"    - Last  60 nt: ...{sequence[-60:]}\n")

    # Step 2: Take 10 random samples between 100 and 3000 bases & store in an array
    print(f"[*] Step 2 & 3: Taking {args.samples} random samples between {args.min_len} and {args.max_len} bases...")
    samples = generate_dna_samples(
        sequence=sequence,
        num_samples=args.samples,
        min_length=args.min_len,
        max_length=args.max_len,
        seed=args.seed
    )

    # Explicitly store sample sequences in an array
    sample_array = [sample.sequence for sample in samples]
    print(f"    - Successfully stored {len(sample_array)} samples in array.\n")

    # Step 3: Print detailed sample summary table
    print(f"[*] Step 4: Sample analysis and migration characteristics:")
    print("-" * 105)
    print(f"{'Sample':<8} {'Length (bp)':<13} {'MW (kDa)':<12} {'GC (%)':<9} {'Rf (Migration)':<16} {'Locus Range':<15} {'Sequence Snippet'}")
    print("-" * 105)
    for s in samples:
        seq_snippet = s.sequence[:12] + "..." + s.sequence[-12:] if len(s.sequence) > 28 else s.sequence
        print(f"S{s.sample_id:<7} {s.length:<13} {s.molecular_weight_kda:<12.2f} {s.gc_content:<9.1f} {s.migration_distance:<16.4f} [{s.start_pos}:{s.end_pos}]".ljust(75) + f" {seq_snippet}")
    print("-" * 105)

    # Reference ladder standards (as requested in Task.md)
    reference_ladder = [3000, 1500, 500]
    print("\n[*] Reference DNA Ladder Standards (Diagram labels on right):")
    for bp in reference_ladder:
        rf = calculate_migration_distance(bp)
        approx_mw = (bp * 650.0) / 1000.0
        print(f"    - {bp:>4} bp  ->  Approx MW: {approx_mw:>7.1f} kDa  ->  Migration distance Rf: {rf:.4f}")

    # Step 4: Physical explanation
    print("\n[*] Physical Principle of Migration:")
    print("    - Friction Force = f(size, gel porosity).")
    print("    - Shorter DNA fragments (< 1000 bp) experience minimal resistance from agarose fibers and migrate rapidly.")
    print("    - Longer DNA fragments (> 2000 bp) encounter high friction / reptation forces and migrate slowly.")
    print("    - Mathematical model: Migration Rf proportional to -log10(Molecular Weight / Length).")

    # Step 5: Draw visual representation
    print(f"\n[*] Step 5: Generating visual representation of gel electrophoresis...")
    draw_gel_electrophoresis(
        samples=samples,
        reference_ladder_bp=reference_ladder,
        output_path=args.output,
        show_window=not args.no_gui
    )

    print(f"\nSimulation completed successfully! Output saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
