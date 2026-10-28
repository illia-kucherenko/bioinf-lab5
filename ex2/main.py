"""
DNA Repetition Detector - Exercise 2
Main Application Script

Fulfills Task 2.md:
- Reads an arbitrary DNA sequence from NCBI (1000 - 3000 nt).
- Uses a brute force engine to generate combinations between 3 and 6 letters.
- Detects repetitions between 6 and 10 letters in the DNA sequence.
- Produces formatted table and visual analysis diagram.
"""

from __future__ import annotations

import argparse
import sys
from repetition_detector import (
    BruteForceRepetitionEngine,
    draw_repetition_analysis,
    load_dna_sequence,
)
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect DNA repetitions between 6 and 10 letters using brute force."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="Milk_vetch_chlorotic_dwarf_virus.fasta",
        help="Path to FASTA file (default: Milk_vetch_chlorotic_dwarf_virus.fasta)."
    )
    parser.add_argument(
        "--accession",
        type=str,
        default=None,
        help="Optional NCBI Nucleotide Accession ID to fetch online."
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=6,
        help="Minimum repetition length in letters (default: 6)."
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=10,
        help="Maximum repetition length in letters (default: 10)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="repetition_analysis.png",
        help="Output image path for repetition analysis diagram (default: repetition_analysis.png)."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without displaying GUI window (saves plot file only)."
    )

    args = parser.parse_args()

    # Step 1: Load / fetch sequence
    print(f"[*] Step 1: Loading DNA Sequence...")
    try:
        header, sequence = load_dna_sequence(filepath=args.file, accession=args.accession)
    except Exception as e:
        print(f"[!] Error loading DNA sequence: {e}", file=sys.stderr)
        return 1

    seq_len = len(sequence)
    print(f"    - Header: {header}")
    print(f"    - Length: {seq_len} nucleotides")
    print(f"    - Preview (first 50 nt): {sequence[:50]}...")
    print(f"    - Preview (last  50 nt): ...{sequence[-50:]}\n")

    # Step 2: Initialize Brute Force Engine
    print(f"[*] Step 2: Initializing Brute Force Combinatorial Engine...")
    engine = BruteForceRepetitionEngine(sequence)

    # Show combinatorial space
    print("    - Combinatorial motif space generated:")
    for k in range(3, 7):
        comb_count = 4 ** k
        print(f"      * {k}-letter combinations: 4^{k} = {comb_count} candidate motifs")
    total_combs = sum(4**k for k in range(3, 7))
    print(f"      * Total combinations evaluated: {total_combs}\n")

    # Step 3: Run Detection
    print(f"[*] Step 3: Detecting repetitions between {args.min_len} and {args.max_len} letters...")
    results = engine.run_all()
    tandem = results["tandem_repeats"]
    dispersed = results["dispersed_repeats"]

    print(f"    - Tandem Repetitions Found:    {len(tandem)}")
    print(f"    - Dispersed Repetitions Found: {len(dispersed)}\n")

    # Step 4: Display Tandem Repetitions Table
    print(f"[*] Step 4A: Tandem Repetitions Summary (Units: 3-6 letters, Total: 6-10 letters):")
    print("-" * 95)
    print(f"{'#':<4} {'Pattern':<14} {'Length':<8} {'Motif Unit':<12} {'Repeats':<9} {'Occurrences':<13} {'Start Positions'}")
    print("-" * 95)
    for idx, r in enumerate(tandem[:25], start=1):
        pos_str = ", ".join(map(str, r.positions[:5]))
        if len(r.positions) > 5:
            pos_str += f", ... (+{len(r.positions)-5} more)"
        print(f"{idx:<4} {r.pattern:<14} {r.length:<8} {r.motif + f' ({r.motif_length}bp)':<12} {r.repeat_count:<9} {r.occurrences:<13} [{pos_str}]")
    if len(tandem) > 25:
        print(f"... and {len(tandem) - 25} more tandem repeats.")
    print("-" * 95)

    # Step 5: Display Top Dispersed Repetitions Table
    print(f"\n[*] Step 4B: Top Dispersed Repetitions (Length: 6-10 letters, Frequent occurrences):")
    print("-" * 95)
    print(f"{'#':<4} {'Pattern':<14} {'Length':<8} {'Decomposition':<16} {'Occurrences':<13} {'Start Positions'}")
    print("-" * 95)
    for idx, r in enumerate(dispersed[:25], start=1):
        pos_str = ", ".join(map(str, r.positions[:6]))
        if len(r.positions) > 6:
            pos_str += f", ... (+{len(r.positions)-6} more)"
        print(f"{idx:<4} {r.pattern:<14} {r.length:<8} {r.motif:<16} {r.occurrences:<13} [{pos_str}]")
    if len(dispersed) > 25:
        print(f"... and {len(dispersed) - 25} more dispersed repeats.")
    print("-" * 95)

    # Step 6: Generate Visual Representation
    print(f"\n[*] Step 5: Generating Visual Analysis Diagram...")
    draw_repetition_analysis(
        sequence=sequence,
        tandem_repeats=tandem,
        dispersed_repeats=dispersed,
        output_path=args.output,
        show_window=not args.no_gui
    )

    print(f"\nEx 2 Analysis Complete! Diagram saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
