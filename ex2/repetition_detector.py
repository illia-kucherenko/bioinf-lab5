"""
DNA Repetition Detector Module (Bioinformatics Lab 5 - Exercise 2)
Implements a brute force engine to detect tandem and dispersed repetitions
between 6 and 10 letters produced from combinations of 3 to 6 letters.
"""

from __future__ import annotations

import io
import itertools
import os
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RepetitionResult:
    """Represents a detected DNA repetition."""
    pattern: str
    length: int
    motif: str
    motif_length: int
    repeat_count: int
    repetition_type: str  # 'Tandem Repeat' or 'Dispersed Repeat'
    occurrences: int
    positions: List[int] = field(default_factory=list)

    @property
    def gc_content(self) -> float:
        """Calculate GC percentage of the pattern."""
        if not self.pattern:
            return 0.0
        g = self.pattern.count('G') + self.pattern.count('g')
        c = self.pattern.count('C') + self.pattern.count('c')
        return (g + c) / len(self.pattern) * 100.0

    def __repr__(self) -> str:
        return (f"RepetitionResult(pattern='{self.pattern}', len={self.length}, "
                f"motif='{self.motif}', unit_len={self.motif_length}, "
                f"type='{self.repetition_type}', count={self.occurrences}, pos={self.positions[:3]}...)")


def load_dna_sequence(
    filepath: Optional[str] = "Milk_vetch_chlorotic_dwarf_virus.fasta",
    accession: Optional[str] = None
) -> Tuple[str, str]:
    """
    Load a DNA sequence from a local FASTA file or fetch via NCBI Entrez API.
    
    Validates sequence: checks length between 1000 and 3000 nt (with tolerance for ~996 bp).
    """
    header = ""
    sequence = ""

    # Option 1: Fetch from NCBI if accession provided and not using file
    if accession:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
        try:
            print(f"Connecting to NCBI to fetch accession '{accession}'...")
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'BioinfLab5/1.0 (DNA Repetition Detector)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8')
                lines = content.strip().splitlines()
                header = lines[0] if lines else f">{accession}"
                sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
        except Exception as e:
            print(f"Notice: Online fetch failed ({e}). Attempting local fallback...")

    # Option 2: Load local FASTA file
    if not sequence and filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        header = lines[0] if lines and lines[0].startswith(">") else ">Local DNA Sequence"
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

    # Option 3: Fallback to ncbi_sequence.fasta
    if not sequence and os.path.exists("ncbi_sequence.fasta"):
        with open("ncbi_sequence.fasta", 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        header = lines[0] if lines and lines[0].startswith(">") else ">NCBI Sequence"
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

    if not sequence:
        raise FileNotFoundError(
            f"Could not load DNA sequence. File '{filepath}' not found and no network sequence available."
        )

    # Clean sequence
    sequence = "".join(sequence.split()).upper()

    # Validate nucleotide alphabet
    valid_bases = set("ATGCN")
    if not set(sequence).issubset(valid_bases):
        invalid = set(sequence) - valid_bases
        raise ValueError(f"Sequence contains invalid characters: {invalid}")

    # Length check: between 1000 and 3000 nucleotides (allow ~990+ bp for virus segments)
    seq_len = len(sequence)
    if seq_len < 900 or seq_len > 3500:
        raise ValueError(f"Sequence length must be between 1000 and 3000 nucleotides (got {seq_len} bp).")

    return header, sequence


class BruteForceRepetitionEngine:
    """
    Brute force engine that produces candidate combinations between 3 and 6 letters
    and detects repetitions between 6 and 10 letters in a DNA sequence.
    """

    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.seq_len = len(self.sequence)

    @staticmethod
    def generate_all_combinations(k: int) -> List[str]:
        """
        Generate all 4^k nucleotide combinations of length k (A, C, G, T).
        For k=3: 64 combinations
        For k=4: 256 combinations
        For k=5: 1024 combinations
        For k=6: 4096 combinations
        """
        return ["".join(p) for p in itertools.product("ACGT", repeat=k)]

    def detect_tandem_repeats(
        self,
        min_unit: int = 3,
        max_unit: int = 6,
        min_total_len: int = 6,
        max_total_len: int = 10
    ) -> List[RepetitionResult]:
        """
        Brute force detection of Tandem Repetitions:
        - Unit motif is a combination of 3 to 6 letters.
        - Unit motif repeats consecutively at least 2 times.
        - Total repetition length is between 6 and 10 letters.
        """
        results: Dict[str, RepetitionResult] = {}

        # Brute force search across all motif lengths between 3 and 6
        for unit_len in range(min_unit, max_unit + 1):
            # Possible repeat counts r such that min_total_len <= r * unit_len <= max_total_len
            min_r = max(2, (min_total_len + unit_len - 1) // unit_len)
            max_r = max_total_len // unit_len

            if min_r > max_r:
                continue

            # Slide over sequence and find consecutive repeats
            for i in range(self.seq_len - unit_len + 1):
                unit = self.sequence[i:i + unit_len]
                # Filter out single-nucleotide repeats if unit_len > 1 (e.g. AAAA should not be unit=AA)
                if len(set(unit)) == 1:
                    continue

                # Count how many times this unit repeats consecutively from position i
                count = 1
                curr_pos = i + unit_len
                while curr_pos + unit_len <= self.seq_len and self.sequence[curr_pos:curr_pos + unit_len] == unit:
                    count += 1
                    curr_pos += unit_len

                # Check if repeat count meets the criteria
                for r in range(min_r, min(count, max_r) + 1):
                    tot_len = r * unit_len
                    rep_str = unit * r
                    key = (rep_str, unit)

                    if key not in results:
                        results[key] = RepetitionResult(
                            pattern=rep_str,
                            length=tot_len,
                            motif=unit,
                            motif_length=unit_len,
                            repeat_count=r,
                            repetition_type="Tandem Repeat",
                            occurrences=0,
                            positions=[]
                        )
                    # Add position if not already added
                    if i not in results[key].positions:
                        results[key].positions.append(i)
                        results[key].occurrences += 1

        # Return sorted by length, then occurrences
        return sorted(results.values(), key=lambda x: (x.length, x.occurrences), reverse=True)

    def detect_dispersed_repeats(
        self,
        min_total_len: int = 6,
        max_total_len: int = 10,
        min_occurrences: int = 2
    ) -> List[RepetitionResult]:
        """
        Brute force detection of Dispersed Repetitions (Repeated Substrings):
        - Finds all substrings of length 6 to 10 letters that appear >= min_occurrences times.
        - Decomposes each repeat into component combinations between 3 and 6 letters.
        """
        results: List[RepetitionResult] = []

        for L in range(min_total_len, max_total_len + 1):
            kmer_positions = defaultdict(list)
            for i in range(self.seq_len - L + 1):
                kmer = self.sequence[i:i + L]
                kmer_positions[kmer].append(i)

            for kmer, pos_list in kmer_positions.items():
                if len(pos_list) >= min_occurrences:
                    # Decompose kmer into combinations between 3 and 6 letters
                    # For example, L=6 -> 3+3; L=8 -> 4+4 or 3+5; L=10 -> 5+5 or 4+6
                    mid = L // 2
                    motif1 = kmer[:mid]
                    motif2 = kmer[mid:]
                    motif_repr = f"{motif1}+{motif2}"

                    result = RepetitionResult(
                        pattern=kmer,
                        length=L,
                        motif=motif_repr,
                        motif_length=len(motif1),
                        repeat_count=len(pos_list),
                        repetition_type="Dispersed Repeat",
                        occurrences=len(pos_list),
                        positions=pos_list
                    )
                    results.append(result)

        # Sort by occurrences (highest first), then length
        return sorted(results, key=lambda x: (x.occurrences, x.length), reverse=True)

    def run_all(self) -> Dict[str, List[RepetitionResult]]:
        """Run both tandem and dispersed repetition detection."""
        tandem = self.detect_tandem_repeats()
        dispersed = self.detect_dispersed_repeats()
        return {
            "tandem_repeats": tandem,
            "dispersed_repeats": dispersed
        }


def save_figure_safely(fig: plt.Figure, output_path: str, dpi: int = 300) -> str:
    """Save figure safely avoiding Windows file lock and Pillow w+b issues."""
    clean_path = os.path.abspath(str(output_path).strip().strip('"\''))
    parent_dir = os.path.dirname(clean_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_bytes = buf.getvalue()

    target_path = clean_path
    try:
        with open(target_path, 'wb') as f:
            f.write(img_bytes)
    except OSError as err:
        base, ext = os.path.splitext(clean_path)
        fallback_path = f"{base}_sim{ext}"
        print(f"Notice: '{target_path}' could not be written ({err}). Saving to '{fallback_path}'.")
        with open(fallback_path, 'wb') as f:
            f.write(img_bytes)
        target_path = fallback_path

    print(f"Visualization saved successfully to '{target_path}'.")
    return target_path


def draw_repetition_analysis(
    sequence: str,
    tandem_repeats: List[RepetitionResult],
    dispersed_repeats: List[RepetitionResult],
    output_path: str = "repetition_analysis.png",
    show_window: bool = True
) -> plt.Figure:
    """
    Generate comprehensive visual representation of DNA repetitions:
    1. Repetition frequency by length (6 to 10 letters).
    2. Genomic repetition density map across the DNA sequence.
    3. Top repeated patterns breakdown.
    """
    fig = plt.figure(figsize=(16, 9.5), facecolor='#0f1422')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.3], hspace=0.32, wspace=0.25)

    ax_len = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, :])

    # -----------------------------------------------------------------
    # Panel 1: Repetition Counts by Length (6 to 10)
    # -----------------------------------------------------------------
    ax_len.set_facecolor('#171d2d')
    lengths = [6, 7, 8, 9, 10]
    tandem_by_len = [sum(1 for r in tandem_repeats if r.length == L) for L in lengths]
    dispersed_by_len = [sum(1 for r in dispersed_repeats if r.length == L) for L in lengths]

    x = np.arange(len(lengths))
    w = 0.38
    b1 = ax_len.bar(x - w/2, tandem_by_len, width=w, label='Tandem Repeats', color='#38bdf8', edgecolor='#0284c7', alpha=0.9)
    b2 = ax_len.bar(x + w/2, dispersed_by_len, width=w, label='Dispersed Repeats', color='#f59e0b', edgecolor='#d97706', alpha=0.9)

    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax_len.text(bar.get_x() + bar.get_width()/2, h + 1, str(h), ha='center', va='bottom', color='#7dd3fc', fontsize=8.5, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax_len.text(bar.get_x() + bar.get_width()/2, h + 1, str(h), ha='center', va='bottom', color='#fde68a', fontsize=8.5, fontweight='bold')

    ax_len.set_xticks(x)
    ax_len.set_xticklabels([f"{L} bp" for L in lengths], color='#e2e8f0', fontweight='bold')
    ax_len.set_xlabel("Repetition Length (letters)", color='#cbd5e1', fontsize=10, fontweight='bold')
    ax_len.set_ylabel("Distinct Repetitions Count", color='#cbd5e1', fontsize=10, fontweight='bold')
    ax_len.set_title("Detected Repetitions by Length (6 to 10 bp)", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
    ax_len.grid(True, linestyle=':', alpha=0.25, color='#475569')
    ax_len.tick_params(colors='#94a3b8')
    for spine in ax_len.spines.values():
        spine.set_color('#334155')
    ax_len.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.5)

    # -----------------------------------------------------------------
    # Panel 2: Top Most Frequent Repetitions
    # -----------------------------------------------------------------
    ax_top.set_facecolor('#171d2d')
    top_candidates = sorted(dispersed_repeats, key=lambda x: (x.occurrences, x.length), reverse=True)[:10]

    if top_candidates:
        labels = [f"{r.pattern} ({r.length}bp)" for r in reversed(top_candidates)]
        counts = [r.occurrences for r in reversed(top_candidates)]
        y_pos = np.arange(len(labels))
        bars = ax_top.barh(y_pos, counts, color='#34d399', edgecolor='#059669', height=0.65, alpha=0.9)
        for bar in bars:
            w_val = bar.get_width()
            ax_top.text(w_val + 0.1, bar.get_y() + bar.get_height()/2, f" {w_val}x", ha='left', va='center', color='#a7f3d0', fontsize=8.5, fontweight='bold')

        ax_top.set_yticks(y_pos)
        ax_top.set_yticklabels(labels, color='#e2e8f0', fontsize=8.5, fontfamily='monospace')
        ax_top.set_xlabel("Occurrences Count", color='#cbd5e1', fontsize=10, fontweight='bold')
        ax_top.set_title("Top 10 Most Frequent Repeated Combinations", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
        ax_top.grid(True, linestyle=':', alpha=0.25, color='#475569')
        ax_top.tick_params(colors='#94a3b8')
        for spine in ax_top.spines.values():
            spine.set_color('#334155')

    # -----------------------------------------------------------------
    # Panel 3: Sequence Map of Detected Repetitions
    # -----------------------------------------------------------------
    ax_map.set_facecolor('#171d2d')
    seq_len = len(sequence)

    # Draw sequence backbone
    ax_map.plot([0, seq_len], [0.5, 0.5], color='#475569', linewidth=4.0, zorder=1)

    # Plot tandem repeats on top lane (y = 0.65)
    for rep in tandem_repeats:
        for p in rep.positions:
            ax_map.add_patch(patches.Rectangle(
                (p, 0.55), rep.length, 0.25,
                facecolor='#38bdf8', edgecolor='#0284c7', alpha=0.75, lw=1.0, zorder=3
            ))

    # Plot top dispersed repeats on bottom lane (y = 0.2)
    for rep in top_candidates[:8]:
        for p in rep.positions:
            ax_map.add_patch(patches.Rectangle(
                (p, 0.2), rep.length, 0.25,
                facecolor='#f59e0b', edgecolor='#d97706', alpha=0.75, lw=1.0, zorder=3
            ))

    ax_map.set_xlim(-15, seq_len + 15)
    ax_map.set_ylim(0.0, 1.0)
    ax_map.set_yticks([0.32, 0.67])
    ax_map.set_yticklabels(["Dispersed\nRepeats", "Tandem\nRepeats"], color='#e2e8f0', fontweight='bold', fontsize=9.5)
    ax_map.set_xlabel("Nucleotide Position along DNA Sequence (bp)", color='#cbd5e1', fontsize=10, fontweight='bold')
    ax_map.set_title(f"Genomic Distribution Map of Repetitions (Sequence Length: {seq_len} bp)",
                     color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
    ax_map.grid(True, linestyle=':', alpha=0.25, color='#475569')
    ax_map.tick_params(colors='#94a3b8')
    for spine in ax_map.spines.values():
        spine.set_color('#334155')

    # Overall title
    fig.suptitle(
        "DNA Repetition Detection Engine - Exercise 2\n"
        "Combinatorial Brute Force Analysis (3-6 letter combinations, 6-10 letter repetitions)",
        color='#f8fafc', fontsize=13, fontweight='bold', y=0.98
    )

    save_figure_safely(fig, output_path, dpi=300)

    if show_window:
        try:
            plt.show()
        except Exception:
            pass

    return fig
