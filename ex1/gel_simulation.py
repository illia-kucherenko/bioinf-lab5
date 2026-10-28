"""
Gel Electrophoresis Simulation Module
Simulates agarose gel electrophoresis for arbitrary DNA fragments obtained from NCBI.
"""

from __future__ import annotations

import io
import math
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DNAFragment:
    """Represents a DNA fragment sample."""
    sample_id: int
    sequence: str
    length: int
    start_pos: int
    end_pos: int
    molecular_weight_da: float
    molecular_weight_kda: float
    migration_distance: float

    @property
    def gc_content(self) -> float:
        """Calculate GC content percentage."""
        if not self.sequence:
            return 0.0
        g_count = self.sequence.count('G') + self.sequence.count('g')
        c_count = self.sequence.count('C') + self.sequence.count('c')
        return (g_count + c_count) / len(self.sequence) * 100.0

    def __repr__(self) -> str:
        seq_preview = self.sequence[:15] + "..." + self.sequence[-15:] if len(self.sequence) > 35 else self.sequence
        return (f"DNAFragment(id={self.sample_id}, length={self.length} bp, "
                f"MW={self.molecular_weight_kda:.2f} kDa, migration={self.migration_distance:.3f}, "
                f"range=[{self.start_pos}:{self.end_pos}], seq='{seq_preview}')")


def calculate_molecular_weight(sequence: str, is_dsdna: bool = True) -> float:
    """
    Calculate the molecular weight of a DNA fragment in Daltons (g/mol).
    
    For double-stranded DNA (dsDNA):
    Uses accurate average mononucleotide monophosphate weights:
    dAMP: 313.21 Da, dTMP: 304.20 Da, dCMP: 289.18 Da, dGMP: 329.21 Da.
    Terminal phosphates and OH adjustments are included.
    Average dsDNA base pair is ~650 Da.
    """
    seq = sequence.upper()
    a = seq.count('A')
    t = seq.count('T')
    c = seq.count('C')
    g = seq.count('G')
    
    # In case there are ambiguous bases (N, R, Y, etc.)
    other = len(seq) - (a + t + c + g)
    if other > 0:
        # Approximate other with average nucleotide mass (~309.2 Da)
        ss_mw = (a * 313.21 + t * 304.20 + c * 289.18 + g * 329.21 + other * 309.2) - 61.96
    else:
        ss_mw = (a * 313.21 + t * 304.20 + c * 289.18 + g * 329.21) - 61.96

    if is_dsdna:
        # Complementary strand counts: A pairs with T, C pairs with G
        comp_a, comp_t, comp_c, comp_g = t, a, g, c
        comp_ss_mw = (comp_a * 313.21 + comp_t * 304.20 + comp_c * 289.18 + comp_g * 329.21 + other * 309.2) - 61.96
        return ss_mw + comp_ss_mw
    return ss_mw


def fetch_ncbi_sequence(
    accession: str = "X52331.1",
    cache_path: str = "ncbi_sequence.fasta"
) -> Tuple[str, str]:
    """
    Fetch a DNA sequence from NCBI Entrez API, or load from local cache if offline.
    
    Validates that the sequence length is between 1000 and 3000 nucleotides.
    """
    sequence = ""
    header = ""

    # Try fetching online first
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
    try:
        print(f"Connecting to NCBI Entrez to fetch accession '{accession}'...")
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'BioinfLab5/1.0 (DNA Gel Electrophoresis Simulation)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')
            lines = content.strip().splitlines()
            if lines and lines[0].startswith(">"):
                header = lines[0]
                sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
                # Save cache for offline use
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Successfully fetched and cached '{accession}' ({len(sequence)} bp).")
    except Exception as e:
        print(f"Notice: Online fetch encountered ({e}). Checking local cache '{cache_path}'...")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = content.strip().splitlines()
            header = lines[0] if lines else f">{accession} cached"
            sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
            print(f"Successfully loaded '{accession}' from local cache ({len(sequence)} bp).")
        else:
            raise RuntimeError(
                f"Failed to fetch accession '{accession}' from NCBI and no cache file found at '{cache_path}'."
            ) from e

    # Clean sequence: strip whitespace and validate
    sequence = "".join(sequence.split()).upper()
    seq_len = len(sequence)

    if not (1000 <= seq_len <= 3000):
        raise ValueError(
            f"NCBI Sequence length must be between 1000 and 3000 nucleotides (got {seq_len} bp)."
        )

    return header, sequence


def calculate_migration_distance(
    length_bp: float,
    min_ref_bp: float = 60.0,
    max_ref_bp: float = 4000.0
) -> float:
    """
    Simulate relative migration distance (Rf, from 0.0 to 1.0) through electrophoresis gel.
    
    Physical principle:
    DNA has a uniform negative charge-to-mass ratio due to its phosphate backbone.
    During agarose gel electrophoresis:
    - Long fragments experience high friction force with the agarose mesh and migrate slowly.
    - Short fragments experience low friction force and migrate quickly and farther.
    - Migration distance is inversely proportional to log10(fragment size / molecular weight):
      Rf = (log10(max_bp) - log10(length)) / (log10(max_bp) - log10(min_bp))
    """
    length = max(float(min_ref_bp), min(float(max_ref_bp), float(length_bp)))
    log_max = math.log10(max_ref_bp)
    log_min = math.log10(min_ref_bp)
    log_len = math.log10(length)
    
    rf = (log_max - log_len) / (log_max - log_min)
    return float(np.clip(rf, 0.0, 1.0))


def generate_dna_samples(
    sequence: str,
    num_samples: int = 10,
    min_length: int = 100,
    max_length: int = 3000,
    seed: Optional[int] = None
) -> List[DNAFragment]:
    """
    Take random samples from the DNA sequence between min_length and max_length bases.
    Stores the samples in an array (list of DNAFragment objects).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    seq_len = len(sequence)
    effective_max_len = min(max_length, seq_len)
    
    if min_length > effective_max_len:
        raise ValueError(f"min_length ({min_length}) cannot be greater than sequence length ({seq_len})")

    samples: List[DNAFragment] = []

    for i in range(1, num_samples + 1):
        # Sample length between min_length and max_length
        frag_len = random.randint(min_length, effective_max_len)
        max_start = seq_len - frag_len
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        end_idx = start_idx + frag_len
        
        frag_seq = sequence[start_idx:end_idx]
        mw_da = calculate_molecular_weight(frag_seq, is_dsdna=True)
        mw_kda = mw_da / 1000.0
        rf = calculate_migration_distance(frag_len)

        fragment = DNAFragment(
            sample_id=i,
            sequence=frag_seq,
            length=frag_len,
            start_pos=start_idx,
            end_pos=end_idx,
            molecular_weight_da=mw_da,
            molecular_weight_kda=mw_kda,
            migration_distance=rf
        )
        samples.append(fragment)

    return samples


def save_figure_safely(fig: plt.Figure, output_path: str, dpi: int = 300) -> str:
    """
    Save figure safely across Windows/Linux/macOS platforms.
    
    Renders directly to an in-memory BytesIO buffer first, bypassing Pillow's
    builtins.open(filename, 'w+b') which causes 'OSError: [Errno 22] Invalid argument'
    on Windows when locked files or strict file sharing modes are encountered.
    """
    clean_path = os.path.abspath(str(output_path).strip().strip('"\''))
    parent_dir = os.path.dirname(clean_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img_bytes = buf.getvalue()

    # Try saving to the requested clean path
    target_path = clean_path
    try:
        with open(target_path, 'wb') as f:
            f.write(img_bytes)
    except OSError as err:
        # If the file is locked by PyCharm's image tab or another viewer, fallback to an alternate name
        base, ext = os.path.splitext(clean_path)
        fallback_path = f"{base}_sim{ext}"
        print(f"Notice: '{target_path}' could not be overwritten directly ({err}). Saving to '{fallback_path}' instead.")
        with open(fallback_path, 'wb') as f:
            f.write(img_bytes)
        target_path = fallback_path

    print(f"Visualization saved successfully to '{target_path}'.")
    return target_path


def draw_gel_electrophoresis(
    samples: List[DNAFragment],
    reference_ladder_bp: List[int] = [3000, 1500, 500],
    output_path: str = "gel_electrophoresis.png",
    show_window: bool = True
) -> plt.Figure:
    """
    Draw a visual representation of the gel electrophoresis migration.
    
    Requirements fulfilled:
    - Wells at top (Cathode -)
    - Migration towards bottom (Anode +)
    - 10 sample lanes
    - Combined mixture lane
    - Reference ladder lane
    - Diagram labels on the right: 3000 bp, 1500 bp, 500 bp
    - Visual representation of friction force principle
    """
    # Create figure with 2 subplots:
    # 1. Authentic Gel Electrophoresis Tank visualization
    # 2. Semi-log Migration Calibration Curve (Distance vs Molecular Weight / bp)
    fig = plt.figure(figsize=(15, 8.5), facecolor='#0b0d14')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.25)
    
    ax_gel = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])

    # -------------------------------------------------------------
    # SUBPLOT 1: GEL ELECTROPHORESIS TANK
    # -------------------------------------------------------------
    ax_gel.set_facecolor('#05070c')

    # Gel dimensions
    # Total lanes:
    # Lanes 1..10: Samples 1 to 10
    # Lane 11: Sample Mix (all 10 samples pooled)
    # Lane 12: Reference Ladder (with 3000 bp, 1500 bp, 500 bp, etc.)
    num_sample_lanes = len(samples)
    lane_mix_idx = num_sample_lanes + 1
    lane_ladder_idx = num_sample_lanes + 2
    total_lanes = lane_ladder_idx

    gel_x_min, gel_x_max = 0.2, total_lanes + 0.8
    gel_y_min, gel_y_max = 0.0, 10.0

    # Draw agarose gel slab background
    gel_slab = patches.FancyBboxPatch(
        (gel_x_min, 0.4), gel_x_max - gel_x_min, 9.2,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor='#0f1422', edgecolor='#2e3856', linewidth=2.0, zorder=1
    )
    ax_gel.add_patch(gel_slab)

    # Well parameters
    well_y = 8.8
    well_w = 0.55
    well_h = 0.35

    # Migration coordinate mapping:
    # From well base (y = 8.6) to bottom limit (y = 1.0)
    top_travel_y = 8.5
    bottom_travel_y = 1.2
    travel_span = top_travel_y - bottom_travel_y

    def bp_to_y(bp: float) -> float:
        rf = calculate_migration_distance(bp)
        return top_travel_y - rf * travel_span

    def draw_band(ax, x_center: float, y_center: float, width: float = 0.52, intensity: float = 1.0, color: str = '#38bdf8'):
        """Draws a glowing DNA band imitating ethidium bromide / SYBR UV fluorescence."""
        # Multi-layer glow
        # Outer soft glow
        ax.add_patch(patches.Rectangle(
            (x_center - width * 0.6, y_center - 0.14), width * 1.2, 0.28,
            facecolor=color, alpha=0.18 * intensity, edgecolor='none', zorder=3
        ))
        # Mid glow
        ax.add_patch(patches.Rectangle(
            (x_center - width * 0.5, y_center - 0.08), width, 0.16,
            facecolor=color, alpha=0.45 * intensity, edgecolor='none', zorder=4
        ))
        # Bright core
        ax.add_patch(patches.Rectangle(
            (x_center - width * 0.42, y_center - 0.035), width * 0.84, 0.07,
            facecolor='#ffffff', alpha=0.85 * min(1.0, intensity * 1.1), edgecolor='none', zorder=5
        ))

    # Draw Wells and Sample Bands
    for idx, sample in enumerate(samples, start=1):
        # Draw well
        ax_gel.add_patch(patches.Rectangle(
            (idx - well_w / 2, well_y - well_h / 2), well_w, well_h,
            facecolor='#060810', edgecolor='#475569', linewidth=1.2, zorder=2
        ))
        # Label above well
        ax_gel.text(idx, 9.25, f"S{idx}", color='#94a3b8', fontsize=8.5,
                    ha='center', va='bottom', fontweight='bold')
        # Sub-label for length
        ax_gel.text(idx, 9.05, f"{sample.length}", color='#64748b', fontsize=6.8,
                    ha='center', va='bottom')

        # Draw DNA band in sample lane
        band_y = bp_to_y(sample.length)
        # Intensity scaled slightly by fragment length (longer fragments bind more dye)
        intensity = 0.75 + 0.35 * (sample.length / 3000.0)
        draw_band(ax_gel, idx, band_y, width=0.52, intensity=intensity, color='#38bdf8')

    # Draw Sample Mix Lane (Lane 11)
    ax_gel.add_patch(patches.Rectangle(
        (lane_mix_idx - well_w / 2, well_y - well_h / 2), well_w, well_h,
        facecolor='#060810', edgecolor='#38bdf8', linewidth=1.4, zorder=2
    ))
    ax_gel.text(lane_mix_idx, 9.25, "Mix", color='#38bdf8', fontsize=8.5,
                ha='center', va='bottom', fontweight='bold')
    ax_gel.text(lane_mix_idx, 9.05, "10 frags", color='#0284c7', fontsize=6.8,
                ha='center', va='bottom')

    # Bands in Mix Lane
    for sample in samples:
        band_y = bp_to_y(sample.length)
        draw_band(ax_gel, lane_mix_idx, band_y, width=0.50, intensity=0.8, color='#38bdf8')

    # Draw Reference Ladder Lane (Lane 12)
    ax_gel.add_patch(patches.Rectangle(
        (lane_ladder_idx - well_w / 2, well_y - well_h / 2), well_w, well_h,
        facecolor='#060810', edgecolor='#f59e0b', linewidth=1.4, zorder=2
    ))
    ax_gel.text(lane_ladder_idx, 9.25, "Ladder", color='#f59e0b', fontsize=8.5,
                ha='center', va='bottom', fontweight='bold')
    ax_gel.text(lane_ladder_idx, 9.05, "Marker", color='#d97706', fontsize=6.8,
                ha='center', va='bottom')

    # Standard ladder bands: mandatory [3000, 1500, 500] plus context standards [2000, 1000, 250, 100]
    all_ladder_bp = [3000, 2000, 1500, 1000, 500, 250, 100]
    for bp in all_ladder_bp:
        band_y = bp_to_y(bp)
        is_key = bp in reference_ladder_bp
        col = '#fbbf24' if is_key else '#d97706'
        intens = 1.0 if is_key else 0.6
        draw_band(ax_gel, lane_ladder_idx, band_y, width=0.52, intensity=intens, color=col)

    # -------------------------------------------------------------
    # DIAGRAM LABELS (RIGHT): 3000 bp, 1500 bp, 500 bp
    # -------------------------------------------------------------
    for bp in reference_ladder_bp:
        y_pos = bp_to_y(bp)
        # Guide dashed line from ladder to right label
        ax_gel.plot([lane_ladder_idx + 0.32, total_lanes + 0.65], [y_pos, y_pos],
                    color='#fbbf24', linestyle=':', linewidth=1.2, alpha=0.8, zorder=6)
        # Right label text
        ax_gel.text(
            total_lanes + 0.72, y_pos, f"  {bp} bp",
            color='#fbbf24', fontsize=9.5, fontweight='bold',
            va='center', ha='left',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#1e2235', edgecolor='#f59e0b', alpha=0.9, lw=1.2),
            zorder=7
        )

    # Electrode Polarity Indicators
    # Cathode (-) at top
    ax_gel.text(0.45, 9.75, "(-) Cathode (Well loading end)", color='#60a5fa',
                fontsize=9.5, fontweight='bold', ha='left', va='center')
    # Anode (+) at bottom
    ax_gel.text(0.45, 0.2, "(+) Anode (Migration direction  ↓)", color='#f87171',
                fontsize=9.5, fontweight='bold', ha='left', va='center')

    # Friction Force Physical Note on gel
    ax_gel.annotate(
        "Low Friction (Fast migration) ↓",
        xy=(lane_mix_idx + 0.1, 1.4), xytext=(lane_mix_idx - 3.5, 0.9),
        arrowprops=dict(facecolor='#34d399', edgecolor='none', width=1.5, headwidth=6, shrink=0.08),
        color='#34d399', fontsize=8.5, fontweight='bold', zorder=8
    )
    ax_gel.annotate(
        "High Friction (Slow migration) ↑",
        xy=(lane_mix_idx + 0.1, 8.0), xytext=(lane_mix_idx - 3.5, 8.2),
        arrowprops=dict(facecolor='#f43f5e', edgecolor='none', width=1.5, headwidth=6, shrink=0.08),
        color='#f43f5e', fontsize=8.5, fontweight='bold', zorder=8
    )

    ax_gel.set_xlim(-0.2, total_lanes + 2.0)
    ax_gel.set_ylim(-0.1, 10.1)
    ax_gel.set_title(
        "Simulated Agarose Gel Electrophoresis (1.0% Agarose)\n"
        "DNA Samples randomly sampled from NCBI sequence",
        color='#f8fafc', fontsize=12, fontweight='bold', pad=12
    )
    ax_gel.set_xticks([])
    ax_gel.set_yticks([])
    for spine in ax_gel.spines.values():
        spine.set_visible(False)

    # -------------------------------------------------------------
    # SUBPLOT 2: SEMI-LOG MIGRATION CALIBRATION DYNAMICS
    # -------------------------------------------------------------
    ax_curve.set_facecolor('#0f1422')
    ax_curve.tick_params(colors='#94a3b8', labelsize=8.5)
    for spine in ax_curve.spines.values():
        spine.set_color('#334155')

    # Theoretical line: Distance vs log10(bp)
    fit_bp = np.logspace(np.log10(100), np.log10(3500), 200)
    fit_rf = [calculate_migration_distance(b) for b in fit_bp]
    ax_curve.plot(fit_bp, fit_rf, color='#64748b', linestyle='--', linewidth=1.5,
                  label='Theoretical Ferguson curve ($R_f \\propto -\\log_{10}(bp)$)')

    # Plot Reference Ladder points
    ladder_rfs = [calculate_migration_distance(b) for b in reference_ladder_bp]
    ax_curve.scatter(reference_ladder_bp, ladder_rfs, color='#fbbf24', s=70, zorder=5,
                     edgecolor='#ffffff', linewidth=1.2, label='Reference Standards (3000, 1500, 500 bp)')
    for bp, rf in zip(reference_ladder_bp, ladder_rfs):
        ax_curve.annotate(f"{bp} bp", (bp, rf), textcoords="offset points", xytext=(8, -3),
                          color='#fbbf24', fontsize=8.5, fontweight='bold')

    # Plot the 10 Sample points
    sample_bps = [s.length for s in samples]
    sample_rfs = [s.migration_distance for s in samples]
    ax_curve.scatter(sample_bps, sample_rfs, color='#38bdf8', s=60, zorder=6,
                     edgecolor='#ffffff', linewidth=1.0, label='10 NCBI DNA Samples (S1-S10)')

    for s in samples:
        ax_curve.text(s.length, s.migration_distance + 0.025, f"S{s.sample_id}",
                      color='#7dd3fc', fontsize=7.5, ha='center')

    ax_curve.set_xscale('log')
    ax_curve.set_xlabel("DNA Fragment Length (base pairs, log scale)", color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax_curve.set_ylabel("Relative Migration Distance ($R_f$)", color='#e2e8f0', fontsize=9.5, fontweight='bold')
    ax_curve.set_title("Mobility Dynamics vs Molecular Size\nInverse Logarithmic Migration", color='#f8fafc', fontsize=11, fontweight='bold', pad=12)
    ax_curve.grid(True, linestyle=':', alpha=0.3, color='#475569')
    ax_curve.set_ylim(-0.05, 1.05)

    leg = ax_curve.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8.0, loc='upper right')
    leg.get_frame().set_alpha(0.9)

    # Explanation text box
    ax_curve.text(
        0.05, 0.06,
        "Physical Law: Friction Force $\\propto$ Length\n"
        "• High MW (3000 bp): High friction $\\rightarrow$ Slow migration\n"
        "• Low MW (500 bp): Low friction $\\rightarrow$ Fast migration\n"
        "• Migration equation: $D = a - b \\cdot \\log_{10}(MW)$",
        transform=ax_curve.transAxes,
        color='#cbd5e1', fontsize=8.0, va='bottom',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='#1e2438', edgecolor='#3b82f6', alpha=0.85, lw=1.0)
    )

    # Save figure safely (handles in-memory rendering and Windows file locks)
    save_figure_safely(fig, output_path, dpi=300)

    if show_window:
        try:
            plt.show()
        except Exception:
            pass

    return fig
