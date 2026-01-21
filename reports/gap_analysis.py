"""
Gap Analysis Report Generator

Generates comprehensive reports on allocation gaps and restriction impacts.
Outputs to Excel format with multiple sheets for easy analysis.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import numpy as np

from ..utils.class_names import get_class_name, format_class_list

logger = logging.getLogger(__name__)


def generate_gap_analysis_reports(
    tracker,
    targets_df: pd.DataFrame,
    areas_df: pd.DataFrame,
    restrictions_metadata: Dict[str, Any],
    output_dir: Path,
    restriction_level: int = 3
) -> Dict[str, Any]:
    """
    Generate comprehensive gap analysis reports.
    
    Outputs:
        - gap_analysis_report_levelX.xlsx: Comprehensive Excel workbook
        - gap_analysis_report.txt: Human-readable summary
        - blocked_transitions.parquet: Full transition database
    
    Parameters
    ----------
    tracker : AllocationTracker
        Tracker with allocation data
    targets_df : pd.DataFrame
        Targets dataframe
    areas_df : pd.DataFrame
        Areas dataframe
    restrictions_metadata : dict
        Restrictions metadata from JSON
    output_dir : Path
        Output directory
    restriction_level : int, default=3
        Restriction level for report naming
    
    Returns
    -------
    dict
        Dictionary with report dataframes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("GENERATING GAP ANALYSIS REPORTS")
    logger.info("="*70)
    
    # 1. Gap Summary Report
    logger.info("Creating gap summary...")
    gap_df = _generate_gap_summary(tracker, targets_df)
    
    # 2. Restriction Impact Report
    logger.info("Creating restriction impact report...")
    impact_df = _generate_restriction_impact_report(tracker)
    
    # 3. Frozen Classes Detail
    logger.info("Creating frozen classes detail...")
    frozen_df = _generate_frozen_classes_detail(tracker, areas_df, targets_df)
    
    # 4. Blocked Transitions Database
    logger.info("Processing blocked transitions...")
    blocked_df = None
    if tracker.blocked_transitions:
        blocked_df = pd.DataFrame(tracker.blocked_transitions)
        # Save full database to Parquet
        blocked_df.to_parquet(output_dir / 'blocked_transitions.parquet')
        logger.info(f"Saved {len(blocked_df):,} blocked transitions to Parquet")
    
    # 5. Recommendations
    logger.info("Generating recommendations...")
    recommendations = _generate_recommendations(
        tracker, gap_df, frozen_df, restrictions_metadata
    )
    with open(output_dir / 'recommendations.txt', 'w') as f:
        f.write(recommendations)
    
    # 6. Human-Readable Report
    logger.info("Generating human-readable report...")
    report_text = _generate_human_readable_report(
        tracker, gap_df, impact_df, frozen_df, restrictions_metadata
    )
    with open(output_dir / 'gap_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    # 7. Comprehensive Excel Workbook
    logger.info("Creating Excel workbook...")
    excel_path = output_dir / f'gap_analysis_report_level{restriction_level}.xlsx'
    _generate_excel_workbook(
        excel_path, gap_df, impact_df, frozen_df, blocked_df,
        recommendations, report_text, tracker, restrictions_metadata
    )
    
    logger.info("="*70)
    logger.info("GAP ANALYSIS REPORTS COMPLETE")
    logger.info("="*70)
    logger.info(f"Excel report: {excel_path}")
    logger.info(f"Text report: {output_dir / 'gap_analysis_report.txt'}")
    logger.info(f"Recommendations: {output_dir / 'recommendations.txt'}")
    logger.info("="*70)
    
    return {
        'gap_summary': gap_df,
        'restriction_impact': impact_df,
        'frozen_classes': frozen_df,
        'blocked_transitions': blocked_df,
        'recommendations': recommendations,
        'excel_path': str(excel_path)
    }


def _generate_gap_summary(tracker, targets_df) -> pd.DataFrame:
    """Generate target vs achieved summary."""
    rows = []
    for lu_class, gap_info in tracker.target_gaps.items():
        rows.append({
            'lu_class': lu_class,
            'lu_name': get_class_name(lu_class),
            'target_ha': gap_info['target'],
            'achieved_ha': gap_info['achieved'],
            'gap_ha': gap_info['gap'],
            'pct_achieved': gap_info['pct_achieved'],
            'status': gap_info['status'],
            'is_frozen': gap_info['is_frozen']
        })
    
    df = pd.DataFrame(rows).sort_values('lu_class')
    return df


def _generate_restriction_impact_report(tracker) -> pd.DataFrame:
    """Generate restriction impact summary."""
    rows = []
    total_blocked = sum(v['area_blocked'] for v in tracker.restriction_impacts.values())
    
    for restriction_type, impact in tracker.restriction_impacts.items():
        pct_of_total = (impact['area_blocked'] / total_blocked * 100) if total_blocked > 0 else 0
        
        rows.append({
            'restriction_type': restriction_type,
            'area_blocked_ha': impact['area_blocked'],
            'transitions_blocked': impact['transitions_blocked'],
            'pct_of_total_gap': pct_of_total
        })
    
    df = pd.DataFrame(rows).sort_values('area_blocked_ha', ascending=False)
    return df


def _generate_frozen_classes_detail(tracker, areas_df, targets_df) -> pd.DataFrame:
    """Generate detailed analysis of frozen classes."""
    current_areas = areas_df.groupby('lu.from')['value'].sum()
    target_lookup = targets_df.set_index('lu.to')['value'].to_dict()
    
    rows = []
    for lu_class in tracker.frozen_classes:
        current = current_areas.get(lu_class, 0)
        target = target_lookup.get(lu_class, current)
        gap = target - current
        
        if abs(gap) > 0.1:  # Only include if there's a mismatch
            rows.append({
                'lu_class': lu_class,
                'lu_name': get_class_name(lu_class),
                'current_ha': current,
                'target_ha': target,
                'cannot_allocate_ha': gap,
                'reason': 'No ML predictions available'
            })
    
    # Handle empty case - create empty DataFrame with proper columns
    if not rows:
        return pd.DataFrame(columns=['lu_class', 'lu_name', 'current_ha', 'target_ha', 'cannot_allocate_ha', 'reason'])
    
    df = pd.DataFrame(rows).sort_values('cannot_allocate_ha', key=abs, ascending=False)
    return df

def _generate_recommendations(tracker, gap_df, frozen_df, metadata) -> str:
    """Generate actionable recommendations."""
    lines = []
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS FOR IMPROVING ALLOCATION FEASIBILITY")
    lines.append("=" * 80)
    lines.append("")
    
    # 1. Frozen classes recommendation
    if len(frozen_df) > 0:
        total_frozen_gap = frozen_df['cannot_allocate_ha'].abs().sum()
        frozen_classes = sorted(frozen_df['lu_class'].tolist())
        
        lines.append(f"1. ADD ML PREDICTIONS FOR FROZEN CLASSES")
        lines.append(f"   Current gap from frozen classes: {total_frozen_gap:,.1f} ha")
        lines.append(f"   Frozen classes needing predictions: {frozen_classes}")
        lines.append(f"   → Train suitability models for these classes")
        lines.append(f"   → Update MODELED_CLASSES in restriction generator")
        lines.append("")
    
    # 2. Target adjustment recommendation
    major_gaps = gap_df[gap_df['status'] == 'MAJOR_GAP']
    if len(major_gaps) > 0:
        lines.append(f"2. ADJUST INFEASIBLE TARGETS")
        for _, row in major_gaps.iterrows():
            lines.append(f"   Class {row['lu_class']} ({row['lu_name']}):")
            lines.append(f"     Target: {row['target_ha']:,.1f} ha")
            lines.append(f"     Achieved: {row['achieved_ha']:,.1f} ha")
            lines.append(f"     Gap: {row['gap_ha']:,.1f} ha")
            if row['is_frozen']:
                lines.append(f"     → Set target = {row['achieved_ha']:,.1f} ha (current value, class is frozen)")
            else:
                lines.append(f"     → Reduce target by {abs(row['gap_ha']):,.1f} ha or relax restrictions")
        lines.append("")
    
    # 3. Restriction level recommendation
    restriction_level = metadata.get('restriction_level', 'unknown')
    total_gap = gap_df['gap_ha'].abs().sum() / 2  # Divide by 2 (in/out)
    
    if total_gap > 1000:
        lines.append(f"3. CONSIDER MORE FLEXIBLE RESTRICTION LEVEL")
        lines.append(f"   Current level: {restriction_level}")
        lines.append(f"   Total allocation gap: {total_gap:,.1f} ha")
        if restriction_level in [1, 2, 3]:
            lines.append(f"   → Try Level {restriction_level + 1} for more flexibility")
            lines.append(f"   → Or adjust suitability threshold (currently applies at Level 4)")
        lines.append("")
    
    # 4. Natura 2000 impacts
    n2k_impact = tracker.restriction_impacts['natura2000']['area_blocked']
    if n2k_impact > 100:
        lines.append(f"4. NATURA 2000 PROTECTION IMPACTS")
        lines.append(f"   Area blocked by N2K rules: {n2k_impact:,.1f} ha")
        lines.append(f"   Protected pixels: {metadata.get('natura2000_pixels', 0):,}")
        lines.append(f"   → Review N2K transition rules if too restrictive")
        lines.append("")
    
    return "\n".join(lines)


def _generate_human_readable_report(tracker, gap_df, impact_df, frozen_df, metadata) -> str:
    """Generate comprehensive human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("ALLOCATION GAP ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generation Date: {datetime.now().isoformat()}")
    lines.append(f"Region: {metadata.get('region', 'Unknown')}")
    lines.append(f"Restriction Level: {metadata.get('restriction_level', 'Unknown')}")
    lines.append(f"Modeled Classes: {len(metadata.get('modeled_classes', []))}")
    lines.append(f"Frozen Classes: {len(tracker.frozen_classes)}")
    lines.append("=" * 80)
    lines.append("")
    
    # Target Achievement Summary
    lines.append("TARGET ACHIEVEMENT SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{'Class':<7} {'Name':<35} {'Target':>12} {'Achieved':>12} {'Gap':>12} {'%':>8}")
    lines.append("-" * 80)
    
    for _, row in gap_df.iterrows():
        status_marker = "✓" if row['status'] == 'FULL' else "⚠"
        lines.append(
            f"{row['lu_class']:<7} {row['lu_name'][:35]:<35} "
            f"{row['target_ha']:>12,.1f} {row['achieved_ha']:>12,.1f} "
            f"{row['gap_ha']:>12,.1f} {row['pct_achieved']:>7.1f}% {status_marker}"
        )
    
    total_gap = gap_df['gap_ha'].abs().sum() / 2
    lines.append("-" * 80)
    lines.append(f"Total Allocation Gap: {total_gap:,.1f} ha")
    lines.append("")
    
    # Restriction Impact Summary
    lines.append("RESTRICTION IMPACT SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{'Restriction Type':<30} {'Area Blocked (ha)':>20} {'% of Gap':>12}")
    lines.append("-" * 80)
    
    for _, row in impact_df.iterrows():
        if row['area_blocked_ha'] > 0:
            lines.append(
                f"{row['restriction_type']:<30} "
                f"{row['area_blocked_ha']:>20,.1f} "
                f"{row['pct_of_total_gap']:>11.1f}%"
            )
    lines.append("")
    
    # Frozen Classes Detail
    if len(frozen_df) > 0:
        lines.append("FROZEN CLASSES (NO ML PREDICTIONS)")
        lines.append("-" * 80)
        lines.append(f"{'Class':<7} {'Name':<35} {'Current':>12} {'Target':>12} {'Gap':>12}")
        lines.append("-" * 80)
        
        for _, row in frozen_df.iterrows():
            lines.append(
                f"{row['lu_class']:<7} {row['lu_name'][:35]:<35} "
                f"{row['current_ha']:>12,.1f} {row['target_ha']:>12,.1f} "
                f"{row['cannot_allocate_ha']:>12,.1f}"
            )
        
        lines.append("-" * 80)
        lines.append(f"Total frozen class gap: {frozen_df['cannot_allocate_ha'].abs().sum():,.1f} ha")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def _generate_excel_workbook(
    excel_path: Path,
    gap_df: pd.DataFrame,
    impact_df: pd.DataFrame,
    frozen_df: pd.DataFrame,
    blocked_df: Optional[pd.DataFrame],
    recommendations: str,
    report_text: str,
    tracker,
    metadata: Dict[str, Any]
):
    """Generate comprehensive Excel workbook with multiple sheets."""
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary
        summary_data = {
            'Metric': [
                'Generation Date',
                'Region',
                'Restriction Level',
                'Modeled Classes',
                'Frozen Classes',
                'Total Allocation Gap (ha)',
                'Total Blocked Transitions',
                'Total Blocked Area (ha)',
            ],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metadata.get('region', 'Unknown'),
                metadata.get('restriction_level', 'Unknown'),
                len(tracker.modeled_classes),
                len(tracker.frozen_classes),
                f"{sum(abs(g['gap']) for g in tracker.target_gaps.values()) / 2:,.1f}",
                f"{len(tracker.blocked_transitions):,}",
                f"{sum(v['area_blocked'] for v in tracker.restriction_impacts.values()):,.1f}",
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Gap By Class
        gap_df.to_excel(writer, sheet_name='Gap_By_Class', index=False)
        
        # Sheet 3: Restriction Impact
        impact_df.to_excel(writer, sheet_name='Restriction_Impact', index=False)
        
        # Sheet 4: Frozen Classes
        if len(frozen_df) > 0:
            frozen_df.to_excel(writer, sheet_name='Frozen_Classes', index=False)
        
        # Sheet 5: Blocked Transitions (sample if too large)
        if blocked_df is not None and len(blocked_df) > 0:
            # Sample if too large for Excel
            if len(blocked_df) > 10000:
                sample_df = blocked_df.sample(n=10000, random_state=42)
                sample_df.to_excel(writer, sheet_name='Blocked_Transitions', index=False)
                logger.info(f"Sampled 10,000 of {len(blocked_df):,} blocked transitions for Excel")
            else:
                blocked_df.to_excel(writer, sheet_name='Blocked_Transitions', index=False)
        
        # Sheet 6: Recommendations
        rec_lines = recommendations.split('\n')
        rec_df = pd.DataFrame({'Recommendations': rec_lines})
        rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        # Sheet 7: Metadata
        meta_data = {
            'Parameter': list(metadata.keys()),
            'Value': [str(v) for v in metadata.values()]
        }
        meta_df = pd.DataFrame(meta_data)
        meta_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    logger.info(f"Excel workbook created: {excel_path}")
