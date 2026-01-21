"""
Allocation tracking for bias correction solver.

Tracks allocation attempts, successes, failures, and restriction impacts
to understand why targets are not being met.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AllocationTracker:
    """
    Track allocation attempts, successes, and failures during optimization.
    
    This class records:
    - Target gaps (target vs achieved allocations)
    - Blocked transitions with detailed reasons
    - Restriction impacts by type
    - Frozen vs modeled classes
    
    Attributes
    ----------
    target_gaps : dict
        Dictionary mapping lu_class to gap information
    blocked_transitions : list
        List of all blocked transition attempts
    restriction_impacts : dict
        Summary of area blocked by restriction type
    modeled_classes : set
        Set of classes with ML predictions
    frozen_classes : set
        Set of classes without ML predictions
    """
    
    def __init__(self):
        """Initialize the allocation tracker."""
        self.target_gaps = {}  # {lu_class: {'target': X, 'achieved': Y, 'gap': Z}}
        self.blocked_transitions = []  # List of blocked transition attempts
        
        # Track impacts by restriction type
        self.restriction_impacts = {
            'frozen_classes': {
                'area_blocked': 0.0,
                'transitions_blocked': 0,
                'details': []
            },
            'natura2000': {
                'area_blocked': 0.0,
                'transitions_blocked': 0,
                'details': []
            },
            'ecological_rules': {
                'area_blocked': 0.0,
                'transitions_blocked': 0,
                'details': []
            },
            'suitability_threshold': {
                'area_blocked': 0.0,
                'transitions_blocked': 0,
                'details': []
            },
            'locked_pixels': {
                'area_blocked': 0.0,
                'transitions_blocked': 0,
                'details': []
            }
        }
        
        self.modeled_classes = set()
        self.frozen_classes = set()
        self.restrictions_metadata = {}
    
    def initialize_from_metadata(self, restrictions_metadata: Dict[str, Any]):
        """
        Initialize tracker with restrictions metadata.
        
        Parameters
        ----------
        restrictions_metadata : dict
            Metadata from restrictions JSON file
        """
        self.restrictions_metadata = restrictions_metadata
        
        # Extract modeled classes
        modeled = restrictions_metadata.get('modeled_classes', [])
        self.modeled_classes = set(modeled)
        
        # Calculate frozen classes (all classes not in modeled_classes)
        all_classes = set(range(1, 34))  # LUM classes 1-33
        self.frozen_classes = all_classes - self.modeled_classes
        
        logger.info(f"Tracker initialized: {len(self.modeled_classes)} modeled classes, "
                   f"{len(self.frozen_classes)} frozen classes")
        logger.debug(f"Modeled classes: {sorted(self.modeled_classes)}")
        logger.debug(f"Frozen classes: {sorted(self.frozen_classes)}")
    
    def record_blocked_transition(
        self,
        pixel_id: str,
        from_class: int,
        to_class: int,
        area_ha: float,
        restriction_type: str,
        reason: str,
        suitability: Optional[float] = None,
        in_natura2000: bool = False
    ):
        """
        Record a transition blocked by restrictions.
        
        Parameters
        ----------
        pixel_id : str
            Pixel identifier
        from_class : int
            Source land use class
        to_class : int
            Target land use class
        area_ha : float
            Area that would have been allocated (hectares)
        restriction_type : str
            Type of restriction: 'frozen_classes', 'natura2000', 
            'ecological_rules', 'suitability_threshold', 'locked_pixels'
        reason : str
            Human-readable reason for block
        suitability : float, optional
            Suitability score if applicable
        in_natura2000 : bool, default=False
            Whether pixel is in Natura 2000 area
        """
        # Record individual transition
        self.blocked_transitions.append({
            'pixel_id': pixel_id,
            'lu_from': from_class,
            'lu_to': to_class,
            'area_ha': area_ha,
            'restriction_type': restriction_type,
            'reason': reason,
            'suitability': suitability,
            'in_natura2000': in_natura2000
        })
        
        # Update impact summary
        if restriction_type in self.restriction_impacts:
            self.restriction_impacts[restriction_type]['area_blocked'] += area_ha
            self.restriction_impacts[restriction_type]['transitions_blocked'] += 1
        else:
            logger.warning(f"Unknown restriction type: {restriction_type}")
    
    def classify_restriction_type(
        self,
        from_class: int,
        to_class: int,
        reason: str,
        in_natura2000: bool = False,
        suitability: Optional[float] = None
    ) -> str:
        """
        Classify restriction type based on transition characteristics.
        
        Parameters
        ----------
        from_class : int
            Source land use class
        to_class : int
            Target land use class
        reason : str
            Reason text from restrictions file
        in_natura2000 : bool
            Whether pixel is in Natura 2000
        suitability : float, optional
            Suitability score
        
        Returns
        -------
        str
            Restriction type category
        """
        # Check frozen classes first (highest priority)
        if from_class in self.frozen_classes or to_class in self.frozen_classes:
            return 'frozen_classes'
        
        # Check Natura 2000
        if in_natura2000 and 'Natura' in reason:
            return 'natura2000'
        
        # Check suitability threshold
        if 'Suitability' in reason or 'suitability' in reason.lower():
            return 'suitability_threshold'
        
        # Check ecological rules
        if 'Ecological' in reason or 'ecological' in reason.lower():
            return 'ecological_rules'
        
        # Check locked pixels
        if 'Locked' in reason or 'locked' in reason.lower():
            return 'locked_pixels'
        
        # Default to ecological rules
        return 'ecological_rules'
    
    def compute_target_gaps(
        self,
        targets_df,
        achieved_areas: Dict[int, float]
    ):
        """
        Compare targets vs achieved allocations.
        
        Parameters
        ----------
        targets_df : pd.DataFrame
            Targets dataframe with lu.to and value columns
        achieved_areas : dict
            Dictionary mapping lu_class to achieved area
        """
        logger.info("Computing target gaps...")
        
        # Aggregate targets by lu.to to sum all transitions to same destination
        # This ensures correct target values when multiple classes transition to same lu.to
        targets_agg = targets_df.groupby('lu.to', as_index=False)['value'].sum()
        
        for _, row in targets_agg.iterrows():
            lu_class = row['lu.to']
            target = row['value']
            achieved = achieved_areas.get(lu_class, 0.0)
            gap = target - achieved
            pct_achieved = (achieved / target * 100) if target > 0 else 100.0
            
            # Determine status
            if abs(gap) < 1.0:
                status = 'FULL'
            elif pct_achieved > 90:
                status = 'PARTIAL'
            else:
                status = 'MAJOR_GAP'
            
            self.target_gaps[lu_class] = {
                'target': target,
                'achieved': achieved,
                'gap': gap,
                'pct_achieved': pct_achieved,
                'status': status,
                'is_frozen': lu_class in self.frozen_classes
            }
        
        # Log summary
        total_gap = sum(abs(g['gap']) for g in self.target_gaps.values()) / 2
        n_major_gaps = sum(1 for g in self.target_gaps.values() if g['status'] == 'MAJOR_GAP')
        
        logger.info(f"Target gap analysis complete:")
        logger.info(f"  Total allocation gap: {total_gap:,.1f} ha")
        logger.info(f"  Classes with major gaps: {n_major_gaps}")
        logger.info(f"  Total blocked transitions: {len(self.blocked_transitions):,}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracker statistics.
        
        Returns
        -------
        dict
            Summary statistics
        """
        total_blocked_area = sum(
            impact['area_blocked'] 
            for impact in self.restriction_impacts.values()
        )
        
        total_blocked_transitions = sum(
            impact['transitions_blocked']
            for impact in self.restriction_impacts.values()
        )
        
        total_gap = sum(abs(g['gap']) for g in self.target_gaps.values()) / 2
        
        return {
            'total_blocked_area_ha': total_blocked_area,
            'total_blocked_transitions': total_blocked_transitions,
            'total_allocation_gap_ha': total_gap,
            'n_modeled_classes': len(self.modeled_classes),
            'n_frozen_classes': len(self.frozen_classes),
            'restriction_impacts': {
                name: {
                    'area_blocked_ha': impact['area_blocked'],
                    'transitions_blocked': impact['transitions_blocked'],
                    'pct_of_total': (impact['area_blocked'] / total_blocked_area * 100) 
                                   if total_blocked_area > 0 else 0
                }
                for name, impact in self.restriction_impacts.items()
            }
        }
