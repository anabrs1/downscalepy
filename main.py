#!/usr/bin/env python3
"""
Main entry point for Bias Correction Solver.

This script provides a complete pipeline for running the bias correction
solver with automatic path detection, data loading, solving, and output generation.

Usage:
    python -m bias_correction_solver.main [options]
    
Or directly:
    python scripts/bias_correction_solver/main.py [options]
"""

import argparse
import sys
import logging
from pathlib import Path

# Import solver components
from .data import detect_input_paths, load_all_inputs, validate_solver_inputs
from .core import BiasCorrectSolver
from .core.allocation_tracker import AllocationTracker
from .output import generate_2030_projection_map
from .reports import generate_gap_analysis_reports
from .utils import setup_logger, SolverConfig, PerformanceMonitor
import json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bias Correction Solver for Land Use Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with automatic path detection
  python -m bias_correction_solver.main
  
  # Specify scenario directory
  python -m bias_correction_solver.main --scenario scenarios/run_2030
  
  # Specify output directory and formats
  python -m bias_correction_solver.main --output ./results --formats parquet csv json
  
  # Use custom config file
  python -m bias_correction_solver.main --config my_config.yaml
  
  # Enable verbose logging
  python -m bias_correction_solver.main --verbose
        """
    )
    
    parser.add_argument(
        '--scenario', '-s',
        help='Scenario directory path (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (default: auto-detected as outputs/{REGION}/projection/{year})'
    )
    
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['csv', 'parquet', 'json', 'tif'],
        default=['parquet', 'csv', 'tif'],
        help='Output formats (default: parquet csv tif)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--target-year', '-y',
        type=int,
        default=2030,
        help='Target year for projection (default: 2030)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(
        level=log_level,
        log_file=args.log_file
    )
    
    logger.info("="*70)
    logger.info("BIAS CORRECTION SOLVER")
    logger.info("Version 1.0.0")
    logger.info("="*70)
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        # Step 1: Detect input paths
        logger.info("\n[1/6] Detecting input paths...")
        paths = detect_input_paths(
            base_dir=".",
            scenario_dir=args.scenario
        )
        
        # Extract region and construct output path
        region = paths.get('region')
        if not region:
            logger.warning("Region code not detected, using 'UNKNOWN' as default")
            region = 'UNKNOWN'
        
        # Construct output path if not specified
        if args.output is None:
            args.output = f"./outputs/{region}_100m/projection/{args.target_year}"
            logger.info(f"Auto-detected output directory: {args.output}")
        
        # Step 2: Load input data
        logger.info("\n[2/6] Loading input data...")
        areas, priors, targets, restrictions = load_all_inputs(paths)
        
        # Step 3: Validate inputs
        logger.info("\n[3/6] Validating inputs...")
        validation_results = validate_solver_inputs(
            areas, priors, targets, restrictions
        )
        
        # Check for validation errors
        has_errors = any(errors for errors in validation_results.values())
        if has_errors:
            logger.error("Validation failed! Please fix errors before proceeding.")
            return 1
        
        # Step 4: Load configuration
        logger.info("\n[4/6] Loading configuration...")
        if args.config:
            from .utils.config import load_config
            config = load_config(args.config)
            logger.info(f"Loaded config from: {args.config}")
        else:
            config = SolverConfig()
            logger.info("Using default configuration")
        
        config.verbose = args.verbose
        
        # Load restrictions metadata for gap analysis
        restrictions_metadata = {}
        if paths.get('restrictions_json'):
            logger.info("Loading restrictions metadata...")
            with open(paths['restrictions_json'], 'r') as f:
                restrictions_metadata = json.load(f)
        
        # Initialize allocation tracker
        tracker = AllocationTracker()
        if restrictions_metadata:
            tracker.initialize_from_metadata(restrictions_metadata)
        
        # Step 5: Run solver
        logger.info("\n[5/7] Running bias correction solver...")
        solver = BiasCorrectSolver(
            areas=areas,
            priors=priors,
            targets=targets,
            restrictions=restrictions,
            config=config.to_dict()
        )
        
        results = solver.solve(target_year=args.target_year)
        
        # Step 6: Generate projection maps
        logger.info("\n[6/7] Generating projection maps...")
        
        # Get reference raster for GeoTIFF generation
        reference_raster = paths.get('reference_raster')
        if not reference_raster and 'tif' in args.formats:
            # Try to find reference raster
            import glob
            raster_candidates = glob.glob(f'outputs/{region}_100m/lum_rasters/*_2018.tif')
            if raster_candidates:
                reference_raster = raster_candidates[0]
                logger.info(f"Auto-detected reference raster: {reference_raster}")
        
        # Update results with reference raster
        if reference_raster:
            from .output.projector import ProjectionGenerator
            projector = ProjectionGenerator(results, reference_raster=reference_raster)
            output_files = projector.generate_projection_map(
                output_dir=args.output,
                target_year=args.target_year,
                formats=args.formats
            )
        else:
            output_files = generate_2030_projection_map(
                results,
                output_dir=args.output,
                target_year=args.target_year,
                formats=args.formats
            )
        
        # Step 7: Generate gap analysis reports
        logger.info("\n[7/7] Generating gap analysis reports...")
        
        # Compute achieved areas from allocations
        allocations = results['allocations']
        achieved_areas = allocations.groupby('lu.to')['value'].sum().to_dict()
        
        # Compute target gaps
        tracker.compute_target_gaps(targets, achieved_areas)
        
        # Generate comprehensive reports
        restriction_level = restrictions_metadata.get('restriction_level', 3)
        gap_reports = generate_gap_analysis_reports(
            tracker=tracker,
            targets_df=targets,
            areas_df=areas,
            restrictions_metadata=restrictions_metadata,
            output_dir=Path(args.output),
            restriction_level=restriction_level
        )
        
        # Log gap analysis summary
        gap_summary = tracker.get_summary()
        logger.info(f"\nGap Analysis Summary:")
        logger.info(f"  Total allocation gap: {gap_summary['total_allocation_gap_ha']:,.1f} ha")
        logger.info(f"  Modeled classes: {gap_summary['n_modeled_classes']}")
        logger.info(f"  Frozen classes: {gap_summary['n_frozen_classes']}")
        logger.info(f"  Excel report: {gap_reports['excel_path']}")
        
        # Stop monitoring
        monitor.stop()
        
        # Print final summary
        logger.info("\n" + "="*70)
        logger.info("SOLVER COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Target Year: {args.target_year}")
        logger.info(f"Total Time: {monitor.metrics['elapsed_time']:.2f} seconds")
        logger.info(f"Peak Memory: {monitor.metrics['peak_memory_mb']:.2f} MB")
        logger.info(f"Output Directory: {args.output}")
        logger.info(f"Output Files: {len(output_files)}")
        logger.info("="*70)
        
        # Print performance summary
        if args.verbose:
            monitor.print_summary()
        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"SOLVER FAILED")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error(f"{'='*70}")
        
        monitor.stop()
        return 1


if __name__ == "__main__":
    sys.exit(main())
