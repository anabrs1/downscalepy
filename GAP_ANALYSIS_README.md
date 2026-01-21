# Gap Analysis Feature

## Overview

The Gap Analysis feature provides comprehensive insights into **why allocation targets are not being met** by analyzing restriction impacts and identifying frozen classes without ML predictions.

## What Gets Generated

After running the solver, you'll automatically get these new reports:

### 1. Excel Workbook (`gap_analysis_report_level3.xlsx`)

**Comprehensive Excel file with 7 sheets:**

- **Summary**: Executive overview with key metrics
- **Gap_By_Class**: Detailed gap analysis for each LUM class
- **Restriction_Impact**: Area blocked by restriction type
- **Frozen_Classes**: Classes without ML predictions
- **Blocked_Transitions**: Sample of blocked allocation attempts
- **Recommendations**: Actionable suggestions
- **Metadata**: Run configuration and parameters

### 2. Text Reports

- **`gap_analysis_report.txt`**: Human-readable summary report
- **`recommendations.txt`**: Specific actionable recommendations
- **`blocked_transitions.parquet`**: Full database of blocked transitions (for large datasets)

## Key Insights Provided

### 1. Target vs Achieved Analysis

For each LUM class:
- Target area (ha)
- Achieved area (ha)
- Gap (ha)
- % Achieved
- Status (FULL, PARTIAL, or MAJOR_GAP)
- Whether class is frozen

### 2. Restriction Impact Breakdown

Quantifies area blocked by each restriction type:
- **Frozen Classes**: No ML predictions available
- **Natura 2000**: Protected area restrictions
- **Ecological Rules**: Land use transition constraints
- **Suitability Threshold**: Low suitability scores
- **Locked Pixels**: Permanently locked areas

### 3. Frozen Classes Identification

Lists all classes that lack ML predictions and cannot be allocated, with:
- Current area
- Target area
- Gap that cannot be filled
- Clear identification for model training needs

### 4. Actionable Recommendations

Automatically generates specific recommendations:

**Example recommendations:**
```
1. ADD ML PREDICTIONS FOR FROZEN CLASSES
   Frozen classes needing predictions: [2, 12, 13, 14, ...]
   → Train suitability models for these classes
   → Update MODELED_CLASSES in restriction generator

2. ADJUST INFEASIBLE TARGETS
   Class 24 (Silvo-pastoral agroforestry):
     Target: 883,255.1 ha
     Gap: 883,255.1 ha
     → Reduce target or relax restrictions

3. CONSIDER MORE FLEXIBLE RESTRICTION LEVEL
   Current level: 3
   → Try Level 4 for more flexibility
```

## How to Use

### Basic Usage

Gap analysis runs automatically with the solver:

```bash
python -m scripts.bias_correction_solver.main --scenario scenarios/run_2030
```

Output files appear in: `scripts/bias_correction_solver/output/`

### Interpreting Results

#### 1. Open the Excel File

The Excel workbook provides the most comprehensive view:

```
gap_analysis_report_level3.xlsx
├── Summary (high-level metrics)
├── Gap_By_Class (detailed class analysis)
├── Restriction_Impact (why allocations failed)
├── Frozen_Classes (classes needing ML predictions)
├── Blocked_Transitions (sample failures)
├── Recommendations (what to do next)
└── Metadata (run configuration)
```

#### 2. Check Frozen Classes

**Sheet**: `Frozen_Classes`

Identifies classes that **cannot be allocated** because they lack ML predictions:

| lu_class | lu_name | current_ha | target_ha | cannot_allocate_ha |
|----------|---------|------------|-----------|-------------------|
| 2 | Close-to-nature management | 0.0 | 487.1 | 487.1 |
| 12 | Intensive heterogeneous cropland | 0.0 | 498.8 | 498.8 |

**Action**: Train suitability models for these classes

#### 3. Review Gap By Class

**Sheet**: `Gap_By_Class`

Shows achievement status for all classes:

| lu_class | lu_name | target_ha | achieved_ha | gap_ha | pct_achieved | status |
|----------|---------|-----------|-------------|--------|--------------|--------|
| 24 | Silvo-pastoral agroforestry | 883,255.1 | 0.0 | 883,255.1 | 0.0% | MAJOR_GAP |
| 3 | Combined objective forestry | 62,395.8 | 58,123.4 | 4,272.4 | 93.2% | PARTIAL |

#### 4. Analyze Restriction Impacts

**Sheet**: `Restriction_Impact`

Quantifies **why** allocations were blocked:

| restriction_type | area_blocked_ha | transitions_blocked | pct_of_total_gap |
|------------------|-----------------|-------------------|------------------|
| frozen_classes | 45,234.2 | 125,432 | 68.5% |
| natura2000 | 12,456.8 | 34,221 | 18.9% |
| ecological_rules | 8,345.1 | 21,098 | 12.6% |

#### 5. Follow Recommendations

**Sheet**: `Recommendations`

Provides specific actions based on analysis:

1. **For Frozen Classes**: Train ML models and update `MODELED_CLASSES`
2. **For Major Gaps**: Reduce targets or relax restrictions
3. **For Restriction Level**: Consider more flexible level

## Understanding Restriction Types

### Frozen Classes
- **Cause**: No ML predictions available (not in `modeled_classes`)
- **Impact**: Cannot allocate to/from these classes
- **Solution**: Train suitability models for these classes

### Natura 2000
- **Cause**: Protected area rules
- **Impact**: Specific transitions forbidden in N2K areas
- **Solution**: Review N2K rules if too restrictive

### Ecological Rules
- **Cause**: Ecologically infeasible transitions
- **Impact**: Prevents unrealistic land use changes
- **Solution**: Generally should not be relaxed

### Suitability Threshold
- **Cause**: Low ML prediction confidence
- **Impact**: Blocks low-probability allocations
- **Solution**: Adjust threshold or improve models

### Locked Pixels
- **Cause**: Permanently fixed land use
- **Impact**: These pixels cannot change
- **Solution**: Ensure targets account for locked areas

## Workflow Integration

### Standard Workflow

```
1. Run solver with gap analysis
2. Open Excel file to review results
3. Identify main causes of gaps (frozen classes, restrictions, etc.)
4. Take action:
   a. Train models for frozen classes
   b. Adjust infeasible targets
   c. Consider different restriction level
5. Re-run solver
6. Compare gap reports
```

### Iterative Refinement

```
First Run → Gap = 1.3M ha
↓ (Train models for frozen classes)
Second Run → Gap = 850K ha
↓ (Adjust infeasible targets)
Third Run → Gap = 120K ha
↓ (Try restriction level 4)
Final Run → Gap = 15K ha ✓
```

## Technical Details

### Data Sources

Gap analysis uses:
- **Areas**: Current land use distribution
- **Targets**: Desired land use targets
- **Restrictions**: Transition rules and metadata
- **Allocations**: Solver results

### Metrics Calculated

- **Total Allocation Gap**: `Σ|target - achieved| / 2` (hectares)
- **% Achieved**: `(achieved / target) × 100`
- **Area Blocked**: Sum of all blocked transition areas by type
- **Frozen Class Impact**: Area that cannot be allocated due to missing models

### Status Codes

- **FULL**: Gap < 1 ha (target met)
- **PARTIAL**: 90% < achieved < 100% (close to target)
- **MAJOR_GAP**: achieved < 90% (significant gap)

## Example Output Files

### Gap Summary Extract
```
Class 24 (Silvo-pastoral agroforestry):
  Target: 883,255.1 ha
  Achieved: 0.0 ha
  Gap: 883,255.1 ha
  Status: MAJOR_GAP
  Is Frozen: False
```

### Restriction Impact Extract
```
Frozen Classes: 45,234.2 ha blocked (68.5% of gap)
Natura 2000: 12,456.8 ha blocked (18.9% of gap)
Ecological Rules: 8,345.1 ha blocked (12.6% of gap)
```

## FAQ

**Q: Why are some classes showing 0% achieved?**
A: Either the class is frozen (no ML predictions) or restrictions are too strict.

**Q: What does "frozen class" mean?**
A: A class without ML suitability predictions, so it cannot be allocated.

**Q: How do I fix frozen classes?**
A: Train Random Forest models for those classes and update `MODELED_CLASSES`.

**Q: Should I always use restriction level 4?**
A: No - start with level 3. Only increase if you have significant gaps and understand the trade-offs.

**Q: Can I ignore frozen class gaps?**
A: If the target matches current area for frozen classes, yes. Otherwise, you need ML predictions.

## Version History

### v1.0.0 (2025-12-01)
- Initial release of gap analysis feature
- Excel output with 7 comprehensive sheets
- Automatic frozen class identification
- Restriction impact quantification
- Actionable recommendations generator

## See Also

- Main README: `scripts/bias_correction_solver/README.md`
- Solver Documentation: Core algorithm details
- MCP Integration: External tool connections
