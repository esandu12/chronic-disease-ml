# EC5203 — Member 1 Handoff

## Completed

- Dataset acquisition and file verification
- Dataset dimensions and schema inspection
- Eight-class target identification
- Missing-value and duplicate audit
- Numerical and categorical quality inspection
- Target-leakage investigation
- Source-dataset comparison
- Statistical summary tables
- Six EDA visualizations
- Initial 39-column data dictionary
- Data-quality documentation

## Handoff to Member 2

Review the data-quality report before preprocessing.

Important issues:
- Exact duplicates
- Suspicious fractional indicators
- Potential target leakage
- Source-specific differences
- Verification of feature units and meanings

Member 2 owns the cleaning and preprocessing pipeline.

## Handoff to Members 3 and 4

- Target: sublabel
- Eight disease classes
- Significant class imbalance
- Confirmed target-derived columns: label, disease_flags
- Other disease indicators require leakage review
- Source and repeated-record structure require review

The modelling team must agree on the data split,
cross-validation and evaluation methodology.

## Outstanding

Verify the data dictionary against the publisher DOCX
and obtain team approval for cleaning recommendations.
