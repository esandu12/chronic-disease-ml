# EC5203 Group 07 — Data Quality Report

## 1. Dataset Overview

- Total records: 280,985
- Total columns: 39
- Target: sublabel
- Classes: 8

## 2. Missing Values

- Total pandas-detected missing cells: 0

No native null values were detected.
Placeholder values remain a separate concern.

## 3. Exact Duplicate Rows

- Duplicate rows: 2,284
- Duplicate percentage: 0.81%

Recommendation:
Investigate duplicates before deciding whether
they represent redundant or legitimate observations.

## 4. Composite Key

- Unique composite keys: 192
- Repeated keys: 280,793

The composite key is not a unique patient identifier.

## 5. Class Distribution

sublabel
N           106884
HY           69806
DI_HY        66473
DI           31918
HT            2595
DI_HT         1398
HT_HY         1066
DI_HT_HY       845

Largest class: N
Smallest class: DI_HT_HY

The dataset contains substantial class imbalance.

## 6. Suspicious Numerical Indicators

- hypertension values outside 0/1: 166,597
- heart_disease values outside 0/1: 155,406

These are observed coding inconsistencies relative to
a binary interpretation. Their exact meanings require
verification against the source documentation.

Do not automatically convert these values to binary.

## 7. Target Leakage

The cross-tabulation shows that:

- label is determined by sublabel.
- disease_flags maps directly to disease combinations.

These must not be used as predictors.

Additional review is required for:
- diabetes
- hypertension
- heart_disease
- composite_key
- source_dataset

## 8. Source Dataset Effects

The source-class chart shows different class
distributions across the original source datasets.

Investigate whether models could identify the
source instead of learning disease patterns.

## 9. Visualization Findings

- Strong disease-class imbalance is visible.
- Clinical measurements show varying distributions.
- Some measurements have sharp distribution peaks.
- Potential outliers appear in several boxplots.
- Numerical correlations are present.
- BMI distributions overlap between disease classes.

The boxplots use a 20,000-record random sample.
Rare classes may be underrepresented in that sample.

## 10. Recommended Cleaning Decisions

For team review:

1. Verify publisher definitions and clinical units.
2. Investigate exact duplicate rows.
3. Check placeholder and unrealistic values.
4. Investigate fractional disease indicators.
5. Exclude confirmed target-derived predictors.
6. Investigate source-dataset artifacts.
7. Review identifiers before choosing a data split.
8. Fit preprocessing only on training data.

## 11. Member 1 Responsibility

This report documents observations and recommendations.

No cleaning, encoding, scaling, feature engineering,
or model training was performed.
