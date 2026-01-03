# Methods: LLM-Based Stenosis Extraction from Catheterization Reports

## Study Population

We retrospectively analyzed 4,249 coronary catheterization studies from 3,935 patients at UCSF Medical Center (2013-2019), comprising 20,990 angiographic video sequences.

## Automated Stenosis Extraction

Coronary stenosis measurements were extracted from free-text catheterization reports using DeepSeek-V2, a mixture-of-experts large language model. The extraction pipeline was containerized using Docker to ensure reproducibility and deployed via the CathAI framework.

For each report, the model extracted stenosis percentages across 18 coronary segments: proximal, mid, and distal segments of the right coronary artery (RCA), left anterior descending (LAD), and left circumflex (LCx) arteries, along with the left main, diagonal branches (D1, D2), obtuse marginals (OM1, OM2), posterior descending artery (PDA), and posterolateral branch.

## Stenosis Mapping

Qualitative descriptions were mapped to standardized percentages using clinical conventions:
- "Normal/no disease": 0%
- "Mild disease/stenosis": 30%
- "Moderate stenosis": 50%
- "Severe stenosis": 70%
- "Critical/subtotal": 90%
- "Total occlusion/CTO": 100%

Binary significant stenosis was defined as >=70% luminal narrowing, consistent with clinical thresholds for revascularization consideration.

## Validation

Model predictions were evaluated against LLM-extracted ground truth labels using area under the receiver operating characteristic curve (AUC) for binary classification and mean absolute error (MAE) for regression. Confidence intervals were computed via bootstrap resampling (n=1,000 iterations).
