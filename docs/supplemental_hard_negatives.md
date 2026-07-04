# Supplementary Methods: Hard Negative Mining for ECG-Text Matching

## Overview

During Stage 2 vision-language alignment, the ECG-text matching (ETM) objective employs hard negative mining to force the Q-Former bridge to discriminate between clinically confusable ECG-text pairs. Hard negatives are selected through two complementary mechanisms: (1) pre-defined clinically targeted families that pair conditions known to produce similar waveform morphologies, and (2) dynamic in-batch similarity-based mining that identifies the most confusable negative for each ECG at every training step.

## Clinically Targeted Hard Negative Families

We define 10 exclusion/confusion groups based on clinical knowledge of ECG waveform similarity. Within each group, conditions share morphological features that make them prone to misclassification. Five of these groups define mutually exclusive labels (only one member can be true per ECG), while the remaining five define targeted hard negative pairs where specific anchor conditions are paired with their most confusable alternatives.

### Mutually Exclusive Groups

These groups enforce label consistency — members cannot co-occur on the same ECG, so any non-matching member serves as a valid negative.

**Table S1. Mutually exclusive ECG condition groups.**

| Group | Members | Clinical Rationale |
|---|---|---|
| **Primary Rhythm** | Sinus rhythm, Atrial fibrillation, Atrial flutter, Atrial tachycardia, Ectopic atrial rhythm, Junctional rhythm, SVT, Ventricular rhythm, Ventricular tachycardia | An ECG has exactly one dominant rhythm mechanism |
| **Rhythm Regularity** | Regular, Regularly irregular, Irregularly irregular | Rhythm regularity is a single categorical descriptor |
| **QRS Axis** | Left axis deviation, Right axis deviation, Right superior axis | The frontal QRS axis occupies one quadrant |
| **Bundle Branch Block** | Left bundle branch block, Right bundle branch block | LBBB and RBBB produce opposite QRS morphologies in the same leads |
| **AV Block Degree** | 1st degree, 2nd degree Mobitz I, 2nd degree Mobitz II, 3rd degree AV block | AV block is graded on a single severity scale |

### Targeted Hard Negative Pairs

For conditions that are not mutually exclusive but produce similar waveform morphologies, we define explicit anchor-to-negative mappings. During ETM training, when an anchor condition's text is the positive match, its targeted negatives are preferentially sampled as hard negatives.

**Table S2. Targeted hard negative pairs for ECG-text matching.**

| Family | Anchor Condition | Hard Negative Conditions | Shared Morphological Feature |
|---|---|---|---|
| **Wide QRS Complex** | LV pacing | Ventricular paced, Ventricular rhythm, V-tach, LBBB | All produce wide QRS (>120ms) with similar LBBB-like morphology; distinguishing LV pacing from RV pacing or intrinsic wide-complex rhythms requires subtle lead-specific criteria |
| | Ventricular paced | LV pacing, Ventricular rhythm, V-tach, LBBB | Pacemaker spikes may be absent in digital ECGs; paced QRS mimics LBBB pattern |
| | Ventricular rhythm | AFib, Junctional rhythm, Ventricular paced, LV pacing, V-tach | Accelerated idioventricular rhythm vs slow VT vs paced rhythm — all wide and regular |
| | Ventricular tachycardia | Ventricular rhythm, AFib, SVT, Ventricular paced | Wide-complex tachycardia differential: VT vs SVT with aberrancy vs pre-excited AFib |
| **ST Segment Polarity** | ST elevation (septal V1-V2) | ST depression (septal V1-V2) | Same leads, opposite deflection; elevation suggests STEMI, depression suggests ischemia or reciprocal change |
| | ST elevation (anterior V3-V4) | ST depression (anterior V3-V4) | Anterior ST changes — elevation vs depression separated by <1mm in early presentations |
| | ST elevation (inferior II,III,aVF) | ST depression (inferior II,III,aVF) | Inferior territory; reciprocal changes in acute MI can cause confusion with primary ST depression |
| | ST elevation (lateral I,aVL,V5-V6) | ST depression (lateral I,aVL,V5-V6) | Lateral wall — subtle ST segment deviations easily confused |
| | ST elevation (posterior V7-V9) | ST depression (septal V1-V2), ST depression (anterior V3-V4) | Posterior STEMI manifests as anterior/septal ST depression — the "mirror image" pattern |
| **AV Block Severity** | Third Degree AV Block | 1st degree AV block, Mobitz I, Mobitz II, Junctional rhythm | Progressive AV conduction disease; high-degree block can mimic complete block if escape rate is close to atrial rate |
| **Repolarization Mimics** | U wave | Prolonged QT, Low voltage, T-wave inversion (lateral), T-wave inversion (anterior) | U waves are low-amplitude deflections easily confused with prolonged T waves, bifid T waves, or noise in low-voltage recordings |
| **Artifact Mimics** | Lead misplacement | LV pacing, Ventricular paced, Ventricular rhythm | Limb lead reversal or precordial misplacement can simulate wide-complex rhythms, paced morphology, or axis deviation |

### Yes/No Counterpart Negatives

For each binary diagnostic question (e.g., "Is there atrial fibrillation?"), the system automatically generates a counterpart negative by flipping the answer polarity. If the positive text is "Is there atrial fibrillation? Yes — irregularly irregular rhythm with absent P waves," the hard negative is the corresponding "No" answer for the same question. This forces the model to attend to the ECG signal rather than learning text-only patterns.

## Dynamic In-Batch Hard Negative Mining

In addition to pre-defined families, ETM employs dynamic in-batch mining at each training step:

1. **Similarity computation**: For each mini-batch of B ECG-text pairs, compute the B x B cosine similarity matrix between ECG embeddings and all text embeddings using the current contrastive representations from the ETC objective.

2. **Masking**: Set similarity to -infinity for:
   - Self-pairs (diagonal)
   - Same-label pairs (ECGs sharing the same text label)
   - Same-ECG pairs (multiple texts from the same waveform)

3. **Top-k selection**: For each ECG, select the k=1 text with highest remaining similarity — this is the hardest in-batch negative, i.e., the text that the current model most confidently (but incorrectly) believes matches this ECG.

4. **Binary classification**: The bridge must classify paired ECG-text inputs as matched (positive) or mismatched (hard negative), trained with binary cross-entropy loss.

5. **Warmup**: ETM loss weight is linearly warmed from 0 to 1.0 over the first 2,000 training steps to allow the contrastive representations to stabilize before hard negatives become meaningful.

**Table S3. Hard negative mining hyperparameters.**

| Parameter | Value | Search Range |
|---|---|---|
| Hard negative k | 1 | {1, 2, 3} |
| ETM warmup steps | 2,000 | — |
| ETM loss weight | 1.0 | — |
| ETG delay steps | 2,000 | — |
| ETG warmup steps | 4,000 | — |
| ETG loss weight | 1.0 | {0.25, 0.5, 1.0} |
| Text bank size | 2,048 | — |
| Bank refresh interval | 256 steps | — |
| Bank negatives per ECG | 5 | — |

## Loss Combination

The total Stage 2 loss combines all three objectives with scheduled weighting:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{ETC}} \cdot \mathcal{L}_{\text{ETC}} + w_{\text{ETM}}(t) \cdot \mathcal{L}_{\text{ETM}} + w_{\text{ETG}}(t) \cdot \mathcal{L}_{\text{ETG}}$$

where $\lambda_{\text{ETC}} = 1.0$, $w_{\text{ETM}}(t)$ linearly ramps from 0 to 1.0 over steps 0-2,000, and $w_{\text{ETG}}(t)$ is zero until step 2,000 then linearly ramps to 1.0 by step 4,000. This staged schedule allows the contrastive representations to initialize before the matching and generation objectives engage.
