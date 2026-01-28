# ECG QA Schema Specification (v1)

## Output Keys (12 total)

| Key | Type | Source | Notes |
|-----|------|--------|-------|
| `ecg_abnormal` | bool | Computed | True if any abnormality detected (anti-hallucination anchor) |
| `shd` | bool | MHI only | Structural heart disease from echocardiography |
| `rhythm` | string | Both | e.g., "Sinusal", "Afib", "Bradycardia" |
| `rate_bpm` | int | MHI only | Heart rate (excluded from MIMIC interpretation) |
| `lvef` | int + unit:"%" | MHI only | Left ventricular ejection fraction |
| `acs` | bool | MHI only | Acute coronary syndrome |
| `afib_risk` | string | MHI only | "high_risk"/"moderate_risk"/"low_risk" (2y/5y AFib prediction) |
| `conduction` | labels[] | Both | Multi-label conduction abnormalities |
| `chamber_enlargement` | labels[] | Both | Multi-label chamber enlargement |
| `ischemia` | labels[] | Both | Multi-label ischemia/infarct findings |
| `pericarditis` | bool | Both | Derived from pericarditis labels |
| `other` | labels[] | Both | Multi-label other findings |
| `findings` | labels[] | Eval only | **Never trained** - backward-compat meta-category |

## Task to Prompt Category Mapping

```
interpretation/json_interpretation/interpretation_complex
    → [ecg_abnormal, rhythm, conduction, chamber_enlargement, ischemia, pericarditis, other]

classification
    → [ecg_abnormal]

structural_heart_disease
    → [shd]

lvef
    → [lvef]

category_rhythm / rhythm
    → [rhythm] (+ rate_bpm if prompt mentions rate/bpm/heart rate)

category_conduction / conduction
    → [conduction]

category_infarct_ischemia / localization_*
    → [ischemia]

category_chamber_enlargement
    → [chamber_enlargement]

category_pericarditis
    → [pericarditis]

category_other
    → [other]

acs_severity / culprit_artery / urgency_assessment
    → [acs]

afib_risk
    → [afib_risk]

ecg_interval
    → [rate_bpm]

localization_qrs_axis
    → [conduction]
```

## Policy A: No Predicted + Null

**Rule**: `status: "predicted"` is only set if the ground truth value exists.

- **Value tasks** (ecg_abnormal, shd, rhythm, rate_bpm, lvef, acs, afib_risk, pericarditis):
  - Must have non-null value to be predicted
- **Label tasks** (conduction, chamber_enlargement, ischemia, other):
  - Empty `[]` is a valid prediction (no findings detected)
  - Always predicted when requested

## Supervision Rules

**Only supervise predicted tasks** - never supervise `not_requested` tasks.

### Example: SHD Prompt
```python
task: "shd"
supervised_paths: ["outputs.shd.status", "outputs.shd.value"]
# ecg_abnormal is NOT supervised
```

### Example: Interpretation Prompt
```python
tasks_requested: ["ecg_abnormal", "rhythm", "conduction", "chamber_enlargement", "ischemia", "pericarditis", "other"]
supervised_paths: [
    "outputs.ecg_abnormal.status", "outputs.ecg_abnormal.value",
    "outputs.rhythm.status", "outputs.rhythm.value",
    "outputs.conduction.status", "outputs.conduction.labels_present", "outputs.conduction.labels",
    "outputs.chamber_enlargement.status", "outputs.chamber_enlargement.labels_present", "outputs.chamber_enlargement.labels",
    "outputs.ischemia.status", "outputs.ischemia.labels_present", "outputs.ischemia.labels",
    "outputs.pericarditis.status", "outputs.pericarditis.value",
    "outputs.other.status", "outputs.other.labels_present", "outputs.other.labels"
]
```

## Output Parquet Fields

| Field | Type | Description |
|-------|------|-------------|
| `ecg_id` | string | Unique ECG identifier |
| `waveform_path_psa` | string | Path to waveform file |
| `prompt` | string | The question/instruction text |
| `prompt_category` | string | Category for task routing |
| `sample_weight` | float | Training weight (default 1.0) |
| `gt_json_full` | string (JSON) | Complete ground truth with all tasks |
| `target_json` | string (JSON) | Only predicted tasks have values |
| `supervised_paths` | list[string] | Dot-paths for masked loss |
| `available_tasks` | list[string] | Tasks with GT available for this ECG |

## Schema Invariants (Validated)

1. **No predicted+null**: If `status == "predicted"`, then `value` must not be null (for value tasks)
2. **lvef.unit always "%"**: Never null, even when not requested
3. **labels_present consistency**: Must match `len(labels) > 0`
4. **findings never predicted**: Always `status: "not_requested"`
5. **tasks_requested/task accuracy**: Must exactly match tasks with `status: "predicted"`

## ecg_abnormal Computation

Returns `true` if ANY of:
- Rhythm is not "Sinusal" or "Regular"
- Any conduction abnormality present
- Any chamber enlargement present
- Any ischemia/infarct present
- Pericarditis present
- Any other abnormality present

Otherwise returns `false`.

## afib_risk Values

| Value | Meaning |
|-------|---------|
| `"high_risk"` | Will develop AFib within 2 years (`afib_label_2y = true`) |
| `"moderate_risk"` | Will develop AFib within 5 years (`afib_label_5y = true`, `afib_label_2y = false`) |
| `"low_risk"` | Both labels false |
| `null` | Data not available (not MHI) |

## Usage

```bash
# Test mode (shows 10 MHI + 10 MIMIC samples)
python dataset_generation/convert_ecg_qa_schema.py --test_mode

# Convert single file
python dataset_generation/convert_ecg_qa_schema.py \
    --input /path/to/input.parquet \
    --output /path/to/output.parquet

# Convert train/val/test splits
python dataset_generation/convert_ecg_qa_schema.py \
    --train /path/to/train.parquet \
    --test /path/to/test.parquet \
    --out_dir /path/to/output/
```
