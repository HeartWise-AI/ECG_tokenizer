from services.verifiable_reward import verify


def test_no_acute_mi_stays_negative():
    assert verify(
        "No - no evidence of acute mi",
        "No - no evidence of acute mi",
        "category_infarct_ischemia",
    ) == 1.0


def test_prompt_echo_is_not_rewarded_as_acute_mi():
    assert verify(
        "To answer whether there is acute MI, I need more information",
        "No - no evidence of acute mi",
        "category_infarct_ischemia",
    ) == 0.0


def test_classification_normal_not_confused_with_abnormal():
    assert verify(
        "Abnormal ECG; Pathological findings: Complete left bundle branch block",
        "Normal ECG; No significant abnormalities detected",
        "classification",
    ) == 0.0


def test_classification_yes_no_gate_mismatch_fails():
    assert verify(
        "Yes - Abnormal ECG; Pathological findings: Atrial Fibrillation",
        "No - Abnormal ECG; Pathological findings: Atrial Fibrillation",
        "classification",
    ) == 0.0


def test_afib_risk_requires_clean_single_risk_level():
    assert verify(
        'a "Low risk" if no relevant data is available. Low Risk - Yes/No',
        "Low risk - this patient is unlikely to develop atrial fibrillation",
        "afib_risk",
    ) == 0.0
    assert verify(
        "Low risk - this patient is unlikely to develop atrial fibrillation",
        "Low risk - this patient is unlikely to develop atrial fibrillation",
        "afib_risk",
    ) == 1.0
    assert verify(
        "Yes - this patient has AFib; Low risk - Risk factors: Atrial Fibrillation",
        "Low risk - this patient is unlikely to develop atrial fibrillation",
        "afib_risk",
    ) == 0.0


def test_axis_rambling_answer_gets_no_credit():
    assert verify(
        "questions about them. Here is an ECG report: Left axis deviation",
        "No - Left axis deviation (-30° to -90°)",
        "localization_qrs_axis",
    ) == 0.0


def test_interval_negative_fails_but_hr_and_qtc_tolerances_apply():
    assert verify(
        "PR interval: -29799 ms (short)",
        "PR interval: -29808 ms (short)",
        "ecg_interval",
    ) == 0.0
    assert verify("73 bpm", "81 bpm", "ecg_interval") == 1.0
    assert verify("73 bpm", "90 bpm", "ecg_interval") < 1.0
    assert verify(
        "QTc (Fridericia): 391 ms (at HR 126 bpm) - prolonged",
        "QTc (Fridericia): 410 ms (at HR 126 bpm) - prolonged",
        "ecg_interval",
    ) == 1.0
    assert verify("QT interval: 452 ms", "QT interval: 472 ms", "ecg_interval") == 1.0


def test_positive_finding_without_yes_gate_gets_partial_credit():
    assert verify(
        "Previous inferior wall MI",
        "Yes - Previous inferior wall MI",
        "category_infarct_ischemia",
    ) == 0.7


def test_pericarditis_no_longer_force_matches_without_st_changes():
    assert verify(
        ": Yes - Acute pericarditis",
        "Yes - pericarditis with ST changes",
        "category_pericarditis",
    ) < 1.0


def test_lvh_uses_ontology_synonyms():
    assert verify("LVH", "Left ventricular hypertrophy", "category_chamber_enlargement") == 1.0
    assert verify("LV hypertrophy", "Left ventricular hypertrophy", "category_chamber_enlargement") == 1.0


def test_bert_label_payload_scores_category_without_model_inference():
    pred = {"text": "LVH", "bert_labels": ["Left ventricular hypertrophy"]}
    gt = {
        "ground_truth": "Yes - Left ventricular hypertrophy",
        "bert_labels": ["Left ventricular hypertrophy"],
    }
    assert verify(pred, gt, "category_chamber_enlargement") == 1.0
    wrong = {"text": "RVH", "bert_labels": ["Right ventricular hypertrophy"]}
    assert verify(wrong, gt, "category_chamber_enlargement") == 0.0


def test_bert_rhythm_payload_still_penalizes_bad_heart_rate():
    gt = {
        "ground_truth": "Regular rhythm (HR: 71.0 bpm)",
        "bert_labels": ["Regular"],
    }
    close = {"text": "Regular rhythm (HR: 80.0 bpm)", "bert_labels": ["Regular"]}
    exact_cutoff = {"text": "Regular rhythm (HR: 81.0 bpm)", "bert_labels": ["Regular"]}
    bad = {"text": "Regular rhythm (HR: 140.0 bpm)", "bert_labels": ["Regular"]}

    assert verify(close, gt, "category_rhythm") == 1.0
    assert verify(exact_cutoff, gt, "category_rhythm") == 0.0
    assert verify(bad, gt, "category_rhythm") == 0.0


def test_classification_pathological_miss_is_near_zero_with_other_findings():
    gt = {
        "ground_truth": "Abnormal ECG; Pathological findings: Atrial Fibrillation",
        "bert_labels": ["Afib"],
    }
    other = {
        "text": "Abnormal ECG; Pathological findings: Left axis deviation",
        "bert_labels": ["Left axis deviation"],
    }
    nothing = {"text": "Abnormal ECG", "bert_labels": []}

    assert verify(other, gt, "classification") == 0.1
    assert verify(nothing, gt, "classification") == 0.0


def test_classification_borderline_miss_is_capped():
    gt = {
        "ground_truth": "Borderline ECG; Minor findings: Left axis deviation",
        "bert_labels": ["Left axis deviation"],
    }
    pred = {
        "text": "Borderline ECG; Minor findings: Right bundle branch block",
        "bert_labels": ["Right bundle branch block"],
    }

    assert verify(pred, gt, "classification") <= 0.5


def test_classification_normal_uses_diagnosis_f1():
    gt = {
        "ground_truth": "Normal ECG; No significant abnormalities detected",
        "bert_labels": ["Sinusal", "Regular", "Monomorph"],
    }
    normal = {
        "text": "Normal ECG; No significant abnormalities detected",
        "bert_labels": ["Sinusal", "Regular", "Monomorph"],
    }
    abnormal = {
        "text": "Abnormal ECG; Pathological findings: Atrial Fibrillation",
        "bert_labels": ["Afib"],
    }

    assert verify(normal, gt, "classification") == 1.0
    assert verify(abnormal, gt, "classification") == 0.0


def test_bert_rhythm_payload_requires_pristine_hard_findings():
    sinus = {
        "ground_truth": "Sinus rhythm (HR: 77.0 bpm)",
        "bert_labels": ["Sinusal"],
    }
    flutter_extra = {
        "text": "Atrial flutter; Sinus rhythm (HR: 77.0 bpm); Regular rhythm",
        "bert_labels": ["Atrial flutter", "Sinusal", "Regular"],
    }
    complex_gt = {
        "ground_truth": (
            "Sinus rhythm (HR: 90.1 bpm); Regular rhythm; "
            "Premature ventricular complex; Premature atrial complex"
        ),
        "bert_labels": [
            "Sinusal",
            "Regular",
            "Premature ventricular complex",
            "Premature atrial complex",
        ],
    }
    missing_pac = {
        "text": "Sinus rhythm (HR: 97.1 bpm); Regular rhythm; Premature ventricular complex",
        "bert_labels": ["Sinusal", "Regular", "Premature ventricular complex"],
    }

    assert verify(flutter_extra, sinus, "category_rhythm") == 0.0
    assert verify(missing_pac, complex_gt, "category_rhythm") == 0.0


def test_structural_and_random_extract_final_yesno():
    assert verify(
        ". No surrogate data. | Yes - structural heart disease is present",
        "Yes - structural heart disease is present based on echocardiography",
        "structural_heart_disease",
    ) == 1.0
    assert verify("eject | No", "No", "random_finding_question") == 1.0


def test_culprit_aliases_and_lvef_partial_bands():
    assert verify(
        "The culprit artery is the Ramus with complete occlusion",
        "The culprit artery is the Mid Circumflex with complete occlusion",
        "culprit_artery",
    ) == 0.7
    assert verify(
        "The culprit artery is the PDA with complete occlusion",
        "The culprit artery is the RCA with complete occlusion",
        "culprit_artery",
    ) == 1.0
    assert verify(
        "The left ventricular ejection fraction is 48%",
        "The left ventricular ejection fraction is 40%",
        "lvef",
    ) == 0.7
