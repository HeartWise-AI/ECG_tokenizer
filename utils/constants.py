
standard_lead_order = [
    "I", 
    "II", 
    "III", 
    "aVR", 
    "aVL", 
    "aVF", 
    "V1", 
    "V2", 
    "V3", 
    "V4", 
    "V5", 
    "V6"
]
lead_to_idx = {lead: idx for idx, lead in enumerate(standard_lead_order)}

ECG_PATTERNS_TRANSLATION = {
    "Sinusal": "Rythme sinusal normal",
    "Regular": "Rythme régulier",
    "Monomorph": "Monomorphe",
    "QS complex in V1-V2-V3": "Complexe QS dans les dérivations V1-V2-V3",
    "R complex in V5-V6": "Complexe R dans les dérivations V5-V6",
    "T wave inversion (inferior - II, III, aVF)": "Inversion de l'onde T dans les dérivations inférieures (II, III, aVF)",
    "Left bundle branch block": "Bloc de branche gauche",
    "RaVL > 11 mm": "RaVL > 11 mm",
    "SV1 + RV5 or RV6 > 35 mm": "SV1 + RV5 ou RV6 > 35 mm",
    "T wave inversion (lateral -I, aVL, V5-V6)": "Inversion de l'onde T dans les dérivations latérales (I, aVL, V5-V6)",
    "T wave inversion (anterior - V3-V4)": "Inversion de l'onde T dans les dérivations antérieures (V3-V4)",
    "Left axis deviation": "Déviation de l'axe gauche",
    "Left ventricular hypertrophy": "Hypertrophie ventriculaire gauche",
    "Bradycardia": "Bradycardie",
    "Q wave (inferior - II, III, aVF)": "Onde Q dans les dérivations inférieures (II, III, aVF)",
    "Afib": "Fibrillation auriculaire",
    "Irregularly irregular": "Rythme irrégulierement irrégulier",
    "Atrial tachycardia (>= 100 BPM)": "Tachycardie atriale (≥ 100 BPM)",
    "Nonspecific intraventricular conduction delay": "Retard de conduction intraventriculaire non spécifique",
    "Premature ventricular complex": "Complexe ventriculaire prématuré",
    "Polymorph": "Polymorphe",
    "T wave inversion (septal- V1-V2)": "Inversion de l'onde T dans les dérivations septales (V1-V2)",
    "Right bundle branch block": "Bloc de branche droit",
    "Ventricular paced": "Rythme ventriculaire électro-entraîné",
    "ST elevation (anterior - V3-V4)": "Élévation du segment ST dans les dérivations antérieures (V3-V4)",
    "ST elevation (septal - V1-V2)": "Élévation du segment ST dans les dérivations septales (V1-V2)",
    "1st degree AV block": "Bloc auriculo-ventriculaire du premier degré",
    "Premature atrial complex": "Complexe auriculaire prématuré",
    "Atrial flutter": "Flutter auriculaire",
    "rSR' in V1-V2": "rSR dans les dérivations V1-V2",
    "qRS in V5-V6-I, aVL": "qRS dans les dérivations V5-V6-I, aVL",
    "Left anterior fascicular block": "Hémibloc antérieur gauche",
    "Right axis deviation": "Déviation de l'axe droit",
    "2nd degree AV block - mobitz 1": "Bloc auriculo-ventriculaire du deuxième degré - Mobitz 1",
    "ST depression (inferior - II, III, aVF)": "Dépression du segment ST dans les dérivations inférieures (II, III, aVF)",
    "Acute pericarditis": "Péricardite aiguë",
    "ST elevation (inferior - II, III, aVF)": "Élévation du segment ST dans les dérivations inférieures (II, III, aVF)",
    "Low voltage": "Complexes QRS à bas voltage",
    "Regularly irregular": "Rythme régulièrement irrégulier",
    "Junctional rhythm": "Rythme jonctionnel",
    "Left atrial enlargement": "Hypertrophie atriale gauche",
    "ST elevation (lateral - I, aVL, V5-V6)": "Élévation du segment ST dans les dérivations latérales (I, aVL, V5-V6)",
    "Atrial paced": "Pacemaker auriculaire avec rythme auriculaire électro-entraîné",
    "Right ventricular hypertrophy": "Hypertrophie ventriculaire droite",
    "Delta wave": "Onde delta",
    "Wolff-Parkinson-White (Pre-excitation syndrome)": "Syndrome de Wolff-Parkinson-White (pré-excitation)",
    "Prolonged QT": "QT prolongé",
    "ST depression (anterior - V3-V4)": "Dépression du segment ST dans les dérivations antérieures (V3-V4)",
    "QRS complex negative in III": "Complexe QRS négatif dans la dérivation III",
    "Q wave (lateral- I, aVL, V5-V6)": "Onde Q dans les dérivations latérales (I, aVL, V5-V6)",
    "Supraventricular tachycardia": "Tachycardie supraventriculaire",
    "ST downslopping": "Inclinaison descendante du segment ST",
    "ST depression (lateral - I, avL, V5-V6)": "Dépression du segment ST dans les dérivations latérales (I, avL, V5-V6)",
    "2nd degree AV block - mobitz 2": "Bloc auriculo-ventriculaire du deuxième degré - Mobitz 2",
    "U wave": "Onde U",
    "R/S ratio in V1-V2 >1": "Rapport R/S dans les dérivations V1-V2 > 1",
    "RV1 + SV6 > 11 mm": "RV1 + SV6 supérieur à 11 mm",
    "Left posterior fascicular block": "Hémibloc postérieur gauche",
    "Right atrial enlargement": "Hypertrophie atriale droite",
    "ST depression (septal- V1-V2)": "Dépression du segment ST dans les dérivations septales (V1-V2)",
    "Q wave (septal- V1-V2)": "Onde Q dans les dérivations septales (V1-V2)",
    "Q wave (anterior - V3-V4)": "Onde Q dans les dérivations antérieures (V3-V4)",
    "ST upslopping": "Inclinaison ascendante du segment ST",
    "Right superior axis": "Axe supérieur droit",
    "Ventricular tachycardia": "Tachycardie ventriculaire",
    "ST elevation (posterior - V7-V8-V9)": "Élévation du segment ST dans les dérivations postérieures (V7-V8-V9)",
    "Ectopic atrial rhythm (< 100 BPM)": "Rythme auriculaire ectopique (< 100 BPM)",
    "Lead misplacement": "Inversion des électrodes",
    "Third Degree AV Block": "Bloc auriculo-ventriculaire complet (3ème degré)",
    "Acute MI": "Infarctus du myocarde aigu",
    "Early repolarization": "Repolarisation précoce",
    "Q wave (posterior - V7-V9)": "Onde Q dans les dérivations postérieures (V7-V9)",
    "Bi-atrial enlargement": "Hypertrophie bi-atriale",
    "LV pacing": "Stimulation ventriculaire gauche",
    "Brugada": "Syndrome de Brugada",
    "Ventricular Rhythm": "Rythme ventriculaire",
    "no_qrs": "Pas de segment QRS"
}

ECG_PATTERNS = [
    "Sinusal",
    "Regular",
    "Monomorph",
    "QS complex in V1-V2-V3",
    "R complex in V5-V6",
    "T wave inversion (inferior - II, III, aVF)",
    "Left bundle branch block",
    "RaVL > 11 mm",
    "SV1 + RV5 or RV6 > 35 mm",
    "T wave inversion (lateral -I, aVL, V5-V6)",
    "T wave inversion (anterior - V3-V4)",
    "Left axis deviation",
    "Left ventricular hypertrophy",
    "Bradycardia",
    "Q wave (inferior - II, III, aVF)",
    "Afib",
    "Irregularly irregular",
    "Atrial tachycardia (>= 100 BPM)",
    "Nonspecific intraventricular conduction delay",
    "Premature ventricular complex",
    "Polymorph",
    "T wave inversion (septal- V1-V2)",
    "Right bundle branch block",
    "Ventricular paced",
    "ST elevation (anterior - V3-V4)",
    "ST elevation (septal - V1-V2)",
    "1st degree AV block",
    "Premature atrial complex",
    "Atrial flutter",
    "rSR' in V1-V2",
    "qRS in V5-V6-I, aVL",
    "Left anterior fascicular block",
    "Right axis deviation",
    "2nd degree AV block - mobitz 1",
    "ST depression (inferior - II, III, aVF)",
    "Acute pericarditis",
    "ST elevation (inferior - II, III, aVF)",
    "Low voltage",
    "Regularly irregular",
    "Junctional rhythm",
    "Left atrial enlargement",
    "ST elevation (lateral - I, aVL, V5-V6)",
    "Atrial paced",
    "Right ventricular hypertrophy",
    "Delta wave",
    "Wolff-Parkinson-White (Pre-excitation syndrome)",
    "Prolonged QT",
    "ST depression (anterior - V3-V4)",
    "QRS complex negative in III",
    "Q wave (lateral- I, aVL, V5-V6)",
    "Supraventricular tachycardia",
    "ST downslopping",
    "ST depression (lateral - I, avL, V5-V6)",
    "2nd degree AV block - mobitz 2",
    "U wave",
    "R/S ratio in V1-V2 >1",
    "RV1 + SV6 > 11 mm",
    "Left posterior fascicular block",
    "Right atrial enlargement",
    "ST depression (septal- V1-V2)",
    "Q wave (septal- V1-V2)",
    "Q wave (anterior - V3-V4)",
    "ST upslopping",
    "Right superior axis",
    "Ventricular tachycardia",
    "ST elevation (posterior - V7-V8-V9)",
    "Ectopic atrial rhythm (< 100 BPM)",
    "Lead misplacement",
    "Third Degree AV Block",
    "Acute MI",
    "Early repolarization",
    "Q wave (posterior - V7-V9)",
    "Bi-atrial enlargement",
    "LV pacing",
    "Brugada",
    "Ventricular Rhythm",
    "no_qrs"
]

BERT_THRESHOLDS = {
    "Rhythm Disorders": {
        "macro_threshold": 0.34,
        "micro_threshold": 0.45
    },
    "Conduction Disorder": {
        "macro_threshold": 0.43,
        "micro_threshold": 0.41
    },
    "Enlargement of the heart chambers": {
        "macro_threshold": 0.38,
        "micro_threshold": 0.38
    },
    "Pericarditis": {
        "macro_threshold": 0.38,
        "micro_threshold": 0.38
    },
    "Infarction or ischemia": {
        "macro_threshold": 0.4,
        "micro_threshold": 0.4
    },
    "Other diagnoses": {
        "macro_threshold": 0.52,
        "micro_threshold": 0.56
    },
    "Sinusal": {
        "threshold": 0.43
    },
    "Regular": {
        "threshold": 0.48
    },
    "Monomorph": {
        "threshold": 0.51
    },
    "QS complex in V1-V2-V3": {
        "threshold": 0.57
    },
    "R complex in V5-V6": {
        "threshold": 0.4
    },
    "T wave inversion (inferior - II, III, aVF)": {
        "threshold": 0.6
    },
    "Left bundle branch block": {
        "threshold": 0.31
    },
    "RaVL > 11 mm": {
        "threshold": 0.65
    },
    "SV1 + RV5 or RV6 > 35 mm": {
        "threshold": 0.48
    },
    "T wave inversion (lateral -I, aVL, V5-V6)": {
        "threshold": 0.59
    },
    "T wave inversion (anterior - V3-V4)": {
        "threshold": 0.58
    },
    "Left axis deviation": {
        "threshold": 0.46
    },
    "Left ventricular hypertrophy": {
        "threshold": 0.38
    },
    "Bradycardia": {
        "threshold": 0.57
    },
    "Q wave (inferior - II, III, aVF)": {
        "threshold": 0.46
    },
    "Afib": {
        "threshold": 0.46
    },
    "Irregularly irregular": {
        "threshold": 0.58
    },
    "Atrial tachycardia (>= 100 BPM)": {
        "threshold": 0.39
    },
    "Nonspecific intraventricular conduction delay": {
        "threshold": 0.34
    },
    "Premature ventricular complex": {
        "threshold": 0.34
    },
    "Polymorph": {
        "threshold": 0.61
    },
    "T wave inversion (septal- V1-V2)": {
        "threshold": 0.65
    },
    "Right bundle branch block": {
        "threshold": 0.38
    },
    "Ventricular paced": {
        "threshold": 0.34
    },
    "ST elevation (anterior - V3-V4)": {
        "threshold": 0.46
    },
    "ST elevation (septal - V1-V2)": {
        "threshold": 0.48
    },
    "1st degree AV block": {
        "threshold": 0.31
    },
    "Premature atrial complex": {
        "threshold": 0.33
    },
    "Atrial flutter": {
        "threshold": 0.44
    },
    "rSR' in V1-V2": {
        "threshold": 0.56
    },
    "qRS in V5-V6-I, aVL": {
        "threshold": 0.63
    },
    "Left anterior fascicular block": {
        "threshold": 0.45
    },
    "Right axis deviation": {
        "threshold": 0.49
    },
    "2nd degree AV block - mobitz 1": {
        "threshold": 0.51
    },
    "ST depression (inferior - II, III, aVF)": {
        "threshold": 0.51
    },
    "Acute pericarditis": {
        "threshold": 0.38
    },
    "ST elevation (inferior - II, III, aVF)": {
        "threshold": 0.36
    },
    "Low voltage": {
        "threshold": 0.5
    },
    "Regularly irregular": {
        "threshold": 0.58
    },
    "Junctional rhythm": {
        "threshold": 0.43
    },
    "Left atrial enlargement": {
        "threshold": 0.52
    },
    "ST elevation (lateral - I, aVL, V5-V6)": {
        "threshold": 0.46
    },
    "Atrial paced": {
        "threshold": 0.42
    },
    "Right ventricular hypertrophy": {
        "threshold": 0.38
    },
    "Delta wave": {
        "threshold": 0.3
    },
    "Wolff-Parkinson-White (Pre-excitation syndrome)": {
        "threshold": 0.28
    },
    "Prolonged QT": {
        "threshold": 0.4
    },
    "ST depression (anterior - V3-V4)": {
        "threshold": 0.48
    },
    "QRS complex negative in III": {
        "threshold": 0.56
    },
    "Q wave (lateral- I, aVL, V5-V6)": {
        "threshold": 0.51
    },
    "Supraventricular tachycardia": {
        "threshold": 0.42
    },
    "ST downslopping": {
        "threshold": 0.37
    },
    "ST depression (lateral - I, avL, V5-V6)": {
        "threshold": 0.51
    },
    "2nd degree AV block - mobitz 2": {
        "threshold": 0.37
    },
    "U wave": {
        "threshold": 0.26
    },
    "R/S ratio in V1-V2 >1": {
        "threshold": 0.52
    },
    "RV1 + SV6 > 11 mm": {
        "threshold": 0.53
    },
    "Left posterior fascicular block": {
        "threshold": 0.35
    },
    "Right atrial enlargement": {
        "threshold": 0.26
    },
    "ST depression (septal- V1-V2)": {
        "threshold": 0.41
    },
    "Q wave (septal- V1-V2)": {
        "threshold": 0.51
    },
    "Q wave (anterior - V3-V4)": {
        "threshold": 0.37
    },
    "ST upslopping": {
        "threshold": 0.39
    },
    "Right superior axis": {
        "threshold": 0.43
    },
    "Ventricular tachycardia": {
        "threshold": 0.35
    },
    "ST elevation (posterior - V7-V8-V9)": {
        "threshold": 0.4
    },
    "Ectopic atrial rhythm (< 100 BPM)": {
        "threshold": 0.4
    },
    "Lead misplacement": {
        "threshold": 0.32
    },
    "Third Degree AV Block": {
        "threshold": 0.37
    },
    "Acute MI": {
        "threshold": 0.38
    },
    "Early repolarization": {
        "threshold": 0.4
    },
    "Q wave (posterior - V7-V9)": {
        "threshold": 0.34
    },
    "Bi-atrial enlargement": {
        "threshold": 0.29
    },
    "LV pacing": {
        "threshold": 0.28
    },
    "Brugada": {
        "threshold": 0.22
    },
    "Ventricular Rhythm": {
        "threshold": 0.33
    },
    "no_qrs": {
        "threshold": 0.27
    }
}

PTBXL_POWER_RATIO = 3.003154

# Dictionary data from deepecg_categories.json
DEEPECG_CATEGORIES = {
    "RHYTHM": [
        "Ventricular tachycardia",
        "Bradycardia",
        "Brugada",
        "Wolff-Parkinson-White (Pre-excitation syndrome)",
        "Atrial flutter",
        "Ectopic atrial rhythm (< 100 BPM)",
        "Atrial tachycardia (>= 100 BPM)",
        "Sinusal",
        "Ventricular Rhythm",
        "Supraventricular tachycardia",
        "Junctional rhythm",
        "Regular",
        "Regularly irregular",
        "Irregularly irregular",
        "Afib",
        "Premature ventricular complex",
        "Premature atrial complex"
    ],
    "CONDUCTION": [
        "Left anterior fascicular block",
        "Delta wave",
        "2nd degree AV block - mobitz 2",
        "Left bundle branch block",
        "Right bundle branch block",
        "Left axis deviation",
        "Atrial paced",
        "Right axis deviation",
        "Left posterior fascicular block",
        "1st degree AV block",
        "Right superior axis",
        "Nonspecific intraventricular conduction delay",
        "Third Degree AV Block",
        "2nd degree AV block - mobitz 1",
        "Prolonged QT",
        "U wave",
        "LV pacing",
        "Ventricular paced"
    ],
    "CHAMBER ENLARGEMENT": [
        "Bi-atrial enlargement",
        "Left atrial enlargement",
        "Right atrial enlargement",
        "Left ventricular hypertrophy",
        "Right ventricular hypertrophy"
    ],
    "PERICARDITIS": [
        "Acute pericarditis"
    ],
    "INFARCT, ISCHEMIA": [
        "Q wave (septal- V1-V2)",
        "ST elevation (anterior - V3-V4)",
        "Q wave (posterior - V7-V9)",
        "Q wave (inferior - II, III, aVF)",
        "Q wave (anterior - V3-V4)",
        "ST elevation (lateral - I, aVL, V5-V6)",
        "Q wave (lateral- I, aVL, V5-V6)",
        "ST depression (lateral - I, avL, V5-V6)",
        "Acute MI",
        "ST elevation (septal - V1-V2)",
        "ST elevation (inferior - II, III, aVF)",
        "ST elevation (posterior - V7-V8-V9)",
        "ST depression (inferior - II, III, aVF)",
        "ST depression (anterior - V3-V4)"
    ],
    "OTHER": [
        "ST downslopping",
        "ST depression (septal- V1-V2)",
        "R/S ratio in V1-V2 >1",
        "RV1 + SV6 > 11 mm",
        "Polymorph",
        "rSR' in V1-V2",
        "QRS complex negative in III",
        "qRS in V5-V6-I, aVL",
        "QS complex in V1-V2-V3",
        "R complex in V5-V6",
        "RaVL > 11 mm",
        "T wave inversion (septal- V1-V2)",
        "SV1 + RV5 or RV6 > 35 mm",
        "T wave inversion (inferior - II, III, aVF)",
        "Monomorph",
        "T wave inversion (anterior - V3-V4)",
        "T wave inversion (lateral -I, aVL, V5-V6)",
        "Low voltage",
        "Lead misplacement",
        "ST depression (anterior - V3-V4)",
        "Early repolarization",
        "ST upslopping",
        "no_qrs"
    ]
}

# Dictionary data from deepecg_diagnosis_translation.json
DEEPECG_DIAGNOSIS_TRANSLATION = {
    "deepecg": [
        {
            "column_name": "Sinusal",
            "translation_en": "Sinus rhythm",
            "explanation_en": "The heart's natural pacemaker (sinus node) is controlling the rhythm normally.",
            "translation_fr": "Rythme sinusal",
            "explanation_fr": "Le pacemaker cardiaque naturel (nœud sinusal) contrôle normalement le rythme."
        },
        {
            "column_name": "Regular",
            "translation_en": "Regular rhythm",
            "explanation_en": "The heart is beating at a steady, consistent pace.",
            "translation_fr": "Rythme régulier",
            "explanation_fr": "Le cœur bat à un rythme constant et régulier."
        },
        {
            "column_name": "Monomorph",
            "translation_en": "Monomorph QRS complexes",
            "explanation_en": "All QRS complexes (representing ventricular depolarization) look similar in shape.",
            "translation_fr": "Complexes QRS monomorphes",
            "explanation_fr": "Tous les complexes QRS (représentant la dépolarisation ventriculaire) ont une forme similaire."
        },
        {
            "column_name": "QS complex in V1-V2-V3",
            "translation_en": "QS complex in V1-V2-V3",
            "explanation_en": "Absence of R waves in these leads, which can indicate a previous anterior wall myocardial infarction.",
            "translation_fr": "Complexe QS en V1-V2-V3",
            "explanation_fr": "Absence d'ondes R dans ces dérivations, ce qui peut indiquer un infarctus du myocarde antérieur ancien."
        },
        {
            "column_name": "R complex in V5-V6",
            "translation_en": "R complex in V5-V6",
            "explanation_en": "Normal R wave progression in left-sided chest leads.",
            "translation_fr": "Complexe R en V5-V6",
            "explanation_fr": "Progression normale de l'onde R dans les dérivations précordiales gauches."
        },
        {
            "column_name": "T wave inversion (inferior - II, III, aVF)",
            "translation_en": "T wave inversion (inferior - II, III, aVF)",
            "explanation_en": "Abnormal T wave direction in the inferior leads, which may indicate ischemia or other cardiac issues.",
            "translation_fr": "Inversion de l'onde T (inférieure - II, III, aVF)",
            "explanation_fr": "Direction anormale de l'onde T dans les dérivations inférieures, ce qui peut indiquer une ischémie ou d'autres problèmes cardiaques."
        },
        {
            "column_name": "Left bundle branch block",
            "translation_en": "Complete left bundle branch block",
            "explanation_en": "Delayed activation of the left ventricle due to a conduction problem in the left bundle branch.",
            "translation_fr": "Bloc de branche gauche complet",
            "explanation_fr": "Activation retardée du ventricule gauche due à un problème de conduction dans la branche gauche du faisceau de His."
        },
        {
            "column_name": "RaVL > 11 mm",
            "translation_en": "RaVL > 11 mm",
            "explanation_en": "Tall R wave in lead aVL, suggestive of left ventricular hypertrophy.",
            "translation_fr": "RaVL > 11 mm",
            "explanation_fr": "Onde R élevée dans la dérivation aVL, suggérant une hypertrophie ventriculaire gauche."
        },
        {
            "column_name": "SV1 + RV5 or RV6 > 35 mm",
            "translation_en": "SV1 + RV5 or RV6 > 35 mm",
            "explanation_en": "Criterion for left ventricular hypertrophy.",
            "translation_fr": "SV1 + RV5 ou RV6 > 35 mm",
            "explanation_fr": "Critère d'hypertrophie ventriculaire gauche."
        },
        {
            "column_name": "T wave inversion (lateral -I, aVL, V5-V6)",
            "translation_en": "T wave inversion (lateral - I, aVL, V5-V6)",
            "explanation_en": "Abnormal T wave direction in lateral leads, potentially indicating ischemia or other cardiac issues.",
            "translation_fr": "Inversion de l'onde T (latérale - I, aVL, V5-V6)",
            "explanation_fr": "Direction anormale de l'onde T dans les dérivations latérales, indiquant potentiellement une ischémie ou d'autres problèmes cardiaques."
        },
        {
            "column_name": "T wave inversion (anterior - V3-V4)",
            "translation_en": "T wave inversion (anterior - V3-V4)",
            "explanation_en": "Abnormal T wave direction in anterior leads, possibly indicating ischemia or other cardiac problems.",
            "translation_fr": "Inversion de l'onde T (antérieure - V3-V4)",
            "explanation_fr": "Direction anormale de l'onde T dans les dérivations antérieures, indiquant possiblement une ischémie ou d'autres problèmes cardiaques."
        },
        {
            "column_name": "Left axis deviation",
            "translation_en": "Left axis deviation",
            "explanation_en": "The heart's electrical axis is shifted leftward, often seen with left ventricular hypertrophy or left anterior fascicular block.",
            "translation_fr": "Déviation axiale gauche",
            "explanation_fr": "L'axe électrique du cœur est dévié vers la gauche, souvent observé avec une hypertrophie ventriculaire gauche ou un hémibloc antérieur gauche."
        },
        {
            "column_name": "Left ventricular hypertrophy",
            "translation_en": "Left ventricular hypertrophy",
            "explanation_en": "Thickening of the left ventricular wall, often due to chronic high blood pressure or aortic valve disease.",
            "translation_fr": "Hypertrophie ventriculaire gauche",
            "explanation_fr": "Épaississement de la paroi du ventricule gauche, souvent dû à une hypertension artérielle chronique ou à une maladie de la valve aortique."
        },
        {
            "column_name": "Bradycardia",
            "translation_en": "Bradycardia",
            "explanation_en": "Abnormally slow heart rate, typically less than 60 beats per minute.",
            "translation_fr": "Bradycardie",
            "explanation_fr": "Rythme cardiaque anormalement lent, typiquement inférieur à 60 battements par minute."
        },
        {
            "column_name": "Q wave (inferior - II, III, aVF)",
            "translation_en": "Previous inferior wall MI",
            "explanation_en": "Deep Q waves in inferior leads, possibly indicating a previous inferior wall myocardial infarction.",
            "translation_fr": "Ancien infarctus du myocarde inférieur",
            "explanation_fr": "Ondes Q profondes dans les dérivations inférieures, indiquant possiblement un ancien infarctus du myocarde de la paroi inférieure."
        },
        {
            "column_name": "Afib",
            "translation_en": "Atrial Fibrillation",
            "explanation_en": "Irregular heart rhythm characterized by chaotic atrial electrical activity.",
            "translation_fr": "Fibrillation auriculaire",
            "explanation_fr": "Rythme cardiaque irrégulier caractérisé par une activité électrique auriculaire chaotique."
        },
        {
            "column_name": "Irregularly irregular",
            "translation_en": "Irregularly irregular rhythm",
            "explanation_en": "Rhythm lacking a consistent pattern, typical of atrial fibrillation, or mutlifocal atrial tachycardia.",
            "translation_fr": "Rythme irrégulièrement irrégulier",
            "explanation_fr": "Rythme sans schéma cohérent, typique de la fibrillation auriculaire ou de la tachycardie auriculaire multifocale."
        },
        {
            "column_name": "Atrial tachycardia (>= 100 BPM)",
            "translation_en": "Atrial tachycardia (>= 100 BPM)",
            "explanation_en": "Rapid heart rate originating from the atria.",
            "translation_fr": "Tachycardie auriculaire (>= 100 BPM)",
            "explanation_fr": "Rythme cardiaque rapide provenant des oreillettes."
        },
        {
            "column_name": "Nonspecific intraventricular conduction delay",
            "translation_en": "Nonspecific intraventricular conduction delay",
            "explanation_en": "Slowed conduction through the ventricles not meeting criteria for specific bundle branch blocks.",
            "translation_fr": "Retard de conduction intraventriculaire non spécifique",
            "explanation_fr": "Conduction ralentie à travers les ventricules ne répondant pas aux critères de blocs de branche spécifiques."
        },
        {
            "column_name": "Premature ventricular complex",
            "translation_en": "Premature ventricular complex",
            "explanation_en": "Early beats originating from the ventricles.",
            "translation_fr": "Complexe ventriculaire prématuré",
            "explanation_fr": "Battements précoces provenant des ventricules."
        },
        {
            "column_name": "Polymorph",
            "translation_en": "Polymorph",
            "explanation_en": "QRS complexes with varying shapes, indicating multiple origins of ventricular depolarization.",
            "translation_fr": "Polymorphe",
            "explanation_fr": "Complexes QRS de formes variées, indiquant des origines multiples de la dépolarisation ventriculaire."
        },
        {
            "column_name": "T wave inversion (septal- V1-V2)",
            "translation_en": "T wave inversion (septal- V1-V2)",
            "explanation_en": "Abnormal T wave direction in septal leads, potentially indicating ischemia or other issues.",
            "translation_fr": "Inversion de l'onde T (septale - V1-V2)",
            "explanation_fr": "Direction anormale de l'onde T dans les dérivations septales, indiquant potentiellement une ischémie ou d'autres problèmes."
        },
        {
            "column_name": "Right bundle branch block",
            "translation_en": "Right bundle branch block",
            "explanation_en": "Delayed activation of the right ventricle due to a conduction problem in the right bundle branch.",
            "translation_fr": "Bloc de branche droit",
            "explanation_fr": "Activation retardée du ventricule droit due à un problème de conduction dans la branche droite du faisceau de His."
        },
        {
            "column_name": "Ventricular paced",
            "translation_en": "Ventricular paced",
            "explanation_en": "The rhythm is being controlled by an artificial pacemaker stimulating the ventricles.",
            "translation_fr": "Stimulation ventriculaire",
            "explanation_fr": "Le rythme est contrôlé par un stimulateur cardiaque artificiel stimulant les ventricules."
        },
        {
            "column_name": "ST elevation (anterior - V3-V4)",
            "translation_en": "ST elevation (anterior - V3-V4)",
            "explanation_en": "Elevation of the ST segment in anterior leads, potentially indicating acute myocardial infarction or other conditions.",
            "translation_fr": "Élévation du segment ST (antérieur - V3-V4)",
            "explanation_fr": "Élévation du segment ST dans les dérivations antérieures, indiquant potentiellement un infarctus aigu du myocarde ou d'autres conditions."
        },
        {
            "column_name": "ST elevation (septal - V1-V2)",
            "translation_en": "ST elevation (septal - V1-V2)",
            "explanation_en": "Elevation of the ST segment in septal leads, possibly indicating acute myocardial infarction or other conditions.",
            "translation_fr": "Élévation du segment ST (septal - V1-V2)",
            "explanation_fr": "Élévation du segment ST dans les dérivations septales, indiquant possiblement un infarctus aigu du myocarde ou d'autres conditions."
        },
        {
            "column_name": "1st degree AV block",
            "translation_en": "1st degree AV block",
            "explanation_en": "Delayed conduction between atria and ventricles, seen as a prolonged PR interval.",
            "translation_fr": "Bloc AV du premier degré",
            "explanation_fr": "Conduction retardée entre les oreillettes et les ventricules, observée comme un intervalle PR prolongé."
        },
        {
            "column_name": "Premature atrial complex",
            "translation_en": "Premature atrial complex",
            "explanation_en": "Early beats originating from the atria.",
            "translation_fr": "Complexe auriculaire prématuré",
            "explanation_fr": "Battements précoces provenant des oreillettes."
        },
        {
            "column_name": "Atrial flutter",
            "translation_en": "Atrial flutter",
            "explanation_en": "Rapid, regular atrial rhythm, typically around 300 beats per minute.",
            "translation_fr": "Flutter auriculaire",
            "explanation_fr": "Rythme auriculaire rapide et régulier, typiquement autour de 300 battements par minute."
        },
        {
            "column_name": "rSR in V1-V2",
            "translation_en": "rSR' in V1-V2",
            "explanation_en": "A specific QRS pattern seen in right bundle branch block.",
            "translation_fr": "rSR' en V1-V2",
            "explanation_fr": "Un motif QRS spécifique observé dans le bloc de branche droit."
        },
        {
            "column_name": "qRS in V5-V6-I, aVL",
            "translation_en": "qRS in V5-V6-I, aVL",
            "explanation_en": "A specific QRS pattern seen in left anterior fascicular block.",
            "translation_fr": "qRS en V5-V6-I, aVL",
            "explanation_fr": "Un motif QRS spécifique observé dans l'hémibloc antérieur gauche."
        },
        {
            "column_name": "Left anterior fascicular block",
            "translation_en": "Left anterior fascicular block",
            "explanation_en": "Delayed conduction in the left anterior fascicle of the left bundle branch.",
            "translation_fr": "Hémibloc antérieur gauche",
            "explanation_fr": "Conduction retardée dans le faisceau antérieur gauche."
        },
        {
            "column_name": "Right axis deviation",
            "translation_en": "Right axis deviation",
            "explanation_en": "The heart's electrical axis is shifted rightward, often seen with right ventricular hypertrophy or left posterior fascicular block.",
            "translation_fr": "Déviation axiale droite",
            "explanation_fr": "L'axe électrique du cœur est dévié vers la droite, souvent observé avec une hypertrophie ventriculaire droite ou un hémibloc postérieur gauche."
        },
        {
            "column_name": "2nd degree AV block - mobitz 1",
            "translation_en": "2nd degree AV block - Mobitz 1",
            "explanation_en": "Intermittent failure of atrial impulses to conduct to the ventricles, with progressive PR prolongation before the dropped beat.",
            "translation_fr": "Bloc AV du deuxième degré - Mobitz 1",
            "explanation_fr": "Échec intermittent des impulsions auriculaires à se conduire vers les ventricules, avec prolongation progressive de l'intervalle PR avant le battement manqué."
        },
        {
            "column_name": "ST depression (inferior - II, III, aVF)",
            "translation_en": "ST depression (inferior leads- II, III, aVF)",
            "explanation_en": "Depression of the ST segment in inferior leads, potentially indicating ischemia.",
            "translation_fr": "Sous-décalage du segment ST (dérivations inférieures - II, III, aVF)",
            "explanation_fr": "Dépression du segment ST dans les dérivations inférieures, indiquant potentiellement une ischémie."
        },
        {
            "column_name": "Acute pericarditis",
            "translation_en": "Acute pericarditis",
            "explanation_en": "Inflammation of the pericardium, often causing diffuse ST elevation and PR depression.",
            "translation_fr": "Péricardite aiguë",
            "explanation_fr": "Inflammation du péricarde, causant souvent un sus-décalage diffus du segment ST et une dépression du segment PR."
        },
        {
            "column_name": "ST elevation (inferior - II, III, aVF)",
            "translation_en": "ST elevation (inferior leads - II, III, aVF)",
            "explanation_en": "Elevation of the ST segment in inferior leads, potentially indicating acute myocardial infarction or other conditions.",
            "translation_fr": "Sus-décalage du segment ST (dérivations inférieures - II, III, aVF)",
            "explanation_fr": "Élévation du segment ST dans les dérivations inférieures, indiquant potentiellement un infarctus aigu du myocarde ou d'autres conditions."
        },
        {
            "column_name": "Low voltage",
            "translation_en": "Low voltage QRS",
            "explanation_en": "Abnormally small QRS complexes throughout the ECG, seen in various conditions including pericardial effusion.",
            "translation_fr": "QRS de bas voltage",
            "explanation_fr": "Complexes QRS anormalement petits sur l'ensemble de l'ECG, observés dans diverses conditions, notamment l'épanchement péricardique."
        },
        {
            "column_name": "Regularly irregular",
            "translation_en": "Regularly irregular Rhythm",
            "explanation_en": "A rhythm with a repeating pattern of irregularity, such as in some forms of heart block.",
            "translation_fr": "Rythme régulièrement irrégulier",
            "explanation_fr": "Un rythme avec un schéma répétitif d'irrégularité, comme dans certaines formes de bloc cardiaque."
        },
        {
            "column_name": "Bifid",
            "translation_en": "Bifid",
            "explanation_en": "Split or notched appearance of certain ECG waves.",
            "translation_fr": "Bifide",
            "explanation_fr": "Apparence divisée ou encochée de certaines ondes ECG."
        },
        {
            "column_name": "Junctional rhythm",
            "translation_en": "Junctional rhythm",
            "explanation_en": "Heart rhythm originating from the AV junction, often a backup when the sinus node fails.",
            "translation_fr": "Rythme jonctionnel",
            "explanation_fr": "Rythme cardiaque provenant de la jonction AV, souvent un rythme de secours lorsque le nœud sinusal est défaillant."
        },
        {
            "column_name": "Left atrial enlargement",
            "translation_en": "Left atrial enlargement",
            "explanation_en": "Increased size of the left atrium, often due to chronic conditions like hypertension or mitral valve disease.",
            "translation_fr": "Hypertrophie auriculaire gauche",
            "explanation_fr": "Augmentation de la taille de l'oreillette gauche, souvent due à des conditions chroniques comme l'hypertension ou une maladie de la valve mitrale."
        },
        {
            "column_name": "ST elevation (lateral - I, aVL, V5-V6)",
            "translation_en": "ST elevation (lateral leads- I, aVL, V5-V6)",
            "explanation_en": "Elevation of the ST segment in lateral leads, potentially indicating acute myocardial infarction or other conditions",
            "translation_fr": "Sus-décalage du segment ST (dérivations latérales - I, aVL, V5-V6)",
            "explanation_fr": "Élévation du segment ST dans les dérivations latérales, indiquant potentiellement un infarctus aigu du myocarde ou d'autres conditions."
        },
        {
            "column_name": "Atrial paced",
            "translation_en": "Atrial paced",
            "explanation_en": "The rhythm is being controlled by an artificial pacemaker stimulating the atria.",
            "translation_fr": "Stimulation auriculaire",
            "explanation_fr": "Le rythme est contrôlé par un stimulateur cardiaque artificiel stimulant les oreillettes."
        },
        {
            "column_name": "Right ventricular hypertrophy",
            "translation_en": "Right ventricular hypertrophy",
            "explanation_en": "Thickening of the right ventricular wall, often due to conditions like pulmonary hypertension.",
            "translation_fr": "Hypertrophie ventriculaire droite",
            "explanation_fr": "Épaississement de la paroi du ventricule droit, souvent dû à des conditions comme l'hypertension pulmonaire."
        },
        {
            "column_name": "Delta wave",
            "translation_en": "Delta wave",
            "explanation_en": "Early activation of the ventricles seen in Wolff-Parkinson-White syndrome.",
            "translation_fr": "Onde delta",
            "explanation_fr": "Activation précoce des ventricules observée dans le syndrome de Wolff-Parkinson-White."
        },
        {
            "column_name": "Wolff-Parkinson-White (Pre-excitation syndrome)",
            "translation_en": "Wolff-Parkinson-White (Pre-excitation syndrome)",
            "explanation_en": "Presence of an accessory pathway between atria and ventricles, causing early ventricular activation.",
            "translation_fr": "Wolff-Parkinson-White (Syndrome de pré-excitation)",
            "explanation_fr": "Présence d'une voie accessoire entre les oreillettes et les ventricules, causant une activation ventriculaire précoce."
        },
        {
            "column_name": "Prolonged QT",
            "translation_en": "Prolonged QT",
            "explanation_en": "Abnormally long QT interval, which can increase risk of certain arrhythmias.",
            "translation_fr": "QT prolongé",
            "explanation_fr": "Intervalle QT anormalement long, ce qui peut augmenter le risque de certaines arythmies."
        },
        {
            "column_name": "ST depression (anterior - V3-V4)",
            "translation_en": "ST depression (anterior leads - V3-V4)",
            "explanation_en": "Depression of the ST segment in anterior leads, potentially indicating ischemia.",
            "translation_fr": "Sous-décalage du segment ST (dérivations antérieures - V3-V4)",
            "explanation_fr": "Dépression du segment ST dans les dérivations antérieures, indiquant potentiellement une ischémie."
        },
        {
            "column_name": "QRS complex negative in III",
            "translation_en": "QRS complex negative in III",
            "explanation_en": "Deep S wave in lead III, can be normal or indicate left anterior fascicular block.",
            "translation_fr": "Complexe QRS négatif en III",
            "explanation_fr": "Onde S profonde dans la dérivation III, peut être normal ou indiquer un hémibloc antérieur gauche."
        },
        {
            "column_name": "RaVL + SV3 > 28 mm (H) or 20 mm (F)",
            "translation_en": "RaVL + SV3 > 28 mm (H) or 20 mm (F)",
            "explanation_en": "Criterion for left ventricular hypertrophy.",
            "translation_fr": "RaVL + SV3 > 28 mm (H) ou 20 mm (F)",
            "explanation_fr": "Critère d'hypertrophie ventriculaire gauche."
        },
        {
            "column_name": "Q wave (lateral- I, aVL, V5-V6)",
            "translation_en": "Previous lateral (V5, V6, I, aVL) Myocardial Infarction",
            "explanation_en": "Deep Q waves in lateral leads, possibly indicating a previous lateral wall myocardial infarction.",
            "translation_fr": "Ancien infarctus du myocarde latéral (V5, V6, I, aVL)",
            "explanation_fr": "Ondes Q profondes dans les dérivations latérales, indiquant possiblement un ancien infarctus du myocarde de la paroi latérale."
        },
        {
            "column_name": "Hyperacute T wave (lateral leads, V5-V6)",
            "translation_en": "Hyperacute T wave (lateral leads, V5-V6)",
            "explanation_en": "Tall, peaked T waves in lateral leads, potentially an early sign of acute myocardial infarction or hyperkalemia.",
            "translation_fr": "Onde T hyperaiguë (dérivations latérales, V5-V6)",
            "explanation_fr": "Ondes T hautes et pointues dans les dérivations latérales, potentiellement un signe précoce d'infarctus aigu du myocarde ou d'hyperkaliémie."
        },
        {
            "column_name": "Hyperacute T wave (septal leads, V1-V2)",
            "translation_en": "Hyperacute T wave (septal leads, V1-V2)",
            "explanation_en": "Tall, peaked T waves in septal leads, potentially an early sign of acute myocardial infarction or hyperkalemia.",
            "translation_fr": "Onde T hyperaiguë (dérivations septales, V1-V2)",
            "explanation_fr": "Ondes T hautes et pointues dans les dérivations septales, potentiellement un signe précoce d'infarctus aigu du myocarde ou d'hyperkaliémie."
        },
        {
            "column_name": "Supraventricular tachycardia",
            "translation_en": "Supraventricular tachycardia",
            "explanation_en": "Rapid heart rate originating above the ventricles.",
            "translation_fr": "Tachycardie supraventriculaire",
            "explanation_fr": "Rythme cardiaque rapide provenant au-dessus des ventricules."
        },
        {
            "column_name": "ST downslopping",
            "translation_en": "Diffuse ST segment downslopping",
            "explanation_en": "Downward sloping of the ST segment across multiple leads, can indicate ischemia.",
            "translation_fr": "Sous-décalage diffus du segment ST",
            "explanation_fr": "Pente descendante du segment ST sur plusieurs dérivations, peut indiquer une ischémie."
        },
        {
            "column_name": "ST depression (lateral - I, avL, V5-V6)",
            "translation_en": "ST depression (lateral leads - I, avL, V5-V6)",
            "explanation_en": "Depression of the ST segment in lateral leads, potentially indicating ischemia.",
            "translation_fr": "Sous-décalage du segment ST (dérivations latérales - I, aVL, V5-V6)",
            "explanation_fr": "Dépression du segment ST dans les dérivations latérales, indiquant potentiellement une ischémie."
        },
        {
            "column_name": "2nd degree AV block - mobitz 2",
            "translation_en": "2nd degree AV block - mobitz 2",
            "explanation_en": "Intermittent failure of atrial impulses to conduct to the ventricles, without PR prolongation before the dropped beat.",
            "translation_fr": "Bloc AV du deuxième degré - Mobitz 2",
            "explanation_fr": "Échec intermittent des impulsions auriculaires à se conduire vers les ventricules, sans prolongation de l'intervalle PR avant le battement manqué."
        },
        {
            "column_name": "U wave",
            "translation_en": "U wave",
            "explanation_en": "Small wave following the T wave, can be normal or indicate electrolyte imbalances or other conditions.",
            "translation_fr": "Onde U",
            "explanation_fr": "Petite onde suivant l'onde T, peut être normale ou indiquer des déséquilibres électrolytiques ou d'autres conditions."
        },
        {
            "column_name": "Large >0.08 s",
            "translation_en": "Large >0.08 s",
            "explanation_en": "Refers to a prolonged QRS duration, indicating delayed ventricular conduction.",
            "translation_fr": "Large >0,08 s",
            "explanation_fr": "Fait référence à une durée QRS prolongée, indiquant une conduction ventriculaire retardée."
        },
        {
            "column_name": "R/S ratio in V1-V2 >1",
            "translation_en": "R/S ratio in V1-V2 >1",
            "explanation_en": "Tall R waves in right precordial leads, can indicate right ventricular hypertrophy.",
            "translation_fr": "Rapport R/S en V1-V2 >1",
            "explanation_fr": "Ondes R élevées dans les dérivations précordiales droites, peut indiquer une hypertrophie ventriculaire droite."
        },
        {
            "column_name": "RV1 + SV6 11 mm",
            "translation_en": "RV1 + SV6 > 11 mm",
            "explanation_en": "Another criterion for right ventricular hypertrophy.",
            "translation_fr": "RV1 + SV6 > 11 mm",
            "explanation_fr": "Critère d'hypertrophie ventriculaire droite."
        },
        {
            "column_name": "RV1 + SV6 > 11 mm",
            "translation_en": "RV1 + SV6 > 11 mm",
            "explanation_en": "Another criterion for right ventricular hypertrophy.",
            "translation_fr": "RV1 + SV6 > 11 mm",
            "explanation_fr": "Un autre critère d'hypertrophie ventriculaire droite."
        },
        {
            "column_name": "Left posterior fascicular block",
            "translation_en": "Left posterior fascicular block",
            "explanation_en": "Delayed conduction in the left posterior fascicle of the left bundle branch.",
            "translation_fr": "Hémibloc postérieur gauche",
            "explanation_fr": "Conduction retardée dans le faisceau postérieur gauche de la branche gauche du faisceau de His."
        },
        {
            "column_name": "Right atrial enlargement",
            "translation_en": "Right atrial enlargement",
            "explanation_en": "Increased size of the right atrium, often due to conditions like pulmonary hypertension or tricuspid valve disease.",
            "translation_fr": "Hypertrophie auriculaire droite",
            "explanation_fr": "Augmentation de la taille de l'oreillette droite, souvent due à des conditions comme l'hypertension pulmonaire ou une maladie de la valve tricuspide."
        },
        {
            "column_name": "ST depression (septal- V1-V2)",
            "translation_en": "ST depression (septal leads- V1-V2)",
            "explanation_en": "Depression of the ST segment in septal leads, potentially indicating ischemia.",
            "translation_fr": "Sous-décalage du segment ST (dérivations septales - V1-V2)",
            "explanation_fr": "Dépression du segment ST dans les dérivations septales, indiquant potentiellement une ischémie."
        },
        {
            "column_name": "Q wave (septal- V1-V2)",
            "translation_en": "Previous septal (V1-V2) Myocardial Infarction",
            "explanation_en": "Deep Q waves in septal leads, possibly indicating a previous septal myocardial infarction.",
            "translation_fr": "Ancien infarctus du myocarde septal (V1-V2)",
            "explanation_fr": "Ondes Q profondes dans les dérivations septales, indiquant possiblement un ancien infarctus du myocarde septal."
        },
        {
            "column_name": "Q wave (anterior - V3-V4)",
            "translation_en": "Previous anterior (V3-V4) Myocardial Infarction",
            "explanation_en": "Deep Q waves in anterior leads, possibly indicating a previous anterior wall myocardial infarction.",
            "translation_fr": "Ancien infarctus du myocarde antérieur (V3-V4)",
            "explanation_fr": "Ondes Q profondes dans les dérivations antérieures, indiquant possiblement un ancien infarctus du myocarde de la paroi antérieure."
        },
        {
            "column_name": "Hyperacute T wave (anterior leads, V3-V4)",
            "translation_en": "Hyperacute T wave (anterior leads, V3-V4)",
            "explanation_en": "Tall, peaked T waves in anterior leads, potentially an early sign of acute myocardial infarction.",
            "translation_fr": "Onde T hyperaiguë (dérivations antérieures, V3-V4)",
            "explanation_fr": "Ondes T hautes et pointues dans les dérivations antérieures, potentiellement un signe précoce d'infarctus aigu du myocarde."
        },
        {
            "column_name": "ST upslopping",
            "translation_en": "ST upslopping",
            "explanation_en": "Upward sloping of the ST segment, can be normal or indicate early repolarization as seen in young adults, men, athletes, african americans.",
            "translation_fr": "Sus-décalage ascendant du segment ST",
            "explanation_fr": "Pente ascendante du segment ST, peut être normal ou indiquer une repolarisation précoce comme observé chez les jeunes adultes, les hommes, les athlètes, les afro-américains."
        },
        {
            "column_name": "Right superior axis",
            "translation_en": "Right superior axis",
            "explanation_en": "Extreme rightward deviation of the heart's electrical axis.",
            "translation_fr": "Axe droit supérieur",
            "explanation_fr": "Déviation extrême vers la droite de l'axe électrique du cœur."
        },
        {
            "column_name": "Auricular bigeminy",
            "translation_en": "Auricular bigeminy",
            "explanation_en": "Every other beat is a premature atrial complex.",
            "translation_fr": "Bigéminisme auriculaire",
            "explanation_fr": "Un battement sur deux est un complexe auriculaire prématuré."
        },
        {
            "column_name": "Ventricular tachycardia",
            "translation_en": "Ventricular tachycardia",
            "explanation_en": "Rapid heart rhythm originating from the ventricles.",
            "translation_fr": "Tachycardie ventriculaire",
            "explanation_fr": "Rythme cardiaque rapide provenant des ventricules."
        },
        {
            "column_name": "ST elevation (posterior - V7-V8-V9)",
            "translation_en": "ST elevation (posterior leads- V7-V8-V9)",
            "explanation_en": "Elevation of the ST segment in posterior leads, potentially indicating acute posterior wall myocardial infarction.",
            "translation_fr": "Sus-décalage du segment ST (dérivations postérieures - V7-V8-V9)",
            "explanation_fr": "Élévation du segment ST dans les dérivations postérieures, indiquant potentiellement un infarctus aigu du myocarde de la paroi postérieure."
        },
        {
            "column_name": "Ectopic atrial rhythm (< 100 BPM)",
            "translation_en": "Ectopic atrial rhythm (< 100 BPM)",
            "explanation_en": "Heart rhythm originating from an atrial site other than the sinus node.",
            "translation_fr": "Rythme auriculaire ectopique (< 100 BPM)",
            "explanation_fr": "Rythme cardiaque provenant d'un site auriculaire autre que le nœud sinusal."
        },
        {
            "column_name": "Lead misplacement",
            "translation_en": "Lead misplacement",
            "explanation_en": "Incorrect placement of ECG leads, leading to misleading ECG patterns.",
            "translation_fr": "Mauvais placement des électrodes",
            "explanation_fr": "Placement incorrect des électrodes ECG, conduisant à des tracés ECG trompeurs."
        },
        {
            "column_name": "Biphasic",
            "translation_en": "Biphasic",
            "explanation_en": "Waves with both positive and negative components.",
            "translation_fr": "Biphasique",
            "explanation_fr": "Ondes avec des composantes à la fois positives et négatives."
        },
        {
            "column_name": "Ventricular bigeminy",
            "translation_en": "Ventricular bigeminy",
            "explanation_en": "Every other beat is a premature ventricular complex.",
            "translation_fr": "Bigéminisme ventriculaire",
            "explanation_fr": "Un battement sur deux est un complexe ventriculaire prématuré."
        },
        {
            "column_name": "J wave",
            "translation_en": "J wave",
            "explanation_en": "Small positive deflection at the junction of the QRS complex and ST segment.",
            "translation_fr": "Onde J",
            "explanation_fr": "Petite déflexion positive à la jonction du complexe QRS et du segment ST."
        },
        {
            "column_name": "Tall >2.5 mm",
            "translation_en": "Tall >2.5 mm",
            "explanation_en": "Refers to unusually tall T waves, which can indicate hyperkalemia or other conditions.",
            "translation_fr": "Élevé >2,5 mm",
            "explanation_fr": "Fait référence à des ondes T inhabituellement hautes, qui peuvent indiquer une hyperkaliémie ou d'autres conditions."
        },
        {
            "column_name": "Third Degree AV Block",
            "translation_en": "Complete (Third Degree) Atrioventricular Block",
            "explanation_en": "Complete dissociation between atrial and ventricular rhythms.",
            "translation_fr": "Bloc atrioventriculaire complet (du troisième degré)",
            "explanation_fr": "Dissociation complète entre les rythmes auriculaire et ventriculaire."
        },
        {
            "column_name": "Sinus Pause",
            "translation_en": "Sinus Pause",
            "explanation_en": "Temporary absence of sinus node activity.",
            "translation_fr": "Pause sinusale",
            "explanation_fr": "Absence temporaire de l'activité du nœud sinusal."
        },
        {
            "column_name": "Acute MI",
            "translation_en": "Acute Myocardial Infarction",
            "explanation_en": "Acute myocardial infarction, commonly known as a heart attack.",
            "translation_fr": "Infarctus aigu du myocarde",
            "explanation_fr": "Infarctus aigu du myocarde, communément appelé crise cardiaque."
        },
        {
            "column_name": "Q wave (posterior - V7-V9)",
            "translation_en": "Previous posterior MI (posterior leads - V7-V9)",
            "explanation_en": "Deep Q waves in posterior leads, possibly indicating a previous posterior wall myocardial infarction.",
            "translation_fr": "Infarctus postérieur ancien (dérivations postérieures - V7-V9)",
            "explanation_fr": "Ondes Q profondes dans les dérivations postérieures, indiquant possiblement un ancien infarctus du myocarde de la paroi postérieure."
        },
        {
            "column_name": "Bi-atrial enlargement",
            "translation_en": "Bi-atrial enlargement",
            "explanation_en": "Increased size of both atria.",
            "translation_fr": "Hypertrophie bi-auriculaire",
            "explanation_fr": "Augmentation de la taille des deux oreillettes."
        },
        {
            "column_name": "LV pacing",
            "translation_en": "Left ventricular pacing",
            "explanation_en": "The rhythm is being controlled by an artificial pacemaker stimulating the left ventricle.",
            "translation_fr": "Stimulation ventriculaire gauche",
            "explanation_fr": "Le rythme est contrôlé par un stimulateur cardiaque artificiel stimulant le ventricule gauche."
        },
        {
            "column_name": "Dextrocardia",
            "translation_en": "Dextrocardia",
            "explanation_en": "The heart is positioned on the right side of the chest instead of the left.",
            "translation_fr": "Dextrocardie",
            "explanation_fr": "Le cœur est positionné du côté droit de la poitrine au lieu du côté gauche."
        },
        {
            "column_name": "Brugada",
            "translation_en": "Brugada",
            "explanation_en": "Brugada syndrome, a genetic disorder characterized by specific ST elevation in right precordial leads.",
            "translation_fr": "Brugada",
            "explanation_fr": "Syndrome de Brugada, un trouble génétique caractérisé par un sus-décalage spécifique du segment ST dans les dérivations précordiales droites."
        },
        {
            "column_name": "Ventricular Rhythm",
            "translation_en": "Ventricular Rhythm",
            "explanation_en": "Heart rhythm originating from the ventricles.",
            "translation_fr": "Rythme ventriculaire",
            "explanation_fr": "Rythme cardiaque provenant des ventricules."
        },
        {
            "column_name": "no_qrs",
            "translation_en": "Unable to identify QRS complexes",
            "explanation_en": "The ECG does not show identifiable QRS complexes, which may indicate extremely poor signal quality or asystole.",
            "translation_fr": "Impossibilité d'identifier les complexes QRS",
            "explanation_fr": "L'ECG ne montre pas de complexes QRS identifiables, ce qui peut indiquer une qualité de signal extrêmement faible ou une asystolie."
        },
        {
            "column_name": "diagnosis_low_quality",
            "translation_en": "Poor ECG quality",
            "explanation_en": "The ECG signal is of low quality, making accurate interpretation difficult or impossible.",
            "translation_fr": "Qualité ECG médiocre",
            "explanation_fr": "Le signal ECG est de faible qualité, rendant l'interprétation précise difficile ou impossible."
        },
        {
            "column_name": "ST depression (posterior - V7-V8-V9)",
            "translation_en": "ST depression (posterior - V7-V8-V9)",
            "explanation_en": "Depression of the ST segment in posterior leads, potentially indicating ischemia or other conditions.",
            "translation_fr": "Dépression du segment ST (postérieur - V7-V8-V9)",
            "explanation_fr": "Dépression du segment ST dans les dérivations postérieures, indiquant potentiellement une ischémie ou d'autres conditions."
        },
        {
            "column_name": "Early repolarization",
            "translation_en": "Early repolarization",
            "explanation_en": "A pattern on an EKG that is usually normal and can be seen in healthy young adults, but sometimes associated with a higher risk of irregular heartbeats.",
            "translation_fr": "Repolarisation précoce",
            "explanation_fr": "Un motif sur un EKG qui est généralement normal et peut être observé chez les jeunes adultes en bonne santé, mais parfois associé à un risque plus élevé de battements de coeur irréguliers."
        }
    ],
    "echonext": [
        {
            "column_name": "lvef_lte_45",
            "translation_en": "Left ventricular ejection fraction <= 45%",
            "explanation_en": "The left ventricle of the heart pumps out 45% or less of its filled volume with each heartbeat, which may indicate heart failure or other heart health issues.",
            "translation_fr": "Fraction d'éjection du ventricule gauche <= 45%",
            "explanation_fr": "Le ventricule gauche du cœur pompe 45% ou moins de son volume rempli à chaque battement de cœur, ce qui peut indiquer une insuffisance cardiaque ou d'autres problèmes de santé cardiaque."
        },
        {
            "column_name": "lvwt_gte_13",
            "translation_en": "Left ventricular wall thickness >= 13 mm",
            "explanation_en": "The wall of the left ventricle of the heart has a thickness of 13mm or more, which could indicate conditions like hypertrophic cardiomyopathy or might be a consequence of persistent high blood pressure.",
            "translation_fr": "Épaisseur de la paroi du ventricule gauche >= 13 mm",
            "explanation_fr": "La paroi du ventricule gauche du cœur a une épaisseur de 13 mm ou plus, ce qui pourrait indiquer des conditions comme la cardiomyopathie hypertrophique ou peut être une conséquence de l'hypertension artérielle persistante."
        },
        {
            "column_name": "aortic_stenosis_moderate_severe",
            "translation_en": "Moderate to severe aortic stenosis",
            "explanation_en": "A condition characterized by a narrowing of the aortic valve opening, restricting the blood flow from the left ventricle to the aorta, considered to be of moderate to severe degree.",
            "translation_fr": "Sténose aortique modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un rétrécissement de l'ouverture de la valve aortique, limitant le flux sanguin du ventricule gauche à l'aorte, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "aortic_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe aortic regurgitation",
            "explanation_en": "A condition characterized by a leaky aortic valve that leads to backward flow of blood from the aorta into the left ventricle, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation aortique modérée à sévère",
            "explanation_fr": "Une condition caractérisée par une valve aortique qui fuit, ce qui entraîne un flux sanguin inverse de l'aorte vers le ventricule gauche, considéré comme modéré à sévère."
        },
        {
            "column_name": "mitral_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe mitral regurgitation",
            "explanation_en": "A condition characterized by backward blood flow from the left ventricle into the left atrium due to failure of the mitral valve to close completely, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation mitrale modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un reflux sanguin du ventricule gauche vers l'oreillette gauche en raison de l'échec de la valve mitrale à se fermer complètement, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "tricuspid_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe tricuspid regurgitation",
            "explanation_en": "A condition characterized by backward flow of blood from the right ventricle into the right atrium due to incomplete closure of the tricuspid valve, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation tricuspide modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un reflux sanguin du ventricule droit vers l'oreillette droite en raison de la fermeture incomplète de la valve tricuspide, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "pulmonary_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe pulmonary regurgitation",
            "explanation_en": "A condition characterized by the backward flow of blood from the pulmonary artery into the right ventricle, as a result of incomplete closure of the pulmonary valve, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation pulmonaire modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un reflux sanguin de l'artère pulmonaire vers le ventricule droit, en raison de la fermeture incomplète de la valve pulmonaire, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "rv_systolic_dysfunction_moderate_severe",
            "translation_en": "Moderate to severe right ventricular systolic dysfunction",
            "explanation_en": "A condition characterized by a diminished ability of the right ventricle of the heart to contract or pump blood properly, considered to be of a moderate to severe degree.",
            "translation_fr": "Dysfonction systolique ventriculaire droite modérée à sévère",
            "explanation_fr": "Une condition caractérisée par une capacité réduite du ventricule droit du cœur à se contracter ou à pomper correctement le sang, considérée comme de degré modéré à sévère."
        },
        {
            "column_name": "pericardial_effusion_moderate_large",
            "translation_en": "Moderate to large pericardial effusion",
            "explanation_en": "A condition characterized by the accumulation of an abnormal amount of fluid in the pericardial cavity surrounding the heart, considered to be of a moderate to large volume.",
            "translation_fr": "Epanchement péricardique modéré à large",
            "explanation_fr": "Une condition caractérisée par l'accumulation d'une quantité anormale de liquide dans la cavité péricardique entourant le cœur, considérée comme de volume modéré à important."
        },
        {
            "column_name": "pasp_gte_45",
            "translation_en": "Pulmonary Artery Systolic Pressure greater than or equal to 45",
            "explanation_en": "A condition characterized by a systolic pressure in the pulmonary artery that is greater than or equal to 45 mmHg, typically indicating severe pulmonary hypertension.",
            "translation_fr": "Pression systolique de l'artère pulmonaire supérieure ou égale à 45",
            "explanation_fr": "Une condition caractérisée par une pression systolique dans l'artère pulmonaire supérieure ou égale à 45 mmHg, indiquant généralement une hypertension pulmonaire sévère."
        },
        {
            "column_name": "tr_max_gte_32",
            "translation_en": "Tricuspid Regurgitant Jet Velocity greater than or equal to 32",
            "explanation_en": "A condition characterized by a tricuspid regurgitant jet velocity that is greater than or equal to 32 cm/sec, typically indicating pulmonary hypertension.",
            "translation_fr": "Vitesse du jet régurgitant tricuspide supérieure ou égale à 32",
            "explanation_fr": "Une condition caractérisée par une vitesse de jet régurgitant tricuspide supérieure ou égale à 32 cm/sec, indiquant généralement une hypertension pulmonaire."
        },
        {
            "column_name": "shd",
            "translation_en": "Structural Heart Disease",
            "explanation_en": "Refers to a defect or abnormality in the heart's structure—its valves, myocardium, the great vessels, septa, etc.—regardless of its cause.",
            "translation_fr": "Maladie Cardiaque Structurale",
            "explanation_fr": "Se réfère à un défaut ou une anomalie dans la structure du cœur—ses valves, le myocarde, les grands vaisseaux, les cloisons, etc.—quel que soit sa cause."
        }
    ],
    "echonext": [
        {
            "column_name": "lvef_lte_45",
            "translation_en": "Left ventricular ejection fraction <= 45%",
            "explanation_en": "The left ventricle of the heart pumps out 45% or less of its filled volume with each heartbeat, which may indicate heart failure or other heart health issues.",
            "translation_fr": "Fraction d'éjection du ventricule gauche <= 45%",
            "explanation_fr": "Le ventricule gauche du cœur pompe 45% ou moins de son volume rempli à chaque battement de cœur, ce qui peut indiquer une insuffisance cardiaque ou d'autres problèmes de santé cardiaque."
        },
        {
            "column_name": "lvwt_gte_13",
            "translation_en": "Left ventricular wall thickness >= 13 mm",
            "explanation_en": "The wall of the left ventricle of the heart has a thickness of 13mm or more, which could indicate conditions like hypertrophic cardiomyopathy or might be a consequence of persistent high blood pressure.",
            "translation_fr": "Épaisseur de la paroi du ventricule gauche >= 13 mm",
            "explanation_fr": "La paroi du ventricule gauche du cœur a une épaisseur de 13 mm ou plus, ce qui pourrait indiquer des conditions comme la cardiomyopathie hypertrophique ou peut être une conséquence de l'hypertension artérielle persistante."
        },
        {
            "column_name": "aortic_stenosis_moderate_severe",
            "translation_en": "Moderate to severe aortic stenosis",
            "explanation_en": "A condition characterized by a narrowing of the aortic valve opening, restricting the blood flow from the left ventricle to the aorta, considered to be of moderate to severe degree.",
            "translation_fr": "Sténose aortique modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un rétrécissement de l'ouverture de la valve aortique, limitant le flux sanguin du ventricule gauche à l'aorte, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "aortic_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe aortic regurgitation",
            "explanation_en": "A condition characterized by a leaky aortic valve that leads to backward flow of blood from the aorta into the left ventricle, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation aortique modérée à sévère",
            "explanation_fr": "Une condition caractérisée par une valve aortique qui fuit, ce qui entraîne un flux sanguin inverse de l'aorte vers le ventricule gauche, considéré comme modéré à sévère."
        },
        {
            "column_name": "mitral_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe mitral regurgitation",
            "explanation_en": "A condition characterized by backward blood flow from the left ventricle into the left atrium due to failure of the mitral valve to close completely, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation mitrale modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un reflux sanguin du ventricule gauche vers l'oreillette gauche en raison de l'échec de la valve mitrale à se fermer complètement, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "tricuspid_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe tricuspid regurgitation",
            "explanation_en": "A condition characterized by backward flow of blood from the right ventricle into the right atrium due to incomplete closure of the tricuspid valve, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation tricuspide modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un reflux sanguin du ventricule droit vers l'oreillette droite en raison de la fermeture incomplète de la valve tricuspide, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "pulmonary_regurgitation_moderate_severe",
            "translation_en": "Moderate to severe pulmonary regurgitation",
            "explanation_en": "A condition characterized by the backward flow of blood from the pulmonary artery into the right ventricle, as a result of incomplete closure of the pulmonary valve, considered to be of moderate to severe degree.",
            "translation_fr": "Régurgitation pulmonaire modérée à sévère",
            "explanation_fr": "Une condition caractérisée par un reflux sanguin de l'artère pulmonaire vers le ventricule droit, en raison de la fermeture incomplète de la valve pulmonaire, considéré comme de degré modéré à sévère."
        },
        {
            "column_name": "rv_systolic_dysfunction_moderate_severe",
            "translation_en": "Moderate to severe right ventricular systolic dysfunction",
            "explanation_en": "A condition characterized by a diminished ability of the right ventricle of the heart to contract or pump blood properly, considered to be of a moderate to severe degree.",
            "translation_fr": "Dysfonction systolique ventriculaire droite modérée à sévère",
            "explanation_fr": "Une condition caractérisée par une capacité réduite du ventricule droit du cœur à se contracter ou à pomper correctement le sang, considérée comme de degré modéré à sévère."
        },
        {
            "column_name": "pericardial_effusion_moderate_large",
            "translation_en": "Moderate to large pericardial effusion",
            "explanation_en": "A condition characterized by the accumulation of an abnormal amount of fluid in the pericardial cavity surrounding the heart, considered to be of a moderate to large volume.",
            "translation_fr": "Epanchement péricardique modéré à large",
            "explanation_fr": "Une condition caractérisée par l'accumulation d'une quantité anormale de liquide dans la cavité péricardique entourant le cœur, considérée comme de volume modéré à important."
        },
        {
            "column_name": "pasp_gte_45",
            "translation_en": "Pulmonary Artery Systolic Pressure greater than or equal to 45",
            "explanation_en": "A condition characterized by a systolic pressure in the pulmonary artery that is greater than or equal to 45 mmHg, typically indicating severe pulmonary hypertension.",
            "translation_fr": "Pression systolique de l'artère pulmonaire supérieure ou égale à 45",
            "explanation_fr": "Une condition caractérisée par une pression systolique dans l'artère pulmonaire supérieure ou égale à 45 mmHg, indiquant généralement une hypertension pulmonaire sévère."
        },
        {
            "column_name": "tr_max_gte_32",
            "translation_en": "Tricuspid Regurgitant Jet Velocity greater than or equal to 32",
            "explanation_en": "A condition characterized by a tricuspid regurgitant jet velocity that is greater than or equal to 32 cm/sec, typically indicating pulmonary hypertension.",
            "translation_fr": "Vitesse du jet régurgitant tricuspide supérieure ou égale à 32",
            "explanation_fr": "Une condition caractérisée par une vitesse de jet régurgitant tricuspide supérieure ou égale à 32 cm/sec, indiquant généralement une hypertension pulmonaire."
        },
        {
            "column_name": "shd",
            "translation_en": "Structural Heart Disease",
            "explanation_en": "Refers to a defect or abnormality in the heart's structure—its valves, myocardium, the great vessels, septa, etc.—regardless of its cause.",
            "translation_fr": "Maladie Cardiaque Structurale",
            "explanation_fr": "Se réfère à un défaut ou une anomalie dans la structure du cœur—ses valves, le myocarde, les grands vaisseaux, les cloisons, etc.—quel que soit sa cause."
        }
    ]
}

# Dictionary data from deepecg.json
DEEPECG_PATHOLOGICAL_LIMIT = {
    "deepecg": {
        "pathological": [
            "2nd degree AV block - mobitz 2",
            "2nd degree AV block - mobitz 1",
            "Acute MI",
            "Acute pericarditis",
            "Afib",
            "Atrial flutter",
            "Atrial paced",
            "Atrial tachycardia (>= 100 BPM)",
            "LV pacing",
            "Bi-atrial enlargement",
            "Brugada",
            "Delta wave",
            "Ectopic atrial rhythm (< 100 BPM)",
            "Irregularly irregular",
            "Junctional rhythm",
            "Left atrial enlargement",
            "Left bundle branch block",
            "Left ventricular hypertrophy",
            "Prolonged QT",
            "Q wave (anterior - V3-V4)",
            "Q wave (inferior - II, III, aVF)",
            "Q wave (lateral- I, aVL, V5-V6)",
            "Q wave (septal- V1-V2)",
            "Q wave (posterior - V7-V9)",
            "QS complex in V1-V2-V3",
            "Right atrial enlargement",
            "Right ventricular hypertrophy",
            "Right superior axis",
            "ST depression (inferior - II, III, aVF)",
            "ST depression (lateral - I, avL, V5-V6)",
            "ST depression (septal- V1-V2)",
            "ST depression (inferior - II, III, aVF)",
            "ST downslopping",
            "ST elevation (anterior - V3-V4)",
            "ST elevation (inferior - II, III, aVF)",
            "ST elevation (lateral - I, aVL, V5-V6)",
            "ST elevation (posterior - V7-V8-V9)",
            "ST elevation (septal - V1-V2)",
            "SV1 + RV5 or RV6 > 35 mm",
            "Supraventricular tachycardia",
            "T wave inversion (lateral -I, aVL, V5-V6)",
            "T wave inversion (anterior - V3-V4)",
            "Third Degree AV Block",
            "Ventricular paced",
            "Ventricular Rhythm",
            "Ventricular tachycardia",
            "Wolff-Parkinson-White (Pre-excitation syndrome)"
        ],
        "limit": [
            "1st degree AV block",
            "Early repolarization",
            "Left anterior fascicular block",
            "Left axis deviation",
            "Left posterior fascicular block",
            "Nonspecific intraventricular conduction delay",
            "QRS complex negative in III",
            "R complex in V5-V6",
            "R/S ratio in V1-V2 >1",
            "RV1 + SV6 11 mm",
            "RaVL > 11 mm",
            "Right axis deviation",
            "Right bundle branch block",
            "ST upslopping",
            "T wave inversion (inferior - II, III, aVF)",
            "T wave inversion (septal- V1-V2)",
            "U wave",
            "qRS in V5-V6-I, aVL",
            "rSR in V1-V2"
        ]
    }
}
