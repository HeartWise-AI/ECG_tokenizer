
standard_lead_order = ["I", 
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
    "RV1 + SV6\xa0> 11 mm": "RV1 + SV6 supérieur à 11 mm",
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

ECG_CATEGORIES = {
    "Rhythm Disorders": [
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
    "Conduction Disorder": [
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
    "Enlargement of the heart chambers": [
        "Bi-atrial enlargement",
        "Left atrial enlargement",
        "Right atrial enlargement",
        "Left ventricular hypertrophy",
        "Right ventricular hypertrophy"
    ],
    "Pericarditis": [
        "Acute pericarditis"
    ],
    "Infarction or ischemia": [
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
    "Other diagnoses": [
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