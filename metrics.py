import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    roc_curve
)

"""
AUthor: Jacques Delfrate
"""

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

def compute_best_threshold(df_gt_col: pd.Series, df_pred_col: pd.Series) -> float:
    """
    Compute the best threshold using the Youden Index.

    Args:
        df_gt_col (pd.Series): Ground truth values for a specific column.
        df_pred_col (pd.Series): Predicted values for a specific column.

    Returns:
        float: The best threshold value.
    """
    fpr, tpr, roc_thresholds = roc_curve(df_gt_col, df_pred_col)
    
    # Compute Youden Index
    youden_index = tpr - fpr
    best_threshold = roc_thresholds[np.argmax(youden_index)]
    
    return float(best_threshold) # convert to float instead of numpy.float32

def compute_metrics_binary(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics for binary ECG classification.

    Args:
        df_gt (pd.DataFrame): Ground truth DataFrame with one column.
        df_pred (pd.DataFrame): Predicted probabilities DataFrame with one column.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """    
    if df_gt.shape[1] != 1 or df_pred.shape[1] != 1:
        raise ValueError("Both df_gt and df_pred must have exactly one column each for binary classification.")

    # Extract the single column
    gt = df_gt.iloc[:, 0]
    pred = df_pred.iloc[:, 0]

    # Initialize metrics dictionary
    metrics = {
        "results": {
            "auc": np.nan,
            "auprc": np.nan,
            "f1": np.nan,
            "threshold": np.nan,
            "prevalence_gt %": np.nan,
            "prevalence_pred %": np.nan
        }
    }

    # Check if there are positive samples in ground truth
    if gt.sum() == 0:
        print(f"Warning: No positive samples in ground truth. Metrics may not be meaningful.")
        return metrics

    try:
        # Compute ROC AUC
        metrics["results"]["auc"] = roc_auc_score(gt, pred)

        # Compute Average Precision (AUPRC)
        metrics["results"]["auprc"] = average_precision_score(gt, pred)

        # Compute Best Threshold
        best_threshold = compute_best_threshold(gt, pred)
        metrics["results"]["threshold"] = best_threshold

        # Compute F1 Score
        predictions_binary = (pred >= best_threshold).astype(int)
        metrics["results"]["f1"] = f1_score(gt, predictions_binary)

        # Compute Prevalence in Ground Truth
        metrics["results"]["prevalence_gt %"] = (gt.sum() / len(gt)) * 100

        # Compute Prevalence in Predictions
        metrics["results"]["prevalence_pred %"] = (predictions_binary.sum() / len(pred)) * 100

    except Exception as e:
        print(f"An error occurred while computing metrics for binary classification: {e}")
    print(metrics)
    return metrics

def compute_metrics(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics for ECG classification.

    Args:
        df_gt (pd.DataFrame): Ground truth DataFrame.
        df_pred (pd.DataFrame): Predicted probabilities DataFrame.

    Returns:
        dict: A dictionary containing evaluation metrics for each category and column.
    """    
    # initialize metrics dictionary
    metrics = {}
    for cat in ECG_CATEGORIES:
        metrics[cat] = {
            "macro_auc": np.nan,
            "macro_auprc": np.nan,
            "macro_f1": np.nan,
            "micro_auc": np.nan,
            "micro_auprc": np.nan,
            "micro_f1": np.nan,
            "threshold": np.nan,
            "prevalence_gt %": np.nan,
            "prevalence_pred %": np.nan
        }
    for col in df_gt.columns:
        metrics[col] = {
            "auc": np.nan,
            "auprc": np.nan,
            "f1": np.nan,
            "threshold": np.nan,
            "prevalence_gt %": np.nan,
            "prevalence_pred %": np.nan
        }

    # Compute category metrics
    for category in ECG_CATEGORIES:
        # Get category columns
        category_columns = [
            col for col in ECG_CATEGORIES[category]
            if df_gt[col].sum() > 0 # filter out columns with no ground truth
        ]
                        
        # Skip category if no columns have ground truth
        if not category_columns:
            continue
            
        # Aggregate ground truth and predictions for the category
        category_gt = df_gt[category_columns]
        category_pred = df_pred[category_columns]

        # Compute macro auc and auprc metrics
        cat_auc_scores = []
        cat_auprc_scores = []
        macro_f1_scores = []
        for col in category_columns:
            # Compute metrics for each column
            col_auc = roc_auc_score(category_gt[col], category_pred[col])
            col_auprc = average_precision_score(category_gt[col], category_pred[col])
            col_threshold = compute_best_threshold(category_gt[col], category_pred[col])
            col_f1 = f1_score(category_gt[col], category_pred[col] >= col_threshold)
            metrics[col] = {
                "auc": col_auc,
                "auprc": col_auprc,
                "threshold": col_threshold,
                "f1": col_f1,
                "prevalence_gt %": category_gt[col].sum() / len(df_gt) * 100,
                "prevalence_pred %": (category_pred[col] >= col_threshold).sum() / len(df_pred) * 100,
            }
            
            # Append metrics to category metrics
            cat_auc_scores.append(col_auc)
            cat_auprc_scores.append(col_auprc)
            macro_f1_scores.append(col_f1)
            
        # Compute macro metrics
        cat_macro_auc = np.mean(cat_auc_scores)
        cat_macro_auprc = np.mean(cat_auprc_scores)
        cat_macro_f1 = np.mean(macro_f1_scores)
                    
        # Compute micro metrics
        ravel_categories_gt = category_gt.values.ravel()
        ravel_categories_pred = category_pred.values.ravel()
        micro_auc = roc_auc_score(
            ravel_categories_gt, 
            ravel_categories_pred, 
            average='micro'
        )
        micro_auprc = average_precision_score(
            ravel_categories_gt, 
            ravel_categories_pred, 
            average='micro'
        )
        best_micro_threshold = compute_best_threshold(ravel_categories_gt, ravel_categories_pred)
        cat_micro_f1 = f1_score(ravel_categories_gt, ravel_categories_pred >= best_micro_threshold, average='micro')                    
        
        # Compute Category Prevalence
        cat_prevalence_gt = ravel_categories_gt.sum() / len(ravel_categories_gt) * 100
        cat_prevalence_micro = (ravel_categories_pred >= best_micro_threshold).sum() / len(ravel_categories_pred) * 100
                        
        # Store Category Metrics
        metrics[category] = {
            "macro_auc":  cat_macro_auc,
            "macro_auprc": cat_macro_auprc,
            "macro_f1": cat_macro_f1,
            "micro_auc": micro_auc,
            "micro_auprc": micro_auprc,
            "micro_f1": cat_micro_f1,
            "threshold": best_micro_threshold,
            "prevalence_gt %": cat_prevalence_gt,
            "prevalence_pred %": cat_prevalence_micro
        }
        
    return metrics