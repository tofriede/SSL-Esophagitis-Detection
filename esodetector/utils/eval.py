import logging
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score

_logger = logging.getLogger(__name__)


def calculate_auc_roc_scores(
    all_targets: np.ndarray,
    all_probabilities: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """Calculate AUC scores for classification.
    
    Args:
        all_targets (np.ndarray): Array of true labels
        all_probabilities (np.ndarray): Array of predicted probabilities (shape: [n_samples, n_classes])
        num_classes (int): Number of classes
        
    Returns:
        dict: Dictionary containing 'auc', 'auc_macro', and 'auc_weighted' scores
    """
    auc_scores = {
        'auc': 0.0,
        'auc_macro': 0.0,
        'auc_weighted': 0.0
    }
    
    try:
        # For binary classification
        if num_classes == 2:
            # AUC for class 1 (positive class)
            auc = roc_auc_score(all_targets, all_probabilities[:, 1])
            auc_scores['auc'] = auc
            auc_scores['auc_macro'] = auc
            auc_scores['auc_weighted'] = auc
        else:
            # For multi-class classification
            auc_macro = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='weighted')
            auc_scores['auc_macro'] = auc_macro
            auc_scores['auc_weighted'] = auc_weighted
            # Also calculate for binary (one-vs-rest for each class)
            try:
                auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
                auc_scores['auc'] = auc
            except:
                auc_scores['auc'] = auc_macro
    except Exception as e:
        _logger.warning(f'Could not calculate AUC-ROC: {e}')
    
    return auc_scores


def calculate_auc_pr_scores(
    all_targets: np.ndarray,
    all_probabilities: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """Calculate AUC-PR scores for classification.
    
    Args:
        all_targets (np.ndarray): Array of true labels
        all_probabilities (np.ndarray): Array of predicted probabilities (shape: [n_samples, n_classes])
        num_classes (int): Number of classes
        
    Returns:
        dict: Dictionary containing 'auc_pr', 'auc_pr_macro', and 'auc_pr_weighted' scores
    """
    auc_pr_scores = {
        'auc': 0.0,
        'auc_macro': 0.0,
        'auc_weighted': 0.0
    }
    
    try:
        # For binary classification
        if num_classes == 2:
            # AUC-PR for class 1 (positive class)
            auc = average_precision_score(all_targets, all_probabilities[:, 1])
            auc_pr_scores['auc'] = auc
            auc_pr_scores['auc_macro'] = auc
            auc_pr_scores['auc_weighted'] = auc
        else:
            # For multi-class classification
            auc_macro = average_precision_score(all_targets, all_probabilities, average='macro')
            auc_weighted = average_precision_score(all_targets, all_probabilities, average='weighted')
            auc_pr_scores['auc_macro'] = auc_macro
            auc_pr_scores['auc_weighted'] = auc_weighted
            # Also calculate micro average
            try:
                auc = average_precision_score(all_targets, all_probabilities, average='micro')
                auc_pr_scores['auc'] = auc
            except:
                auc_pr_scores['auc'] = auc_macro
    except Exception as e:
        _logger.warning(f'Could not calculate AUC-PR: {e}')
    
    return auc_pr_scores
    

def get_class1_metrics(
    cm: np.ndarray,
    all_targets: np.ndarray
) -> Tuple[float, float, float]:
    """Calculate sensitivity, specificity, and precision from a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        all_targets (np.ndarray): Array of true labels.

    Returns:
        tuple: A tuple containing sensitivity, specificity, and precision for class 1.
    """
   # Calculate sensitivity, specificity, and precision for class 1 if it exists
    sensitivity_class1 = 0.0
    specificity_class1 = 0.0
    precision_class1 = 0.0
    
    if cm.shape[0] > 1 and cm.shape[1] > 1:  # Ensure we have at least 2 classes
        # For class 1: sensitivity = TP / (TP + FN), specificity = TN / (TN + FP), precision = TP / (TP + FP)
        if len(np.unique(all_targets)) > 1:  # Check if class 1 exists in targets
            # True Positives for class 1
            tp_class1 = cm[1, 1] if cm.shape[0] > 1 else 0
            # False Negatives for class 1 (class 1 predicted as other classes)
            fn_class1 = np.sum(cm[1, :]) - tp_class1 if cm.shape[0] > 1 else 0
            # True Negatives for class 1 (all correct predictions that are not class 1)
            tn_class1 = np.sum(np.diag(cm)) - tp_class1
            # False Positives for class 1 (other classes predicted as class 1)
            fp_class1 = np.sum(cm[:, 1]) - tp_class1 if cm.shape[1] > 1 else 0
            
            # Calculate sensitivity (recall) for class 1
            sensitivity_class1 = tp_class1 / (tp_class1 + fn_class1) if (tp_class1 + fn_class1) > 0 else 0.0
            
            # Calculate specificity for class 1
            specificity_class1 = tn_class1 / (tn_class1 + fp_class1) if (tn_class1 + fp_class1) > 0 else 0.0
            
            # Calculate precision for class 1
            precision_class1 = tp_class1 / (tp_class1 + fp_class1) if (tp_class1 + fp_class1) > 0 else 0.0

    return sensitivity_class1, specificity_class1, precision_class1


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """
    Save confusion matrix as a heatmap image.
    
    Args:
        cm: confusion matrix from sklearn
        class_names: list of class names
        save_path: path to save the image
        title: title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, 
                annot=True,           # Show numbers in cells
                fmt='d',              # Integer format
                cmap='Blues',         # Color scheme
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Confusion matrix heatmap saved to: {save_path}")