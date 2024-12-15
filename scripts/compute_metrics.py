# Notebook for computing metrics during training and testing.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score,
                             PrecisionRecallDisplay,
                             precision_recall_curve,
                             matthews_corrcoef,
                             roc_auc_score,
                             f1_score)
from scipy.special import softmax

precision_recall_data = []

def compute_metrics(eval_pred, model_name, training_mode=True, decision_threshold=None):
    """
    Evaluate a range of classification metrics for sequence and token-level predictions.

    This function computes Accuracy, F1 Score, Average Precision (AP), Matthews Correlation Coefficient (MCC), 
    and Area Under the Receiver Operating Characteristic Curve (ROC AUC). 

    - Accuracy is calculated at the sequence level.
    - F1, MCC, AP, and ROC AUC are calculated at the token (nucleotide) level.

    Average Precision is interested as it considers non-trivial decision boundaries. During training, 
    the decision threshold is dynamically computed to maximize the F1 score. For evaluation, the decision threshold 
    must be provided.

    Args:
        eval_pred (tuple): A tuple (logits, labels) where logits are the raw model outputs 
             and labels are the true values. Padding tokens in labels should be set to -100.
        model_name (str): A name or identifier for the model. Used for saving output files during evaluation.
        training_mode (bool): If True, script will compute the optimal decision threshold to maximize F1 score. 
            If False, a fixed decision threshold must be provided.
        decision_threshold (float, optional): The decision threshold to classify predictions as positive (1) or negative (0). 
            Required if training_mode is False. 

    Returns:
        dict: A dictionary containing the calculated metrics:
            - "F1": F1 score at the token level.
            - "seq_accuracy": Accuracy at the sequence level.
            - "AP": Average Precision score.
            - "MCC": Matthews Correlation Coefficient.
            - "AUC": Area Under the ROC Curve.
            - "ideal_threshold" (optional): The decision threshold that maximized F1 during training. Will only be returned in training mode.
    """
    if not training_mode:
        assert decision_threshold is not None, "decision_threshold must not be None when training_mode is False"

    logits, labels = eval_pred

    # Find predictions from logits
    predictions = softmax(logits, axis=2)[:,:,1] # Probability of positive label

    # Reshape predictions and labels into long strings to compute metrics per token
    predictions_flat = predictions.reshape((-1,))
    labels_flat = labels.reshape((-1,))

    # Remove all the padded ones
    predictions_flat = predictions_flat[labels_flat!=-100]
    labels_flat_cleaned = labels_flat[labels_flat!=-100]

    # Compute average precision
    AP = average_precision_score(labels_flat_cleaned, predictions_flat)

    # Plot precision curves
    precision, recall, thresholds = precision_recall_curve(labels_flat_cleaned, predictions_flat)

    if training_mode: 
        # Decision threshold not passed in, compute the ideal boundary and optimized F1
        decision_threshold = thresholds[np.nanargmax(2 * (precision * recall) / (precision + recall))]
    else: 
        # Precision-recall text file used to create plot in paper
        # Don't plot this during training
        np.savetxt(f"pr_curve_test_{model_name}.txt", np.vstack((precision,recall)).T)

    # Calculate accuracy
    categorical_predictions = np.where(predictions>decision_threshold, 1, 0)
    accuracy = compute_accuracy(labels, categorical_predictions)

    categorical_predictions_flat = categorical_predictions.reshape((-1,))
    categorical_predictions_flat = categorical_predictions_flat[labels_flat!=-100]

    F1 = f1_score(categorical_predictions_flat, labels_flat_cleaned)
    MCC = matthews_corrcoef(labels_flat_cleaned, categorical_predictions_flat)
    AUC = roc_auc_score(labels_flat_cleaned, predictions_flat)

    # Combine metrics
    metrics = {"F1": F1, "seq_accuracy": accuracy, "AP": AP, "MCC": MCC, "AUC": AUC}

    if training_mode:
        precision_recall_data.append(list(zip(precision, recall)))
        metrics['ideal_threshold'] = decision_threshold

    return metrics

def compute_accuracy(labels, categorical_predictions):
    """
    Compute sequence-level accuracy by comparing predicted and true sequences.

    Args:
        labels: Ground-truth labels
            Padding tokens should be set to -100.
        categorical_predictions: Binary predictions

    Returns:
        float: Sequence-level accuracy, calculated as the ratio of correctly predicted sequences 
        to the total number of sequences.
    """
    sequence_matches = 0
    total_sequences = 0

    for label, cat_preds in zip(labels, categorical_predictions):
        # Ignore padded tokens
        cat_preds = cat_preds[label != -100]
        label = label[label != -100]

        assert len(cat_preds) == len(label)

        # Sequence-level accuracy
        if np.array_equal(label, cat_preds):  # Entire sequence matches
            sequence_matches += 1
        total_sequences += 1

    # Sequence-level accuracy
    sequence_accuracy = sequence_matches / total_sequences if total_sequences > 0 else 0
    return sequence_accuracy