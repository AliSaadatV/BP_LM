# Notebook for computing metrics
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

def compute_metrics(eval_pred, filename):
    """
    Function to simultaneously evaluate accuracy, F1 and average precision (AP)

    The function does the evaluation per label for accuracy, and per token otherwise.

    average precision is the most interesting as it accounts for the fact that the
    ideal decision boundary may be something non trivial.

    F1 and accuracy are reported at the decision boundary which maximises the F1
    """
    raw_predictions, labels = eval_pred

    logits = raw_predictions # Discard hidden states and keep logits

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

    # Compute ideal boundary and optimized F1
    ideal_threshold = thresholds[np.nanargmax(2 * (precision * recall) / (precision + recall))]
    F1 = np.nanmax(2 * (precision * recall) / (precision + recall))

    # Calculate accuracy
    categorical_predictions = np.where(predictions>ideal_threshold, 1, 0)
    accuracy = compute_accuracy(labels, categorical_predictions)

    categorical_predictions_flat = categorical_predictions.reshape((-1,))
    categorical_predictions_flat = categorical_predictions_flat[labels_flat!=-100]

    MCC = matthews_corrcoef(labels_flat_cleaned, categorical_predictions_flat)
    AUC = roc_auc_score(labels_flat_cleaned, categorical_predictions_flat)

    metrics = {
        "F1": F1,
        "seq_accuracy": accuracy,
        "AP": AP,
        "MCC": MCC,
        "AUC": AUC,
        "ideal_threshold": ideal_threshold}

    precision_recall_data.append(list(zip(precision, recall)))

    return metrics

def compute_accuracy(labels, categorical_predictions):
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

def compute_metrics_test(eval_pred, filename, decision_threshold):
    """
    Same as above "compute_metrics" function, but without decision boundary optimization.
    """
    logits, labels = eval_pred

    # Find predictions from logits
    predictions = softmax(logits, axis=2)[:,:,1] #probability of positive label

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
    fig, ax = plt.subplots(dpi = 300, figsize = (5,3))
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_title(f"Precision-Recall Curve test set: {filename}", fontsize = 12)
    ax.plot(recall, precision)
    fig.savefig(filename + ".png")

    np.savetxt(f"pr_curve_test_{filename}.txt", np.vstack((precision,recall)).T)

    # Calculate accuracy
    categorical_predictions = np.where(predictions>decision_threshold, 1, 0)
    accuracy = compute_accuracy(labels, categorical_predictions)

    categorical_predictions_flat = categorical_predictions.reshape((-1,))
    categorical_predictions_flat = categorical_predictions_flat[labels_flat!=-100]

    F1 = f1_score(categorical_predictions_flat, labels_flat_cleaned)
    MCC = matthews_corrcoef(labels_flat_cleaned, categorical_predictions_flat)
    AUC = roc_auc_score(labels_flat_cleaned, categorical_predictions_flat)

    # Combine metrics
    dictionary = {"F1": F1, "seq_accuracy": accuracy, "AP": AP, "MCC": MCC, "AUC": AUC, "ideal_threshold": ideal_threshold}

    # Save the performance metrics to a text file
    with open(filename + ".txt", 'w') as f:
      print(dictionary, file=f)

    # Return joint dictionary
    return dictionary