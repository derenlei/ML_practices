import torch

def compute_pr_auc_no_loop_no_trapz(y_true, y_pred):
    # Sort predictions and corresponding labels in descending order
    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true = y_true[sorted_indices]

    # Compute cumulative true positives (TP) and false positives (FP)
    # y_true (sorted): [1, 1, 1, 0, 0, 1, 0]
    # tp: [1, 2, 3, 3, 3, 4, 4]  # Cumulative true positives
    # fp: [0, 0, 0, 1, 2, 2, 3]  # Cumulative false positives
    tp = torch.cumsum(y_true, dim=0)  # Cumulative sum of true labels
    fp = torch.cumsum(1 - y_true, dim=0)  # Cumulative sum of false labels

    # Total number of positives in the dataset
    total_positives = y_true.sum()

    # Compute precision and recall
    precision = tp / (tp + fp)
    recall = tp / total_positives

    # Add a starting point (0, 1) to close the curve
    recall = torch.cat([torch.tensor([0.0], device=recall.device), recall])
    precision = torch.cat([torch.tensor([1.0], device=precision.device), precision])

    # Compute area under the curve manually using the trapezoidal rule
    # Area for each segment: (recall[i+1] - recall[i]) * (precision[i] + precision[i+1]) / 2
    delta_recall = recall[1:] - recall[:-1]  # Differences in recall
    pr_auc = torch.sum(delta_recall * (precision[:-1] + precision[1:]) / 2)

    return pr_auc

# Example usage
y_pred = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1])
y_true = torch.tensor([1, 1, 1, 0, 0, 1, 0])

pr_auc = compute_pr_auc_no_loop_no_trapz(y_true, y_pred)
print(f"PR-AUC: {pr_auc}")



def pr_auc_multiclass_ovr(y_pred, y_true):
    """
    Multi-class PR AUC by One-vs-Rest.
    y_pred: [N, C], raw scores or probabilities for each class
    y_true: [N] with integer class labels in {0..C-1}
    Returns: Macro-average PR AUC over all classes
    """
    n_classes = y_pred.shape[1]
    pr_aucs = []

    for c in range(n_classes):
        # Convert labels to binary: 1 if class == c, else 0
        y_true_c = (y_true == c).float()
        # Scores for the c-th class
        y_pred_c = y_pred[:, c]
        # Compute binary PR AUC
        auc_c = pr_auc_binary(y_pred_c, y_true_c)
        pr_aucs.append(auc_c)

    # Macro-average
    return torch.mean(torch.stack(pr_aucs))


def pr_auc_multilabel(y_pred, y_true):
    """
    y_pred: [N, C]
    y_true: [N, C] in {0,1}
    Return macro-average PR AUC across labels
    """
    assert y_pred.shape == y_true.shape
    n_labels = y_pred.shape[1]
    pr_aucs = []

    for c in range(n_labels):
        y_pred_c = y_pred[:, c]
        y_true_c = y_true[:, c]
        pr_aucs.append(pr_auc_binary(y_pred_c, y_true_c))  # same binary function

    return torch.mean(torch.stack(pr_aucs))
