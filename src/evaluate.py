from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

def evaluate_model(true_y, pred_y):
    classes = ["foul", "hard_contact", "no_swing", "swing_and_miss", "weak_contact"]
    print(pred_y[classes].head())
    logloss = log_loss(true_y, pred_y[classes], labels=classes)
    brier = brier_score_loss(true_y, pred_y[classes], labels=classes)
    accuracy = accuracy_score(true_y, pred_y[classes].idxmax(axis=1))
    return logloss, brier, accuracy
