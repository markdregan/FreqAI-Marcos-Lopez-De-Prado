# mypy: ignore-errors
# Functions to plot things
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer


def plot_precision_recall_curve(estimator, y_train, X_test, y_test, file_name):
    encoder = LabelBinarizer()
    encoder.fit(y_train)
    y_test_enc = encoder.transform(y_test)

    y_pred_proba = estimator.predict_proba(X_test)

    precision = dict()
    recall = dict()
    fig, ax = plt.subplots(figsize=(16, 10))
    for i in range(len(encoder.classes_)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_enc[:, i],
                                                            y_pred_proba[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=encoder.classes_[i])

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title(f"precision vs. recall curve: {file_name}")
    plt.savefig(file_name)
    print("file saved")
