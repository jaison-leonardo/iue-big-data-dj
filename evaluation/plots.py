import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        ax=ax,
        cmap="Blues",
        xticks_rotation=45
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_regression(y_true, y_pred, save_path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Real redshift")
    plt.ylabel("Predicted redshift")
    plt.title("Regression: Real vs Predicted")
    plt.savefig(save_path)
    plt.close()

def plot_clustering(X, labels, title, save_path):
    # Usamos dos dimensiones derivadas para visualizar
    # g-r vs u-g (aunque no las uses como features del modelo)
    u_g = X[:, 0] - X[:, 1]
    g_r = X[:, 1] - X[:, 2]

    plt.figure()
    scatter = plt.scatter(g_r, u_g, c=labels)
    plt.xlabel("g - r")
    plt.ylabel("u - g")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()