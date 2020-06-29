"""
To evaluate our models before running a test on kaggle
"""
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from matplotlib import pyplot as plt


def spearman_measure(y1, y2):
    return spearmanr(y1, y2)[0]


def evaluate_model(model, dfX_train, y_true, cv=3):
    """Evaluate model by cross validation. Splits train datasets into `cv` folds.
       For each fold, makes predictions by using training on other folds, and returns related score."""
    spearman_scorer = make_scorer(spearman_measure)  # create a scorer object
    return cross_val_score(model, dfX_train, y_true, cv=cv, scoring=spearman_scorer)


def visualize_predictions(models, dfX_train, y_true, cv=3, models_labels=None):
    """Compute and display predictions, using cross validation, from previously fitted models."""

    def norm(y):
        """Normalize a pd.Series"""
        return (y - y.min()) / (y.max() - y.min())

    if not models_labels:
        models_labels = ['model_' + str(i + 1) for i in range(len(models))]
    df_compare = pd.DataFrame([], index=dfX_train.index, columns=models_labels + ['true'])
    for i, model in enumerate(models):
        y_pred = cross_val_predict(model, dfX_train, cv=cv)  # Computes predictions (cross validation)
        df_compare[models_labels[i]] = norm(y_pred)
    df_compare['true'] = norm(y_true)
    df_compare = df_compare.sort_values('true')
    df_compare.index = [i for i in range(len(df_compare.index))]
    ax = df_compare.drop('true', axis='columns').plot(alpha=0.5)
    df_compare['true'].plot(ax=ax, figsize=(20, 10), linewidth=3)
    plt.grid(True)  # Displays grid
    plt.legend()  # Displays legend

