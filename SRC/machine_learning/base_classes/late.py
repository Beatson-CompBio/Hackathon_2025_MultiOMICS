import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)


def evaluate_model_on_preds(preds: np.ndarray, y_true: pd.Series) -> dict:
    """
    Evaluate the model predictions against true labels.
    Returns a dictionary with evaluation metrics: auc, accuracy, and f1 score.
    """
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if preds.shape[1] > 1:
        preds = preds[:, 1]

    auc = roc_auc_score(y_true, preds)
    accuracy = accuracy_score(y_true, np.round(preds))
    f1 = f1_score(y_true, np.round(preds))

    return {'auc': auc, 'accuracy': accuracy, 'f1': f1}


class LateIntegrationModel:
    def __init__(self, model: BaseEstimator = None):
        """
        Initialize the late integration model with one model per modality.
        """
        self.model_constructor = lambda: model if model else LogisticRegression(max_iter=1000)
        self.models = {}  # Dict[str, BaseEstimator]
        self.fitted = False

    def data(self, modality_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and label from a single modality DataFrame.
        """
        y = modality_df['subtype']
        X = modality_df.drop(columns=['subtype', 'submitter_id.samples'])
        return X, y

    def fit(self, train_modalities: dict[str, pd.DataFrame]):
        """
        Fit a model for each modality using its corresponding DataFrame.
        """

        logging.info('Fitting late model...')

        self.models = {}
        for name, df in train_modalities.items():
            X, y = self.data(df)
            model = self.model_constructor()
            model.fit(X, y)
            self.models[name] = model
        self.fitted = True

    def predict(self, val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Predict using each modality-specific model and return the sample-wise max probability.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        preds_dict = {}
        for name, df in val_modalities.items():
            X, _ = self.data(df)
            prob = self.models[name].predict_proba(X)
            preds_dict[name] = prob[:, 1]

        preds_df = pd.DataFrame(preds_dict)
        print(preds_df)
        final_preds = preds_df.max(axis=1).to_numpy()
        return final_preds

    def wrapper(self, train_modalities: dict[str, pd.DataFrame], val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Fit and predict using the late integration strategy.
        """
        self.fit(train_modalities)
        return self.predict(val_modalities)


# --- Example usage ---

if __name__ == "__main__":
    train_modalities = {
        'rna': pd.read_csv("../../../processed_data/train_rna.csv"),
        'mirna': pd.read_csv("../../../processed_data/train_mir.csv"),
        'meth': pd.read_csv("../../../processed_data/train_meth.csv"),
    }

    val_modalities = {
        'rna': pd.read_csv("../../../processed_data/val_rna.csv"),
        'mirna': pd.read_csv("../../../processed_data/val_mir.csv"),
        'meth': pd.read_csv("../../../processed_data/val_meth.csv")
    }

    model = LateIntegrationModel()
    predictions = model.wrapper(train_modalities, val_modalities)

    # Use labels from any modality (they are assumed to be the same)
    performance = evaluate_model_on_preds(predictions, val_modalities['rna']['subtype'])
    print(performance)
