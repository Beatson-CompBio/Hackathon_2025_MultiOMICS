import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def evaluate_model_on_preds(preds: np.ndarray, y_true: pd.Series) -> dict:
    """
    Evaluate the model predictions against true labels.
    Returns a dictionary with evaluation metrics- auc, accuracy, and f1 score.
    """


    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # Ensure predictions are probabilities
    if preds.shape[1] > 1:
        preds = preds[:, 1]  # Use the positive class probabilities

    auc = roc_auc_score(y_true, preds)
    accuracy = accuracy_score(y_true, np.round(preds))
    f1 = f1_score(y_true, np.round(preds))

    return {
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1
    }

class Early_or_Single_Model:
    def __init__(self, model: BaseEstimator = None):
        """
        Initialize the model with an optional sklearn-style estimator.
        """
        self.model = model if model else LogisticRegression(max_iter=1000)
        self.fitted = False

    def data(self, *modalities: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and labels from one or more modality DataFrames.
        Assumes 'subtype' column is the label in each DataFrame.
        """
        # Extract label from the first modality (assumes all have the same labels)
        y = modalities[0]['subtype']

        # Drop label column from all modalities and concatenate features
        features = [mod.drop(columns=['subtype', 'submitter_id.samples']) for mod in modalities]
        X = pd.concat(features, axis=1)
        assert 'submitter_id.samples' not in X.columns, "submitter_id.samples should not be in features."
        assert 'subtype' not in X.columns, "subtype should not be in features."

        return X, y

    def fit(self, *train_modalities: pd.DataFrame):
        """
        Fit model using modality DataFrames that contain 'subtype' labels.
        """
        X_train, y_train = self.data(*train_modalities)
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict(self, *val_modalities: pd.DataFrame) -> np.ndarray:
        """
        Predict using the fitted model on validation modality DataFrames.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        X_val, _ = self.data(*val_modalities)
        return self.model.predict_proba(X_val)

    def wrapper(self, train_modalities, val_modalities) -> np.ndarray:
        """
        Full pipeline: train the model and return predictions on validation data.
        """
        self.fit(*train_modalities)
        preds = self.predict(*val_modalities)
        return preds[:, 1]


if __name__ == "__main__":

    train_rna = pd.read_csv("../../processed_data/train_rna.csv")
    train_mirna = pd.read_csv("../../processed_data/train_mir.csv")

    val_rna = pd.read_csv("../../processed_data/val_rna.csv")
    val_mirna = pd.read_csv("../../processed_data/val_mir.csv")

    model = Early_or_Single_Model()
    predictions = model.wrapper([train_rna, train_mirna], [val_rna, val_mirna])

    performance = evaluate_model_on_preds(predictions, val_rna['subtype'])

    print(performance)