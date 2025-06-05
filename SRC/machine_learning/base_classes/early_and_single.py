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

    def data(self, modality_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and labels from one or more modality DataFrames in a dict.
        Assumes 'subtype' column is the label in each DataFrame.
        """
        # Extract label from the first modality (assumes all have the same labels)
        first_key = next(iter(modality_dfs))
        y = modality_dfs[first_key]['subtype']

        # Drop label and ID columns from all modalities and concatenate features
        features = [df.drop(columns=['subtype', 'submitter_id.samples']) for df in modality_dfs.values()]
        X = pd.concat(features, axis=1)

        assert 'submitter_id.samples' not in X.columns, "submitter_id.samples should not be in features."
        assert 'subtype' not in X.columns, "subtype should not be in features."

        return X, y

    def fit(self, train_modalities: dict[str, pd.DataFrame]):
        """
        Fit model using modality DataFrames that contain 'subtype' labels.
        """
        logging.info('Fitting early or single model...')

        X_train, y_train = self.data(train_modalities)
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict(self, val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the fitted model on validation modality DataFrames.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        X_val, _ = self.data(val_modalities)
        return self.model.predict_proba(X_val)

    def wrapper(self, train_modalities: dict[str, pd.DataFrame], val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Full pipeline: train the model and return predictions on validation data.
        """
        self.fit(train_modalities)
        preds = self.predict(val_modalities)
        return preds[:, 1]


if __name__ == "__main__":

    train_modalities = {
        #'rna': pd.read_csv("../../../processed_data/train_rna.csv"),
        #'mirna': pd.read_csv("../../../processed_data/train_mir.csv"),
        #drop first coolumn from meth using iloc
        'cnv': pd.read_csv("../../../processed_data/train_cnv.csv"),
        #'meth': pd.read_csv("../../../processed_data/train_meth.csv")
    }

    val_modalities = {
        #'rna': pd.read_csv("../../../processed_data/val_rna.csv"),
        #'mirna': pd.read_csv("../../../processed_data/val_mir.csv"),
        'cnv': pd.read_csv("../../../processed_data/val_cnv.csv"),
        #'meth': pd.read_csv("../../../processed_data/val_meth.csv")
    }

    model = Early_or_Single_Model()
    predictions = model.wrapper(train_modalities, val_modalities)

    performance = evaluate_model_on_preds(predictions, val_modalities['cnv']['subtype'])

    print(performance)
