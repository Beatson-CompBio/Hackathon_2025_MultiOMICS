import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import streamlit as st

logging.basicConfig(level=logging.INFO)

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import streamlit as st

logging.basicConfig(level=logging.INFO)

# --- Utility Functions and Classes ---

def evaluate_model_on_preds(preds: np.ndarray, y_true: pd.Series) -> dict:
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if preds.shape[1] > 1:
        preds = preds[:, 1]

    auc = roc_auc_score(y_true, preds)
    accuracy = accuracy_score(y_true, np.round(preds))
    f1 = f1_score(y_true, np.round(preds))

    return {'auc': auc, 'accuracy': accuracy, 'f1': f1}

class Early_or_Single_Model:
    def __init__(self, model: BaseEstimator = None):
        self.model = model if model else LogisticRegression(max_iter=1000)
        self.fitted = False

    def data(self, modality_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
        first_key = next(iter(modality_dfs))
        y = modality_dfs[first_key]['subtype']
        features = [df.drop(columns=['subtype', 'submitter_id.samples']) for df in modality_dfs.values()]
        X = pd.concat(features, axis=1)
        assert 'submitter_id.samples' not in X.columns
        assert 'subtype' not in X.columns
        return X, y

    def fit(self, train_modalities: dict[str, pd.DataFrame]):
        logging.info('Fitting early or single model...')
        X_train, y_train = self.data(train_modalities)
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict(self, val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X_val, _ = self.data(val_modalities)
        return self.model.predict_proba(X_val)

    def wrapper(self, train_modalities: dict[str, pd.DataFrame], val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        self.fit(train_modalities)
        preds = self.predict(val_modalities)
        return preds[:, 1]

# --- Streamlit Interface with Tabs ---

st.set_page_config(page_title="Multimodal Model App", layout="wide")
st.title('🧬 My Cool Multimodal Webapp')

tab1, tab2 = st.tabs(["📊 Model Training", "🧫 Histology"])

# --- Tab 1: Model Training ---
with tab1:
    possible_train_modalities = {
        'rna': "../../processed_data/train_rna.csv",
        'mirna': "../../processed_data/train_mir.csv",
        'cnv': "../../processed_data/train_cnv.csv",
        'meth': "../../processed_data/train_meth.csv"
    }

    possible_val_modalities = {
        'rna': "../../processed_data/val_rna.csv",
        'mirna': "../../processed_data/val_mir.csv",
        'cnv': "../../processed_data/val_cnv.csv",
        'meth': "../../processed_data/val_meth.csv"
    }

    selected_modalities = st.multiselect(
        'Select modalities to use for training and validation',
        options=list(possible_train_modalities.keys()),
        default=list(possible_train_modalities.keys())
    )

    if st.button(f'Train and Predict on {selected_modalities}'):
        with st.spinner('Loading data...'):
            if not selected_modalities:
                st.error("Please select at least one modality.")
                st.stop()
            train_modalities = {k: pd.read_csv(possible_train_modalities[k]) for k in selected_modalities}
            val_modalities = {k: pd.read_csv(possible_val_modalities[k]) for k in selected_modalities}

        with st.spinner('Training model...'):
            model = Early_or_Single_Model()
            predictions = model.wrapper(train_modalities, val_modalities)
            performance = evaluate_model_on_preds(predictions, val_modalities[selected_modalities[0]]['subtype'])

        st.success('Model trained successfully!')
        st.write('Model Performance:')
        st.bar_chart(performance)


with tab2:
    st.subheader("🧫 Histology Image Viewer")
    uploaded_image = st.file_uploader("Upload a Histology Image (PNG)", type=["png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Histology Slide", use_column_width=True)
    else:
        st.info("Please upload a .png file to view the histology image.")
