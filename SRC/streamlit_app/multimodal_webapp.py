import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import streamlit as st
import altair as alt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)


#you can get cool emojis from https://emojipedia.org/

def evaluate_model_on_preds(preds: np.ndarray, y_true: pd.Series) -> dict:
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if preds.shape[1] > 1:
        preds = preds[:, 1]

    auc = roc_auc_score(y_true, preds)
    accuracy = accuracy_score(y_true, np.round(preds))
    f1 = f1_score(y_true, np.round(preds), average="weighted")

    return {'auc': auc, 'accuracy': accuracy, 'f1': f1}


class Early_or_Single_Model:
    def __init__(self, model: BaseEstimator = None):
        self.model = model if model else LogisticRegression(max_iter=1000)
        self.fitted = False

    def data(self, modality_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
        first_key = next(iter(modality_dfs))
        y = modality_dfs[first_key]['subtype']
        features = []

        for modality, df in modality_dfs.items():
            df = df.copy()
            non_feature_cols = ['submitter_id.samples', 'subtype']
            feature_cols = [col for col in df.columns if col not in non_feature_cols]
            df = df[feature_cols].add_suffix(f'_{modality}')
            features.append(df)

        X = pd.concat(features, axis=1)
        return X, y

    def fit(self, train_modalities: dict[str, pd.DataFrame]):
        logging.info('Fitting early or single model...')
        X_train, y_train = self.data(train_modalities)
        self.model.fit(X_train, y_train)
        self.fitted = True
        return X_train  # For coefficient analysis

    def predict(self, val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X_val, _ = self.data(val_modalities)
        return self.model.predict_proba(X_val)



    def wrapper(self, train_modalities: dict[str, pd.DataFrame],
                val_modalities: dict[str, pd.DataFrame],
                test_modalities: dict[str, pd.DataFrame]
                ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:

        X_train = self.fit(train_modalities)
        preds = self.predict(val_modalities)  # For coefficient analysis
        test_set_preds = self.predict(test_modalities)
        coefs = pd.Series(self.model.coef_[0], index=X_train.columns, name="coefficient").sort_values(key=abs, ascending=False)
        coef_df = coefs.head(500).reset_index()
        coef_df.columns = ['feature', 'coefficient']
        coef_df['modality'] = coef_df['feature'].apply(lambda x: x.split('_')[-1])
        return preds[:, 1], coef_df, test_set_preds
    
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class IntermediateIntegrationModel(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, encoded_dim=64, dropout=0.3):
        super().__init__()
        self.encoders = nn.ModuleList([
            ModalityEncoder(in_dim, hidden_dim, encoded_dim, dropout)
            for in_dim in input_dims
        ])

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(len(input_dims) * encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        feats = [encoder(x) for encoder, x in zip(self.encoders, inputs)]
        combined = torch.cat(feats, dim=-1)
        return self.classifier(combined).squeeze(-1)


class IntermediateIntegrationWrapper:
    def __init__(self, hidden_dim=128, encoded_dim=64, dropout=0.3, epochs=30, lr=1e-3, batch_size=32):
        self.model = None
        self.fitted = False
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.encoded_dim = encoded_dim
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def data(self, modalities: dict[str, pd.DataFrame]):
        """
        Extract features from each modality DataFrame.
        Assumes all share the same labels in a 'subtype' column.
        """
        y = next(iter(modalities.values()))['subtype']
        x_dfs = [
            df.drop(columns=['subtype', 'submitter_id.samples']) for df in modalities.values()
        ]
        return x_dfs, y

    def fit(self, train_modalities: dict[str, pd.DataFrame]):
        x_dfs, y_series = self.data(train_modalities)
        x_tensors = [torch.tensor(x.values, dtype=torch.float32) for x in x_dfs]
        y = torch.tensor(y_series.values, dtype=torch.float32)

        dataset = TensorDataset(*x_tensors, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dims = [x.shape[1] for x in x_tensors]
        self.model = IntermediateIntegrationModel(
            input_dims=input_dims,
            hidden_dim=self.hidden_dim,
            encoded_dim=self.encoded_dim,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for *batch_xs, batch_y in loader:
                batch_xs = [x.to(self.device) for x in batch_xs]
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_xs)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.fitted = True

    def predict(self, val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        x_dfs, _ = self.data(val_modalities)
        x_tensors = [torch.tensor(x.values, dtype=torch.float32).to(self.device) for x in x_dfs]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensors)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def wrapper(self, train_modalities: dict[str, pd.DataFrame], val_modalities: dict[str, pd.DataFrame], test_modalities: dict[str, pd.DataFrame]) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        X_train = self.fit(train_modalities)
        preds = self.predict(val_modalities)  # For coefficient analysis
        test_set_preds = self.predict(test_modalities)
        coef_df = None
        return preds, coef_df, test_set_preds
    
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

    def wrapper(self, train_modalities: dict[str, pd.DataFrame],
            val_modalities: dict[str, pd.DataFrame],
            test_modalities: dict[str, pd.DataFrame]
           ) -> tuple[np.ndarray, dict, np.ndarray]:

        self.fit(train_modalities)
        preds = self.predict(val_modalities)
        test_set_preds = self.predict(test_modalities)

        # Coefficient extraction from each modality-specific model
        coef_dict = {}
        for modality, model in self.models.items():
            if hasattr(model, "coef_"):
                X, _ = self.data(train_modalities[modality])
                coefs = pd.Series(model.coef_[0], index=X.columns, name="coefficient")
                df = coefs.abs().sort_values(ascending=False).head(500).reset_index()
                df.columns = ['feature', 'coefficient']
                df['modality'] = modality
                coef_dict[modality] = df

        return preds, coef_dict, test_set_preds


# --- Streamlit Interface with Tabs ---
st.set_page_config(page_title="Multimodal Model App", layout="wide")
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo_updated.png", width=220)

with col2:
    st.markdown("""<div style="height: 220px; display: flex; align-items: center;"><h1 style="margin: 0;">BreaSight: A Deep Dive into Lobular & Ductal Cancer</h1></div>""",unsafe_allow_html=True,)

# === CACHE DATA LOADER ===
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# === TABS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '🧬 Project Page', "📈 Exploratory Data Analysis", "📊 Model Training", "🧫 Histology", "🚑 Patient Inference"
])

# === TAB 1: Project Overview ===
with tab1:
    st.title("🔬 Welcome to the Multimodal Breast Cancer Classifier App!")
    st.write("""
        The aim of this Hackathon project was to build a Machine Learning classifier to distinguish between two subtypes of Breast Cancer: **Lobular** and **Ductal**.  

        **🩸 Invasive Ductal Carcinoma (IDC)** accounts for about 80% of all breast cancer cases.  
        **🧬 Invasive Lobular Carcinoma (ILC)** accounts for about 15%.  

        Accurate classification is crucial because they require different treatments.
    """)
    st.markdown("---")

    st.subheader("💡 Integration Methods")
    st.markdown("""
    - **Early Integration**: Concatenate all modality datasets and train the model.  
    - **Intermediate Integration**: Transform modalities to a common space, merge, and train.  
    - **Late Integration**: Train separate models and ensemble their outputs.
    """)
    st.markdown("---")

    st.subheader("📊 Model Evaluation Metrics")
    st.markdown("""
    1. **ROC-AUC Curve**  
    2. **Accuracy**  
    3. **F1 Score**  
    4. **Weighted F1 Score** *(to address class imbalance)*
    """)
    st.markdown("---")

    st.subheader("🖼️ Histology Image Preprocessing")
    st.write("""
        Histology H&E images were pre-processed using **[CLAM](https://github.com/mahmoodlab/CLAM)**.  
        This boosted model performance from F1: **0.717 ➔ 0.825**
    """)
    st.markdown("---")

    st.subheader("📂 Data Source")
    st.write("""
        All data is from **TCGA-BRCA** and split into Lobular and Ductal.
    """)
    st.markdown("---")

    st.subheader("🚀 Project Workflow")
    st.write("""
        - Experimented with architectures, integration strategies, and preprocessing.  
        - Best-performing pipelines integrated into this app.
    """)
    st.markdown("---")

    st.subheader("✨ App Features")
    st.markdown("""
    - **🔎 EDA**: Visualize modality data.  
    - **🛠️ Model Training**: Select modalities & train.  
    - **🖼️ Histology Viewer**: Upload and inspect slides.
    """)
    st.info("✨ Powered by [Streamlit](https://streamlit.io/) — use the tabs above to explore!")

# === TAB 2: EDA ===
with tab2:
    st.subheader("Exploratory Data Analysis")

    modality_fullnames = {
        'rna': 'Gene Expression (RNA-Seq)',
        'mir': 'MicroRNA Expression',
        'cnv': 'Copy Number Variation',
        'meth': 'DNA Methylation',
        'hist': 'Histology Image',
        'mutation': 'Somatic Mutation'
    }

    selected_modality_temp = st.selectbox("Select a modality to explore", modality_fullnames.values())
    selected_modality = [k for k, v in modality_fullnames.items() if v == selected_modality_temp][0]

    train_data = load_csv(f"../../processed_data/train_{selected_modality}.csv")
    val_data = load_csv(f"../../processed_data/val_{selected_modality}.csv")
    test_data = load_csv(f"../../processed_data/test_{selected_modality}.csv")

    for df in [train_data, val_data, test_data]:
        if 'submitter_id.samples' in df.columns:
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index('submitter_id.samples')))
            df = df[cols]

    y, y_val, y_test = train_data['subtype'], val_data['subtype'], test_data['subtype']
    class_labels = {0: "ductal", 1: "lobular"}

    subtype_counts = pd.DataFrame({
        'Train Samples': y.value_counts().sort_index(),
        'Validation Samples': y_val.value_counts().sort_index(),
        'Test Samples': y_test.value_counts().sort_index()
    }).fillna(0).astype(int)
    subtype_counts.index = subtype_counts.index.map(class_labels)

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Samples per Subtype")
        st.dataframe(subtype_counts, use_container_width=True)

    for df in [train_data, val_data, test_data]:
        df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

    st.subheader(f"{modality_fullnames[selected_modality]} Train Data")
    st.write(train_data.iloc[:, :50])
    st.write("Train shape:", train_data.shape)
    st.write("Val shape:", val_data.shape)
    st.write("Test shape:", test_data.shape)

    include_val = st.checkbox("Include validation data in UMAP", value=False)
    st.write("UMAP projection for clustering visualization.")

    umap_path = f"./umaps/{selected_modality}_umap_with_val.png" if include_val else f"./umaps/{selected_modality}_umap.png"
    try:
        st.image(umap_path, use_container_width=False, width=700)
    except Exception:
        st.warning("Sorry! UMAP visualization not available.")

# === TAB 3: Model Training ===
with tab3:
    st.subheader("📊 Model Training")

    train_paths = {
        'rna': "../../processed_data/train_rna.csv",
        'mirna': "../../processed_data/train_mir.csv",
        'cnv': "../../processed_data/train_cnv.csv",
        'meth': "../../processed_data/train_meth.csv",
        'histology': "../../processed_data/train_hist.csv",
        'mutation': "../../processed_data/train_mutation.csv"
    }

    val_paths = {k: v.replace('train', 'val') for k, v in train_paths.items()}
    test_paths = {k: v.replace('train', 'test') for k, v in train_paths.items()}

    selected_modalities = st.multiselect("Select modalities", list(train_paths.keys()),
                                         default=['mirna', 'meth', 'cnv'])

    # Session state setup
    for key in ['early_coef_df', 'early_performance', 'early_test_performance',
                'intermediate_coef_df', 'intermediate_performance', 'intermediate_test_performance',
                'late_coef_dict', 'late_performance', 'late_test_performance']:
        if key not in st.session_state:
            st.session_state[key] = None
    selected_modalities_str = ', '.join(selected_modalities).upper() if selected_modalities else "None"
    if st.button(f'Train and Predict on: {selected_modalities_str}'):
        if not selected_modalities:
            st.error("Please select at least one modality.")
            st.stop()

        with st.spinner("Loading data..."):
            train_modalities = {k: load_csv(train_paths[k]) for k in selected_modalities}
            val_modalities = {k: load_csv(val_paths[k]) for k in selected_modalities}
            test_modalities = {k: load_csv(test_paths[k]) for k in selected_modalities}

        with st.spinner("Training Early Model..."):
            early_model = Early_or_Single_Model()
            early_preds, early_coef_df, early_test_preds = early_model.wrapper(train_modalities, val_modalities,
                                                                               test_modalities)
            st.session_state.early_coef_df = early_coef_df
            st.session_state.early_performance = evaluate_model_on_preds(early_preds,
                                                                         val_modalities[selected_modalities[0]][
                                                                             'subtype'])
            st.session_state.early_test_performance = evaluate_model_on_preds(early_test_preds,
                                                                              test_modalities[selected_modalities[0]][
                                                                                  'subtype'])

        with st.spinner("Training Intermediate Model..."):
            intermediate_model = IntermediateIntegrationWrapper()
            int_preds, int_coef_df, int_test_preds = intermediate_model.wrapper(train_modalities, val_modalities,
                                                                                test_modalities)
            st.session_state.intermediate_coef_df = int_coef_df
            st.session_state.intermediate_performance = evaluate_model_on_preds(int_preds,
                                                                                val_modalities[selected_modalities[0]][
                                                                                    'subtype'])
            st.session_state.intermediate_test_performance = evaluate_model_on_preds(int_test_preds, test_modalities[
                selected_modalities[0]]['subtype'])

        with st.spinner("Training Late Model..."):
            late_model = LateIntegrationModel()
            late_preds, late_coef_dict, late_test_preds = late_model.wrapper(train_modalities, val_modalities,
                                                                             test_modalities)
            st.session_state.late_coef_dict = late_coef_dict
            st.session_state.late_performance = evaluate_model_on_preds(late_preds,
                                                                        val_modalities[selected_modalities[0]][
                                                                            'subtype'])
            st.session_state.late_test_performance = evaluate_model_on_preds(late_test_preds,
                                                                             test_modalities[selected_modalities[0]][
                                                                                 'subtype'])

    # Visualization of Coefficients
    if st.session_state.early_coef_df is not None:
        num_features = st.slider("Select number of features to display", min_value=10, max_value=1000, value=200,
                                 step=10)

        st.subheader("Early Model Coefficients")
        st.write('F1 Score (val):', st.session_state.early_performance['f1'])
        chart_df = st.session_state.early_coef_df.head(num_features)
        st.altair_chart(
            alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("coefficient:Q"),
                y=alt.Y("feature:N", sort='-x'),
                color="modality:N",
                tooltip=["feature", "coefficient", "modality"]
            ).properties(height=600),
            use_container_width=True
        )

        st.subheader("Intermediate Model Performance")
        st.write('F1 Score (val):', st.session_state.intermediate_performance['f1'])

        st.subheader("Late Model Coefficients")
        st.write('F1 Score (val):', st.session_state.late_performance['f1'])

        for modality, df in st.session_state.late_coef_dict.items():
            st.markdown(f"**{modality.upper()}**")
            top_df = df.head(num_features)
            st.altair_chart(
                alt.Chart(top_df).mark_bar().encode(
                    x="coefficient:Q",
                    y=alt.Y("feature:N", sort='-x'),
                    tooltip=["feature", "coefficient", "modality"]
                ).properties(height=500, title=f"Top {num_features} for {modality.upper()}"),
                use_container_width=True
            )

# === TAB 4: Histology Viewer ===
with tab4:
    st.subheader("🧫 Histology Image Viewer")
    uploaded_image = st.file_uploader("Upload a Histology Image (PNG)", type=["png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Slide", use_container_width=True)
    else:
        st.info("Please upload a .png file.")

# === TAB 5: Patient Inference ===
with tab5:
    st.subheader("🧬 Patient Inference")
    patient_ids = [f'Patient_{i}' for i in range(1, 31)]
    probabilities = np.random.beta(0.5, 0.5, len(patient_ids))

    selected_patient = st.selectbox("🔍 Select a Patient", patient_ids)
    prob = probabilities[patient_ids.index(selected_patient)]
    label = "Ductal" if prob > 0.5 else "Lobular"
    confidence = "High" if prob < 0.2 or prob > 0.8 else "Low"

    st.subheader(f"Results for {selected_patient}")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧪 Probability", f"{prob:.2f}")
    col2.metric("🧬 Predicted Class", label)
    col3.metric("🎯 Confidence", confidence)

    if confidence == "High":
        st.success("✅ High confidence")
    else:
        st.warning("⚠️ Prediction is uncertain")