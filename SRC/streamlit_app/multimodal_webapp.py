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

    def wrapper(self, train_modalities: dict[str, pd.DataFrame], val_modalities: dict[str, pd.DataFrame], test_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        self.fit(train_modalities)
        return self.predict(val_modalities)
    
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
           ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:

        self.fit(train_modalities)
        preds = self.predict(val_modalities)
        test_set_preds = self.predict(test_modalities)

        # Coefficient extraction from each modality-specific model
        coef_dfs = []
        for modality, model in self.models.items():
            if hasattr(model, "coef_"):
                X, _ = self.data(train_modalities[modality])
                coefs = pd.Series(model.coef_[0], index=X.columns, name="coefficient")
                df = coefs.abs().sort_values(ascending=False).head(500).reset_index()
                df.columns = ['feature', 'coefficient']
                df['modality'] = modality
                coef_dfs.append(df)

        coef_df = pd.concat(coef_dfs, ignore_index=True) if coef_dfs else pd.DataFrame(columns=['feature', 'coefficient', 'modality'])

        return preds, coef_df, test_set_preds


# --- Streamlit Interface with Tabs ---

st.set_page_config(page_title="Multimodal Model App", layout="wide")
st.title('🧬 My Cool Multimodal Webapp')

tab1, tab2, tab3, tab4 = st.tabs(["📈 Exploratory Data Analysis", "📊 Model Training", "🧫 Histology", "Patient Inference"])

with tab1:
    # read in meth train
    st.subheader("Exploratory Data Analysis")
    st.write("This tab is for exploratory data analysis. You can visualize and explore the data here.")

    # Example EDA: read in meth train, ignore col 1
    meth_train = pd.read_csv("../../processed_data/train_meth.csv", index_col=0)

    #plot first to cols in scatter coloured by subtype
    st.subheader("Meth Train Data")

    st.write(meth_train.head())
    st.write("Shape of Meth Train Data:", meth_train.shape)

    st.write("Columns in Meth Train Data:", meth_train.columns.tolist())

    # Plot first two columns in scatter plot coloured by subtype
    if 'submitter_id.samples' in meth_train.columns:
        meth_train = meth_train.drop(columns=['submitter_id.samples'])
    if 'subtype' in meth_train.columns:
        # do matplotlib to make a scatter plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(meth_train.iloc[:, 0], meth_train.iloc[:, 1], c=meth_train['subtype'].astype('category').cat.codes, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Subtype')
        plt.xlabel(meth_train.columns[0])
        plt.ylabel(meth_train.columns[1])
        plt.title('Scatter Plot of First Two Features in Meth Train Data')
        st.pyplot(plt)

        #also show it in altair!
        scatter = alt.Chart(meth_train).mark_circle(size=60).encode(
            x=alt.X(meth_train.columns[0], title=meth_train.columns[0]),
            y=alt.Y(meth_train.columns[1], title=meth_train.columns[1]),
            color=alt.Color('subtype:N', title='Subtype'),
            tooltip=[meth_train.columns[0], meth_train.columns[1], 'subtype']
        ).properties(
            width=800,
            height=400
        )
        st.altair_chart(scatter, use_container_width=True)

# --- Tab 1: Model Training ---
with tab2:
    possible_train_modalities = {
        'rna': "../../processed_data/train_rna.csv",
        'mirna': "../../processed_data/train_mir.csv",
        'cnv': "../../processed_data/train_cnv.csv",
        'meth': "../../processed_data/train_meth.csv",
        'histology': "../../processed_data/train_hist.csv",
        'mutation': "../../processed_data/train_mutation.csv"
    }

    possible_val_modalities = {
        'rna': "../../processed_data/val_rna.csv",
        'mirna': "../../processed_data/val_mir.csv",
        'cnv': "../../processed_data/val_cnv.csv",
        'meth': "../../processed_data/val_meth.csv",
        'histology': "../../processed_data/val_hist.csv",
        'mutation': "../../processed_data/val_mutation.csv"
    }

    possible_test_modalities = {
        'rna': "../../processed_data/test_rna.csv",
        'mirna': "../../processed_data/test_mir.csv",
        'cnv': "../../processed_data/test_cnv.csv",
        'meth': "../../processed_data/test_meth.csv",
        'histology': "../../processed_data/test_hist.csv",
        'mutation': "../../processed_data/test_mutation.csv"
    }

    selected_modalities = st.multiselect(
        'Select modalities to use for training and validation',
        options=list(possible_train_modalities.keys()),
        default=['mirna', 'meth', 'cnv']
    )

    # Initialize session state for both models
    if 'early_coef_df' not in st.session_state:
        st.session_state.early_coef_df = None
        
    if 'early_performance' not in st.session_state:
        st.session_state.early_performance = None
        
    if 'late_coef_df' not in st.session_state:
        st.session_state.late_coef_df = None
        
    if 'late_performance' not in st.session_state:
        st.session_state.late_performance = None
        
    if 'early_test_performance' not in st.session_state:
        st.session_state.early_test_performance = None
        
    if 'late_test_performance' not in st.session_state:
        st.session_state.late_test_performance = None

    if st.button(f'Train and Predict on {selected_modalities}'):
        with st.spinner('Loading data...'):
            if not selected_modalities:
                st.error("Please select at least one modality.")
                st.stop()
            train_modalities = {k: pd.read_csv(possible_train_modalities[k]) for k in selected_modalities}
            val_modalities = {k: pd.read_csv(possible_val_modalities[k]) for k in selected_modalities}
            test_modalities = {k: pd.read_csv(possible_test_modalities[k]) for k in selected_modalities}

        # Train Early/Single Model
        with st.spinner('Training Early/Single Integration model...'):
            early_model = Early_or_Single_Model()
            early_predictions, early_coef_df, early_test_predictions = early_model.wrapper(train_modalities, val_modalities, test_modalities)
            early_val_performance = evaluate_model_on_preds(early_predictions, val_modalities[selected_modalities[0]]['subtype'])
            early_test_performance = evaluate_model_on_preds(early_test_predictions, test_modalities[selected_modalities[0]]['subtype'])
            
            st.success('Early/Single Integration model trained successfully!')
            

        # Train Late Integration Model
        with st.spinner('Training Late Integration model...'):
            late_model = LateIntegrationModel()
            late_predictions, late_coef_df, late_test_predictions = late_model.wrapper(train_modalities, val_modalities, test_modalities)
            late_val_performance = evaluate_model_on_preds(late_predictions, val_modalities[selected_modalities[0]]['subtype'])
            late_test_performance = evaluate_model_on_preds(late_test_predictions, test_modalities[selected_modalities[0]]['subtype'])
            
            st.success('Late Integration model trained successfully!')
        

        # Store results in session state
        st.session_state.early_coef_df = early_coef_df
        st.session_state.early_performance = early_val_performance
        st.session_state.early_test_performance = early_test_performance
        st.session_state.late_coef_df = late_coef_df
        st.session_state.late_performance = late_val_performance
        st.session_state.late_test_performance = late_test_performance

    # Display coefficient graphs for both models
    if st.session_state.early_coef_df is not None and st.session_state.late_coef_df is not None:
        
        num_features = st.slider("Select number of features to display", min_value=10, max_value=2000, value=200, step=10)

        # Early/Single Integration Model Coefficients
        st.subheader("🔍 Early/Single Integration Model - Important Features by Coefficient")
        st.write('Validation F1 Score:', st.session_state.early_performance['f1'])
        early_display_df = st.session_state.early_coef_df.head(num_features)
        st.write("**Early/Single Integration - Modalities in Data:**", early_display_df['modality'].unique())
        
        early_bar = alt.Chart(early_display_df).mark_bar().encode(
            x=alt.X('coefficient:Q', title='Logistic Regression Coefficient'),
            y=alt.Y('feature:N', sort='-x', title='Feature'),
            color=alt.Color('modality:N', title='Modality'),
            tooltip=['feature', 'coefficient', 'modality']
        ).properties(height=900)
        st.altair_chart(early_bar, use_container_width=True)

        # Late Integration Model Coefficients
        st.subheader("🔍 Late Integration Model - Important Features by Coefficient")
        st.write('Validation F1 Score:', st.session_state.late_performance['f1'])
        late_display_df = st.session_state.late_coef_df.head(num_features)
        st.write("**Late Integration - Modalities in Data:**", late_display_df['modality'].unique())
        
        late_bar = alt.Chart(late_display_df).mark_bar().encode(
            x=alt.X('coefficient:Q', title='Logistic Regression Coefficient'),
            y=alt.Y('feature:N', sort='-x', title='Feature'),
            color=alt.Color('modality:N', title='Modality'),
            tooltip=['feature', 'coefficient', 'modality']
        ).properties(height=900)
        st.altair_chart(late_bar, use_container_width=True)

# --- Tab 2: Histology Viewer ---
with tab3:
    st.subheader("🧫 Histology Image Viewer")
    uploaded_image = st.file_uploader("Upload a Histology Image (PNG)", type=["png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Histology Slide", use_column_width=True)
    else:
        st.info("Please upload a .png file to view the histology image.")

with tab4:
    patient_ids = [f'Patient_{i}' for i in range(1, 31)]
    probabilities = np.random.beta(a=0.5, b=0.5, size=len(patient_ids))

    # Dropdown to select a patient
    selected_patient = st.selectbox("🔍 Select a Patient", patient_ids)
    selected_probability = probabilities[patient_ids.index(selected_patient)]
    patient_class = "Ductal" if selected_probability > 0.5 else "Lobular"
    confidence = "High" if selected_probability < 0.2 or selected_probability > 0.8 else "Low"
    
    # Title
    st.subheader(f"🧬 Results for {selected_patient}")

    # Layout in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("🧪 Predicted Probability", f"{selected_probability:.2f}")
    col2.metric("🧬 Predicted Class", patient_class)
    col3.metric("🎯 Confidence", confidence)

    # Optional extra formatting
    if confidence == "High":
        st.success("✅ High confidence in prediction")
    else:
        st.warning("⚠️ Prediction is uncertain (low confidence)")


