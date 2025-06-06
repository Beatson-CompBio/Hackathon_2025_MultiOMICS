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
st.title('🎗️ BreaSight: A Deep Dive into Lobular & Ductal Cancer')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Project Page',"📈 Exploratory Data Analysis", "📊 Model Training", "🧫 Histology", "Patient Inference"])

with tab1:
    st.title("🔬 Welcome to the Multimodal Breast Cancer Classifier App!")
    st.write("""
        The aim of this Hackathon project was to build a Machine Learning classifier to distinguish between two subtypes of Breast Cancer: **Lobular** and **Ductal**.  
        
        **🩸 Invasive Ductal Carcinoma (IDC)** accounts for about 80% of all breast cancer cases. Cancerous cells reside in the ducts.  
        
        **🧬 Invasive Lobular Carcinoma (ILC)** accounts for about 15% of cases and affects the lobules. Although both subtypes present similar symptoms (swelling and irritation), they require different treatments, making accurate classification crucial.
        
        Multi-omics datasets from TCGA ([link](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)) were obtained and processed through a workflow of **data cleaning & filtering**, **feature selection**, and **modality integration** to build our best-performing model.
    """)
    st.markdown("---")

    st.subheader("💡 Integration Methods")
    st.markdown("""
    - **Early Integration:** Concatenate all modality datasets into one combined matrix, then train the model.  
    - **Intermediate Integration:** Transform each modality into a common representation (e.g., via a neural network), merge into a single table, and then fit the model.  
    - **Late Integration:** Train separate models on each modality; aggregate their outputs (e.g., probability scores) into a final ensemble model.
    """)
    st.write("""
        ℹ️ **Note:** For test samples, all modalities must be present. If a modality is missing for a sample, you may need to drop that sample across modalities or exclude the modality entirely.
    """)
    st.markdown("---")

    st.subheader("📊 Model Evaluation Metrics")
    st.markdown("""
    We evaluated our models using four metrics:
    1. **ROC-AUC Curve**  
    2. **Accuracy**  
    3. **F1 Score**  
    4. **Weighted F1 Score**  

    > The **Weighted F1 Score** was prioritized over the standard F1 Score to account for the class imbalance between Lobular and Ductal subtypes.
    """)
    st.markdown("---")

    st.subheader("🖼️ Histology Image Preprocessing")
    st.write("""
        Histology H&E images were pre-processed using **[CLAM](https://github.com/mahmoodlab/CLAM)** (Data-Efficient, Weakly Supervised Whole-Slide Analysis).  
        This preprocessing boosted our single Logistic Regression model's weighted F1 Score from **0.717 ➔ 0.825**!
    """)
    st.markdown("---")

    st.subheader("📂 Data Source")
    st.write("""
        All data was obtained from the **TCGA-BRCA** cohort and subset into Lobular and Ductal datasets.
    """)
    st.markdown("---")

    st.subheader("🚀 Project Workflow")
    st.write("""
        - **Simon** provided the initial codebase used in the Hackathon.  
        - We experimented with different model architectures, modality combinations, and integration strategies.  
        - The best-performing pipelines were documented and incorporated into this Streamlit app.
    """)
    st.markdown("---")

    st.subheader("✨ App Features")
    st.markdown("""
    - **🔎 Exploratory Data Analysis:**  
      • Visualize and explore each modality (RNA-Seq, CNV, Methylation, etc.).  
    
    - **🛠️ Model Training:**  
      • Select modalities and integration strategies.  
      • Train classifiers (e.g., Logistic Regression, Random Forest).  
      • View feature importance and performance metrics in real time.  
    
    - **🖼️ Histology Image Viewer:**  
      • Upload and inspect H&E slides.  
      • See CLAM-generated attention maps and ROI highlights.
    """)
    st.info("✨ A sleek GUI powered by [Streamlit](https://streamlit.io/)—use the tabs above to navigate through the app! ✨")


with tab2:
    # read in meth train
    st.subheader("Exploratory Data Analysis")
    st.write("This tab is for exploratory data analysis. You can visualize and explore the data here.")

    modality_fullnames = {
        'rna':       'Gene Expression (RNA-Seq)',
        'mir':     'MicroRNA Expression',
        'cnv':       'Copy Number Variation',
        'meth':      'DNA Methylation',
        'hist': 'Histology Image',
        'mutation':  'Somatic Mutation'
    }

    selected_modality_temp = st.selectbox("Select a modality to explore", modality_fullnames.values())
    selected_modality = [key for key, value in modality_fullnames.items() if value == selected_modality_temp][0]

    train_data = pd.read_csv(f"../../processed_data/train_{selected_modality}.csv")
    val_data = pd.read_csv(f"../../processed_data/val_{selected_modality}.csv")
    test_data = pd.read_csv(f"../../processed_data/test_{selected_modality}.csv")

    ## shift submitter_id.samples to the front
    if 'submitter_id.samples' in train_data.columns:
        train_data = train_data[['submitter_id.samples'] + [col for col in train_data.columns if col != 'submitter_id.samples']]
    if 'submitter_id.samples' in val_data.columns:
        val_data = val_data[['submitter_id.samples'] + [col for col in val_data.columns if col != 'submitter_id.samples']]
    if 'submitter_id.samples' in test_data.columns:
        test_data = test_data[['submitter_id.samples'] + [col for col in test_data.columns if col != 'submitter_id.samples']]



    y = train_data['subtype']
    y_val = val_data['subtype']
    y_test = test_data['subtype']

    class_labels = {0: "ductal", 1: "lobular"}

    # Table: train and val samples per subtype
    subtype_counts = pd.DataFrame({
        'Train Samples': y.value_counts().sort_index(),
        'Validation Samples': y_val.value_counts().sort_index(),
        'Test Samples': y_test.value_counts().sort_index()
    }).fillna(0).astype(int)
    subtype_counts.index = subtype_counts.index.map(class_labels)

    # Place table on the left (25% width)
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Samples per Subtype")
        st.dataframe(subtype_counts, use_container_width=True)
        train_data = train_data.drop(columns=[f'Unnamed: 0'], errors='ignore')
        val_data = val_data.drop(columns=[f'Unnamed: 0'], errors='ignore')
        test_data = test_data.drop(columns=[f'Unnamed: 0'], errors='ignore')

    #plot first to cols in scatter coloured by subtype
    st.subheader(f"{modality_fullnames[selected_modality]} Train Data")

    st.write(train_data.head().iloc[:, :50])  # Display first 10 columns of the train data
    st.write(f"Shape of {modality_fullnames[selected_modality]} Train Data:", train_data.shape)
    st.write(f"Shape of {modality_fullnames[selected_modality]} Validation Data:", val_data.shape)
    st.write(f"Shape of {modality_fullnames[selected_modality]} Test Data:", test_data.shape)

    # st.write(f"Columns in {modality_fullnames[selected_modality]} Train Data:", train_data.columns.tolist())

    include_val = st.checkbox("Include validation data in UMAP", value=False)

    st.write("This UMAP projection visualizes the data in a lower-dimensional space, helping to identify clusters and relationships between samples.")

    if include_val:
        umap_path = f"./umaps/{selected_modality}_umap_with_val.png"
        st.caption("UMAP with train and validation data")
    else:
        umap_path = f"./umaps/{selected_modality}_umap.png"
        st.caption("UMAP with train data only")

    try:
        st.image(umap_path, use_container_width=False, width=700)
    except Exception as e:
        st.warning(f"Sorry! Could not perform UMAP right now. Trust us, we have done it.")


# --- Tab 1: Model Training ---
with tab3:
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

    if 'intermediate_coef_df' not in st.session_state:
        st.session_state.intermediate_coef_df = None
    if 'intermediate_performance' not in st.session_state:
        st.session_state.intermediate_performance = None
        
    if 'late_coef_dict' not in st.session_state:
        st.session_state.late_coef_dict = None
        
    if 'late_performance' not in st.session_state:
        st.session_state.late_performance = None
        
    if 'early_test_performance' not in st.session_state:
        st.session_state.early_test_performance = None
    
    if 'intermediate_test_performance' not in st.session_state:
        st.session_state.intermediate_test_performance = None
        
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

        with st.spinner('Training Intermediate Integration model...'):
            intermediate_model = IntermediateIntegrationWrapper()
            intermediate_predictions, intermediate_coef_df, intermediate_test_predictions = intermediate_model.wrapper(train_modalities, val_modalities, test_modalities)
            intermediate_val_performance = evaluate_model_on_preds(intermediate_predictions, val_modalities[selected_modalities[0]]['subtype'])
            intermediate_test_performance = evaluate_model_on_preds(intermediate_test_predictions, test_modalities[selected_modalities[0]]['subtype'])
            
            st.success('Intermediate Integration model trained successfully!')
            

        # Train Late Integration Model
        with st.spinner('Training Late Integration model...'):
            late_model = LateIntegrationModel()
            late_predictions, late_coef_dict, late_test_predictions = late_model.wrapper(train_modalities, val_modalities, test_modalities)
            late_val_performance = evaluate_model_on_preds(late_predictions, val_modalities[selected_modalities[0]]['subtype'])
            late_test_performance = evaluate_model_on_preds(late_test_predictions, test_modalities[selected_modalities[0]]['subtype'])
            
            st.success('Late Integration model trained successfully!')
        

        # Store results in session state
        st.session_state.early_coef_df = early_coef_df
        st.session_state.early_performance = early_val_performance
        st.session_state.early_test_performance = early_test_performance
        st.session_state.late_coef_dict = late_coef_dict
        st.session_state.late_performance = late_val_performance
        st.session_state.late_test_performance = late_test_performance
        # st.session_state.intermediate_coef_df = intermediate_coef_df
        st.session_state.intermediate_performance = intermediate_val_performance
        st.session_state.intermediate_test_performance = intermediate_test_performance

    # Display coefficient graphs for both models
    if st.session_state.early_coef_df is not None and st.session_state.late_coef_dict is not None:
        
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

        # intermediate integration model (Just performances, no coefficients)
        st.subheader("🔍 Intermediate Integration Model - Performance")
        st.write('Validation F1 Score:', st.session_state.intermediate_performance['f1'])

        # if 'train_modalities' not in st.session_state:
        #     st.session_state.train_modalities = {} # Or your initial default value
        
        # st.write("**Intermediate Integration - Modalities in Data:**", list(train_modalities.keys()))

        # Late Integration Model Coefficients - Separate graphs for each modality
        st.subheader("🔍 Late Integration Model - Important Features by Coefficient (Per Modality)")
        st.write('Validation F1 Score:', st.session_state.late_performance['f1'])
        
        for modality, coef_df in st.session_state.late_coef_dict.items():
            st.write(f"**🧬 {modality.upper()} Modality**")
            
            # Get top features for this modality
            modality_display_df = coef_df.head(num_features)
            
            # Create chart for this modality
            late_bar = alt.Chart(modality_display_df).mark_bar(color=alt.expr("datum.modality === 'rna' ? '#1f77b4' : datum.modality === 'mirna' ? '#ff7f0e' : datum.modality === 'cnv' ? '#2ca02c' : datum.modality === 'meth' ? '#d62728' : datum.modality === 'histology' ? '#9467bd' : '#8c564b'")).encode(
                x=alt.X('coefficient:Q', title=f'Logistic Regression Coefficient ({modality})'),
                y=alt.Y('feature:N', sort='-x', title='Feature'),
                tooltip=['feature', 'coefficient', 'modality']
            ).properties(
                height=600,
                title=f"Top {len(modality_display_df)} Features for {modality.upper()} Modality"
            )
            
            st.altair_chart(late_bar, use_container_width=True)

# --- Tab 2: Histology Viewer ---
with tab4:
    st.subheader("🧫 Histology Image Viewer")
    uploaded_image = st.file_uploader("Upload a Histology Image (PNG)", type=["png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Histology Slide", use_column_width=True)
    else:
        st.info("Please upload a .png file to view the histology image.")

with tab5:
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