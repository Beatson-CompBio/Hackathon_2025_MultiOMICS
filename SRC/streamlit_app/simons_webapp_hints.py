import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import streamlit as st
import altair as alt

logging.basicConfig(level=logging.INFO)


#you can get cool emojis from https://emojipedia.org/

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

# --- Streamlit Interface with Tabs ---

st.set_page_config(page_title="Multimodal Model App", layout="wide")
st.title('🧬 My Cool Multimodal Webapp')

tab1, tab2, tab3 = st.tabs(["📈 Exploratory Data Analysis", "📊 Model Training", "🧫 Histology"])

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

    if 'coef_df' not in st.session_state:
        st.session_state.coef_df = None
        st.session_state.performance = None

    if st.button(f'Train and Predict on {selected_modalities}'):
        with st.spinner('Loading data...'):
            if not selected_modalities:
                st.error("Please select at least one modality.")
                st.stop()
            train_modalities = {k: pd.read_csv(possible_train_modalities[k]) for k in selected_modalities}
            val_modalities = {k: pd.read_csv(possible_val_modalities[k]) for k in selected_modalities}
            test_modalities = {k: pd.read_csv(possible_test_modalities[k]) for k in selected_modalities}

        with st.spinner('Training model...'):
            model = Early_or_Single_Model()
            predictions, coef_df, test_set_predictions = model.wrapper(train_modalities, val_modalities, test_modalities)
            val_performance = evaluate_model_on_preds(predictions, val_modalities[selected_modalities[0]]['subtype'])
            test_performance = evaluate_model_on_preds(test_set_predictions, test_modalities[selected_modalities[0]]['subtype'])
            st.success('Model trained successfully!')

            st.write('Validation F1 Score:', val_performance['f1'])

            #if st.button('Show Test Set Predictions'):
            st.warning('Test set predictions should only be viewed once at the end of the project')
            #have a dropdown to show test?
            st.write('Test F1 Score:', test_performance['f1'])

        st.session_state.coef_df = coef_df
        #st.session_state.val_performance = val_performance




    if st.session_state.coef_df is not None:
        # st.write('Validation F1 Score:', st.session_state.val_performance['f1'])
        #
        # if st.button('Show Test Set Predictions'):
        #     st.warning('Test set predictions should only be viewed once at the end of the project')
        #     st.write('Test Set Predictions:', test_performance['f1'])

        num_features = st.slider("Select number of features to display", min_value=10, max_value=500, value=100, step=10)

        display_df = st.session_state.coef_df.head(num_features)

        st.subheader("🔍 Important Features by Coefficient")
        bar = alt.Chart(display_df).mark_bar().encode(
            x=alt.X('coefficient:Q', title='Logistic Regression Coefficient'),
            y=alt.Y('feature:N', sort='-x', title='Feature'),
            color=alt.Color('modality:N', title='Modality'),
            tooltip=['feature', 'coefficient', 'modality']
        ).properties(height=900)
        st.altair_chart(bar, use_container_width=True)

# --- Tab 2: Histology Viewer ---
with tab3:
    st.subheader("🧫 Histology Image Viewer")
    uploaded_image = st.file_uploader("Upload a Histology Image (PNG)", type=["png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Histology Slide", use_column_width=True)
    else:
        st.info("Please upload a .png file to view the histology image.")


