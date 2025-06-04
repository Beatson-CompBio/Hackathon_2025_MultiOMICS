import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score
import sys
import os
local_dir = '/Users/farzaneh/Downloads/hackathon_resources/'
sys.path.append(local_dir+'Hackathon_2025_MultiOMICS')
import SRC
from SRC.Preprocess.mutation_preprocess import process_mutation_data
from SRC.Preprocess.meth_preprocess import process_methylation_data



def fit_model(X_train, y_train, X_val, y_val):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    auc = roc_auc_score(y_val, y_pred)
    prc = precision_score(y_val, y_pred, average='weighted')
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f'AUC: {auc:.4f}, Precision: {prc:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}')

def x_y_maker(train_df, val_df, test_df):
    X_train = train_df.drop(columns=['subtype', 'submitter_id.samples'])
    y_train = train_df['subtype']
    X_val = val_df.drop(columns=['subtype', 'submitter_id.samples'])
    y_val = val_df['subtype']
    X_test = test_df.drop(columns=['subtype', 'submitter_id.samples'])
    y_test = test_df['subtype']
    return X_train, y_train, X_val, y_val, X_test, y_test

df_label = pd.read_csv(local_dir + r'/hackathon_manifest.csv')
# RNA Preprocessing
df_rna = pd.read_csv(local_dir + r'/expression/TCGA-BRCA.htseq_counts.tsv', sep='\t')
# df_rna.set_index(df_rna.columns[0], inplace=True)
# df_rna = df_rna.transpose()
# df_rna = df_rna.merge(df_label, left_index=True, right_on='submitter_id.samples')

# cnv Preprocessing
df_cnv = pd.read_csv(local_dir + r'/cnv/TCGA-BRCA.gistic.tsv', sep='\t')
# df_cnv.set_index(df_cnv.columns[0], inplace=True)
# df_cnv = df_cnv.transpose()
# df_cnv = df_cnv.merge(df_label, left_index=True, right_on='submitter_id.samples')

# mRNA Preprocessing
df_mir = pd.read_csv(local_dir + r'/miR_expression/TCGA-BRCA.mirna.tsv', sep='\t')
# df_mir.set_index(df_mir.columns[0], inplace=True)
# df_mir = df_mir.transpose()
# df_mut = pd.read_csv(local_dir + r'/mutation/TCGA-BRCA.mutect2_snv.tsv', sep='\t')

df_meth = pd.read_csv(local_dir + r'/methylation/TCGA-BRCA.methylation450.tsv', sep='\t')
train_meth, val_meth, test_meth = process_methylation_data(df_meth, df_label)
X_train_meth, y_train_meth, X_val_meth, y_val_meth, X_test_meth, y_test_meth = x_y_maker(train_meth, val_meth, test_meth)
fit_model(X_train_meth, y_train_meth, X_val_meth, y_val_meth)

df_mut = pd.read_csv(local_dir + r'/mutation/TCGA-BRCA.mutect2_snv.tsv', sep='\t')
train_mut, val_mut, test_mut = process_mutation_data(df_mut, df_label)
X_train, y_train, X_val, y_val, X_test, y_test = x_y_maker(train_mut, val_mut, test_mut)
fit_model(X_train, y_train, X_val, y_val)



