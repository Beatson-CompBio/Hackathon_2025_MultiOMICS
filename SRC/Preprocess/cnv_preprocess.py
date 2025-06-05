import pandas as pd
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)

### Input Files: Please adjust to your liking to pass as parameters ###
# file = pd.read_csv("TCGA-BRCA.gistic.tsv", sep='\t')
# manifest = pd.read_csv("hackathon_manifest.csv")

# file.set_index(file.columns[0], inplace=True)

### CNV Preprocess: ###

def cnv_preprocess(input_df: pd.DataFrame, manifest: pd.DataFrame, gene_List=False, n_components=100):

    logging.info("Preprocessing CNV")

    input_df = input_df.T
    df_cnv= input_df.copy() #techincal debt

    df_cnv = df_cnv.merge(manifest, left_index=True, right_on='submitter_id.samples')


    df_cnv.dropna(axis=1, how='any', inplace=True)
    pca = PCA(n_components=100, random_state=42)

    train_df = df_cnv[df_cnv['split'] == 'train'].drop(columns=['split'])
    cnv_features_train = pca.fit_transform(train_df.drop(columns=['submitter_id.samples', 'subtype']))
    train_df = pd.DataFrame(cnv_features_train, index=train_df['submitter_id.samples'])
    train_df['subtype'] = df_cnv[df_cnv['split'] == 'train']['subtype'].values
    train_df['submitter_id.samples'] = train_df.index

    val_df = df_cnv[df_cnv['split'] == 'val'].drop(columns=['split'])
    cnv_features_val = pca.transform(val_df.drop(columns=['submitter_id.samples', 'subtype']))
    val_df = pd.DataFrame(cnv_features_val, index=val_df['submitter_id.samples'])
    val_df['subtype'] = df_cnv[df_cnv['split'] == 'val']['subtype'].values
    val_df['submitter_id.samples'] = val_df.index

    test_df = df_cnv[df_cnv['split'] == 'test'].drop(columns=['split'])
    cnv_features_test = pca.transform(test_df.drop(columns=['submitter_id.samples', 'subtype']))
    test_df = pd.DataFrame(cnv_features_test, index=test_df['submitter_id.samples'])
    test_df['subtype'] = df_cnv[df_cnv['split'] == 'test']['subtype'].values
    test_df['submitter_id.samples'] = test_df.index

    return train_df, val_df, test_df

if __name__ == "__main__":
    df = pd.read_csv("../../data/TCGA-BRCA.gistic.tsv", sep="\t")
    manifest = pd.read_csv("../../data/hackathon_manifest.csv")
    train, val, test = cnv_preprocess(df, manifest)

    print(train)
