import pandas as pd
from sklearn.decomposition import PCA

def process_mythylation_data(df_meth: pd.DataFrame, manifest: pd.DataFrame, gene_list:list=None) -> pd.DataFrame:
    df_meth = df_meth.merge(manifest, left_index=True, right_on='submitter_id.samples')

    if gene_list is not None:
        cols = [col for col in df_meth.columns if col in gene_list or col in ['submitter_id.samples', 'subtype', 'split']]
        df_meth = df_meth[cols]
    else:
        gene_list = df_meth.columns[:-3].tolist()  

    df_meth.dropna(axis=1, how='any', inplace=True)  # Drop columns with all NaN values
    pca = PCA(n_components=100, random_state=42)    

    train_df = df_meth[df_meth['split'] == 'train'].drop(columns=['split'])
    meth_features_train = pca.fit_transform(train_df.drop(columns=['submitter_id.samples', 'subtype']))
    train_df = pd.DataFrame(meth_features_train, index=train_df['submitter_id.samples'])
    train_df['subtype'] = df_meth[df_meth['split'] == 'train']['subtype'].values

    val_df = df_meth[df_meth['split'] == 'val'].drop(columns=['split'])
    meth_features_val = pca.transform(val_df.drop(columns=['submitter_id.samples', 'subtype']))
    val_df = pd.DataFrame(meth_features_val, index=val_df['submitter_id.samples'])
    val_df['subtype'] = df_meth[df_meth['split'] == 'val']['subtype'].values

    test_df = df_meth[df_meth['split'] == 'test'].drop(columns=['split'])
    meth_features_test = pca.transform(test_df.drop(columns=['submitter_id.samples', 'subtype']))
    test_df = pd.DataFrame(meth_features_test, index=test_df['submitter_id.samples'])
    test_df['subtype'] = df_meth[df_meth['split'] == 'test']['subtype'].values

    
    return train_df, val_df, test_df