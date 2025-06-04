import pandas as pd
from sklearn.decomposition import PCA

### Input Files: Please adjust to your liking to pass as parameters ### 
# file = pd.read_csv("TCGA-BRCA.gistic.tsv", sep='\t')
# manifest = pd.read_csv("hackathon_manifest.csv")

# file.set_index(file.columns[0], inplace=True)

### CNV Preprocess: ###

def cnv_preprocess(input_df:pd.DataFrame, manifest:pd.DataFrame, gene_List=False, n_components=100):
    input_df = input_df.T
    
    merged_df = pd.merge(input_df, manifest, left_index=True, right_on='submitter_id.samples', how='inner')

    train_df = merged_df[merged_df['split'] == 'train'].drop(columns=['submitter_id.samples','split','subtype'])
    val_df = merged_df[merged_df['split'] == 'val'].drop(columns=['submitter_id.samples','split','subtype'])
    test_df = merged_df[merged_df['split'] == 'test'].drop(columns=['submitter_id.samples','split','subtype'])

    # Applying PCA
    pca = PCA(n_components)
    pca.fit(train_df)
    transformed_train = pca.transform(train_df)
    transformed_val = pca.transform(val_df)
    transformed_test = pca.transform(test_df)

    transformed_train = pd.DataFrame(transformed_train, index=train_df.index)

    return transformed_train, transformed_val, transformed_test
