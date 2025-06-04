import pandas as pd

df = pd.read_csv("expression/TCGA-BRCA.htseq_counts.tsv", sep="\t")
manifest = pd.read_csv("hackathon_manifest.csv")

def rna_preprocess(input_df: pd.DataFrame, manifest: pd.DataFrame, gene_list=False):
    """
    A function to preprocess rna sequencing data 
    :param input_df: A DataFrame containing RNA sequencing data with EnsembleIDs as index
    :return
    """
    df = input_df
    df.set_index('Ensembl_ID', inplace=True)
    df = df.T
    df = df.drop(['__no_feature','__ambiguous','__too_low_aQual','__not_aligned','__alignment_not_unique'], axis=1)
    df = df.T
    df = df.merge(gene_names_dict, left_index = True, right_on = 'Gene stable ID version', how='inner')
    df.set_index('Gene name', inplace=True)
    df = df.drop(['Gene stable ID', 'Gene stable ID version', 'Transcript stable ID', 'Transcript stable ID version'], axis=1)
    df = df.T
    df_merge = df.merge(manifest, left_index=True, right_on = 'submitter_id.samples', how= 'inner')

    train_df = df_merge[df_merge['split'] == 'train']
    val_df = df_merge[df_merge['split'] == 'val']
    test_df = df_merge[df_merge['split'] == 'test']

    train_df = train_df.drop('split', axis=1)
    val_df = val_df.drop('split', axis=1)
    test_df = test_df.drop('split', axis=1)
    
    return train_df, val_df, test_df

train, val, test = rna_preprocess(df, manifest)