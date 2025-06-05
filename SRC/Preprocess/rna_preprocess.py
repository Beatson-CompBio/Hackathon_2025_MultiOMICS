import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)

def rna_preprocess(input_df: pd.DataFrame, manifest: pd.DataFrame, gene_names_dict: pd.DataFrame, gene_list=False):
    """
    A function to preprocess rna sequencing data 
    :param input_df: A DataFrame containing RNA sequencing data with EnsembleIDs as index
    :return
    """

    logging.info("Preprocessing RNA sequencing data")

    df = input_df
    df.set_index('Ensembl_ID', inplace=True)
    df = df.T
    df = df.drop(['__no_feature', '__ambiguous', '__too_low_aQual', '__not_aligned', '__alignment_not_unique'], axis=1)
    df = df.T
    df = df.merge(gene_names_dict, left_index=True, right_on='Gene stable ID version', how='inner')
    df.set_index('Gene name', inplace=True)
    df = df.drop(['Gene stable ID', 'Gene stable ID version', 'Transcript stable ID', 'Transcript stable ID version'],
                 axis=1)
    df = df.T
    #if gene list not false filter cols to gene list
    if gene_list is not False:
        cols = [col for col in df.columns if col in gene_list or col in ['submitter_id.samples', 'subtype', 'split']]
        df = df[cols]

    df_merge = df.merge(manifest, left_index=True, right_on='submitter_id.samples', how='inner')

    train_df = df_merge[df_merge['split'] == 'train']
    val_df = df_merge[df_merge['split'] == 'val']
    test_df = df_merge[df_merge['split'] == 'test']

    train_df = train_df.drop('split', axis=1)
    val_df = val_df.drop('split', axis=1)
    test_df = test_df.drop('split', axis=1)

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = pd.read_csv("../../data/TCGA-BRCA.htseq_counts.tsv", sep="\t")
    manifest = pd.read_csv("../../data/hackathon_manifest.csv")
    gene_names_dict = pd.read_csv("../../data/ens_genename.txt", sep='\t')
    train, val, test = rna_preprocess(df, manifest, gene_names_dict)

    print(train)


