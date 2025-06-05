import pandas as pd

def mutation_preprocess(df_mut: pd.DataFrame, manifest: pd.DataFrame, gene_list: list = None) -> pd.DataFrame:
    """
    Process mutation data to extract features and counts.

    Parameters:
    df_mut (pd.DataFrame): DataFrame containing mutation data.
    manifest (pd.DataFrame): DataFrame containing sample manifest with 'submitter_id.samples' column.
    gene_list (list, optional): List of genes to filter mutations. If None, all genes are used.

    Returns:
    pd.DataFrame: Processed DataFrame with mutation features.
    """
    df_mut.set_index(df_mut.columns[0], inplace=True)
    df_mut = df_mut.merge(manifest, left_index=True, right_on='submitter_id.samples')
    df_mut = df_mut[df_mut['filter'] == 'PASS']
    if gene_list is not None:
        df_mut = df_mut[df_mut['gene'].isin(gene_list)]
    else:
        gene_list = df_mut['gene'].unique()

    df_mut['missense'] = df_mut['effect'].apply(lambda x: 1 if x == 'Missense_Mutation' else 0)
    df_mut['stopgained'] = df_mut['effect'].apply(lambda x: 1 if x == 'Stop_Gained' else 0)
    df_mut['3UTR'] = df_mut['effect'].apply(lambda x: 1 if x == '3\'UTR' else 0)

    df_mut['num_mutations'] = df_mut.groupby('submitter_id.samples')['gene'].transform('count')
    df_mut['num_missense'] = df_mut.groupby('submitter_id.samples')['missense'].transform('sum')
    df_mut['num_stopgained'] = df_mut.groupby('submitter_id.samples')['stopgained'].transform('sum')
    df_mut['num_3UTR'] = df_mut.groupby('submitter_id.samples')['3UTR'].transform('sum')

    mutation_counts = df_mut.groupby(['submitter_id.samples', 'gene']).size().unstack(fill_value=0)
    mutation_counts.columns = [f'{col}' for col in mutation_counts.columns]
    mutation_counts.reset_index(inplace=True)

    processed_df = df_mut[
        ['submitter_id.samples', 'num_mutations', 'num_missense', 'num_stopgained', 'num_3UTR', 'subtype',
         'split']].drop_duplicates()
    processed_df = processed_df.merge(mutation_counts, on='submitter_id.samples', how='left')

    train_df = processed_df[processed_df['split'] == 'train']
    val_df = processed_df[processed_df['split'] == 'val']
    test_df = processed_df[processed_df['split'] == 'test']

    train_df = train_df.drop(columns=['split'])
    val_df = val_df.drop(columns=['split'])
    test_df = test_df.drop(columns=['split'])

    return train_df, val_df, test_df

if __name__ == "__main__":

    df_mut = pd.read_csv("../../data/TCGA-BRCA.mutect2_snv.tsv", sep="\t")
    manifest = pd.read_csv("../../data/hackathon_manifest.csv")
    gene_list = ['TP53', 'BRCA1', 'BRCA2']  # Example gene list

    train, val, test = mutation_preprocess(df_mut, manifest, gene_list)

    print(train)

