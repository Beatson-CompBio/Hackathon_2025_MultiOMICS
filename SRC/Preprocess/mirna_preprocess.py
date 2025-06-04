import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def process_mirna_data(df_mir: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:

    logging.info("Processing miRNA data")

    df_mir.set_index('miRNA_ID', inplace=True)
    df_mir = df_mir.T

    df_mir = df_mir.merge(manifest, left_index=True, right_on='submitter_id.samples')

    df_train = df_mir[df_mir['split'] == 'train'].drop(columns=['split'])
    df_val = df_mir[df_mir['split'] == 'val'].drop(columns=['split'])
    df_test = df_mir[df_mir['split'] == 'test'].drop(columns=['split'])

    return df_train, df_val, df_test

if __name__ == "__main__":
    df_mir = pd.read_csv("../../data/TCGA-BRCA.mirna.tsv", sep="\t")
    manifest = pd.read_csv("../../data/hackathon_manifest.csv")

    train, val, test = process_mirna_data(df_mir, manifest)
    print(train)