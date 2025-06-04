import logging

import pandas as pd

from SRC.Preprocess import cnv_preprocess, rna_preprocess, meth_preprocess, mirna_preprocess, mutation_preprocess

logging.basicConfig(level=logging.INFO)

def run_preprocesing():
    """
    Run preprocessing for all omics data.
    """

    logging.info("Running preprocessing for all omics data.")

    # Load data
    df_meth = pd.read_csv("../../data/TCGA-BRCA.methylation450.tsv", sep="\t")
    df_mir = pd.read_csv("../../data/TCGA-BRCA.mirna.tsv", sep="\t")
    df_rna = pd.read_csv("../../data/TCGA-BRCA.htseq_counts.tsv", sep="\t")
    df_cnv = pd.read_csv("../../data/TCGA-BRCA.gistic.tsv", sep="\t")
    df_mutation = pd.read_csv("../../data/TCGA-BRCA.mutect2_snv.tsv", sep="\t")

    manifest = pd.read_csv("../../data/hackathon_manifest.csv")

    gene_names_dict = pd.read_csv("../../data/ens_genename.txt", sep='\t')

    # Process each omics data
    train_meth, val_meth, test_meth = meth_preprocess.process_methylation_data(df_meth, manifest)
    train_mir, val_mir, test_mir = mirna_preprocess.process_mirna_data(df_mir, manifest)
    train_rna, val_rna, test_rna = rna_preprocess.rna_preprocess(df_rna, manifest, gene_names_dict)
    train_cnv, val_cnv, test_cnv = cnv_preprocess.cnv_preprocess(df_cnv, manifest)
    train_mutation, val_mutation, test_mutation = mutation_preprocess.mutation_preprocess(df_mutation, manifest)

    # assert the lengths of train are the same for all dfs
    assert len(train_meth) == len(train_mir) == len(train_rna) == len(train_cnv) == len(train_mutation), \
        "Training sets have different lengths. Please check the preprocessing steps."

    # assert the lengths of val are the same for all dfs
    assert len(val_meth) == len(val_mir) == len(val_rna) == len(val_cnv) == len(val_mutation), \
        "Validation sets have different lengths. Please check the preprocessing steps."

    # assert the lengths of test are the same for all dfs
    assert len(test_meth) == len(test_mir) == len(test_rna) == len(test_cnv) == len(test_mutation), \
        "Test sets have different lengths. Please check the preprocessing steps."

    #save to ../../processed_data/
    train_meth.to_csv("../../processed_data/train_meth.csv")
    val_meth.to_csv("../../processed_data/val_meth.csv")
    test_meth.to_csv("../../processed_data/test_meth.csv")

    train_mir.to_csv("../../processed_data/train_mir.csv")
    val_mir.to_csv("../../processed_data/val_mir.csv")
    test_mir.to_csv("../../processed_data/test_mir.csv")

    train_rna.to_csv("../../processed_data/train_rna.csv")
    val_rna.to_csv("../../processed_data/val_rna.csv")
    test_rna.to_csv("../../processed_data/test_rna.csv")

    train_cnv.to_csv("../../processed_data/train_cnv.csv")
    val_cnv.to_csv("../../processed_data/val_cnv.csv")
    test_cnv.to_csv("../../processed_data/test_cnv.csv")

    train_mutation.to_csv("../../processed_data/train_mutation.csv")
    val_mutation.to_csv("../../processed_data/val_mutation.csv")
    test_mutation.to_csv("../../processed_data/test_mutation.csv")

    logging.info("Preprocessing completed and results saved to ../../processed_data/")

if __name__ == "__main__":
    run_preprocesing()