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

    # print modality shapes
    #get the intersection of all submitter_id.samples for train, val, and test sets

    train_intersection_ids = set(train_meth['submitter_id.samples']).intersection(
        set(train_mir['submitter_id.samples']),
        set(train_rna['submitter_id.samples']),
        set(train_cnv['submitter_id.samples']),
        set(train_mutation['submitter_id.samples'])
    )

    val_intersection_ids = set(val_meth['submitter_id.samples']).intersection(
        set(val_mir['submitter_id.samples']),
        set(val_rna['submitter_id.samples']),
        set(val_cnv['submitter_id.samples']),
        set(val_mutation['submitter_id.samples'])
    )

    test_intersection_ids = set(test_meth['submitter_id.samples']).intersection(
        set(test_mir['submitter_id.samples']),
        set(test_rna['submitter_id.samples']),
        set(test_cnv['submitter_id.samples']),
        set(test_mutation['submitter_id.samples'])
    )

    logging.info(f"Train set intersection IDs: {len(train_intersection_ids)}")
    logging.info(f"Validation set intersection IDs: {len(val_intersection_ids)}")
    logging.info(f"Test set intersection IDs: {len(test_intersection_ids)}")

    #filter the dataframes to only include the intersection IDs
    train_meth = train_meth[train_meth['submitter_id.samples'].isin(train_intersection_ids)]
    val_meth = val_meth[val_meth['submitter_id.samples'].isin(val_intersection_ids)]
    test_meth = test_meth[test_meth['submitter_id.samples'].isin(test_intersection_ids)]

    train_mir = train_mir[train_mir['submitter_id.samples'].isin(train_intersection_ids)]
    val_mir = val_mir[val_mir['submitter_id.samples'].isin(val_intersection_ids)]
    test_mir = test_mir[test_mir['submitter_id.samples'].isin(test_intersection_ids)]

    train_rna = train_rna[train_rna['submitter_id.samples'].isin(train_intersection_ids)]
    val_rna = val_rna[val_rna['submitter_id.samples'].isin(val_intersection_ids)]
    test_rna = test_rna[test_rna['submitter_id.samples'].isin(test_intersection_ids)]

    train_cnv = train_cnv[train_cnv['submitter_id.samples'].isin(train_intersection_ids)]
    val_cnv = val_cnv[val_cnv['submitter_id.samples'].isin(val_intersection_ids)]
    test_cnv = test_cnv[test_cnv['submitter_id.samples'].isin(test_intersection_ids)]

    train_mutation = train_mutation[train_mutation['submitter_id.samples'].isin(train_intersection_ids)]
    val_mutation = val_mutation[val_mutation['submitter_id.samples'].isin(val_intersection_ids)]
    test_mutation = test_mutation[test_mutation['submitter_id.samples'].isin(test_intersection_ids)]

    #assert that the number of samples in each modality is the same
    assert len(train_meth) == len(train_mir) == len(train_rna) == len(train_cnv) == len(train_mutation), "Train sets do not match in size"
    assert len(val_meth) == len(val_mir) == len(val_rna) == len(val_cnv) == len(val_mutation), "Validation sets do not match in size"
    assert len(test_meth) == len(test_mir) == len(test_rna) == len(test_cnv) == len(test_mutation), "Test sets do not match in size"

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