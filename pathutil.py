import os
from config import TEXT_PREPROCESS_CHOICE

PROJECT_ABS_DIR = "/media/dell/disk/wujw/dlss"

# GitHub data
GH_DATA_DIR = PROJECT_ABS_DIR + "/Data/GH"
RAW_GH_REPOSITORIES_FILE = os.path.join(GH_DATA_DIR, "RawData", "repositories.txt")
RAW_GH_REPOSITORY_READEME_DIR = os.path.join(GH_DATA_DIR, "RawData", "Readmes")
CLEANED_GH_REPOSITORY_FILE = os.path.join(GH_DATA_DIR, "CleanedData", "repositories.txt")
CLEANED_GH_REPOSITORY_README_DIR = os.path.join(GH_DATA_DIR, "RawData", "Readmes")

# NER data
NER_DATA_DIR = PROJECT_ABS_DIR + "/Data/NER"
LABELING_DATA_DIR = os.path.join(NER_DATA_DIR, "LabelingData")
LABELED_NER_DATA = os.path.join(NER_DATA_DIR, "labeled_"+TEXT_PREPROCESS_CHOICE)
UNLABELED_NER_DATA = os.path.join(NER_DATA_DIR, "unlabeled_"+TEXT_PREPROCESS_CHOICE)

# NER training process
NER_TRAINING_DIR = PROJECT_ABS_DIR + "/NER"
NER_TRAINING_DATASET_DIR = os.path.join(NER_TRAINING_DIR, "data")

# KG
KG = PROJECT_ABS_DIR + "/StructureInfoExtraction/dls.gexf"
KG_NODE_EMBEDDING = PROJECT_ABS_DIR + "/StructureInfoExtraction/KGE/KE/ent_embedding_vec.txt"
KG_RELATION_EMBEDDING = PROJECT_ABS_DIR + "/StructureInfoExtraction/KGE/KE/rel_embedding_vec.txt"

# HIN
HIN = PROJECT_ABS_DIR + "/NonstructureInfoExtraction/exports/hin.gexf"
HIN_EMBEDDING = PROJECT_ABS_DIR + "/NonstructureInfoExtraction/exports/parsed_vec.txt"

# Verification data
VERIFICATION_DATA_DIR = PROJECT_ABS_DIR + "/Data/Verification/test_samples.txt"