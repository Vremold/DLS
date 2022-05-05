import os
from config import TEXT_PREPROCESS_CHOICE

# GitHub data
GH_DATA_DIR = "./Data/GH"
RAW_GH_REPOSITORIES_FILE = os.path.join(GH_DATA_DIR, "RawData", "repositories.txt")
RAW_GH_REPOSITORY_READEME_DIR = os.path.join(GH_DATA_DIR, "RawData", "Readmes")
CLEANED_GH_REPOSITORY_FILE = os.path.join(GH_DATA_DIR, "CleanedData", "repositories.txt")
CLEANED_GH_REPOSITORY_README_DIR = os.path.join(GH_DATA_DIR, "RawData", "Readmes")

# NER data
NER_DATA_DIR = "./Data/NER"
LABELING_DATA_DIR = os.path.join(NER_DATA_DIR, "LabelingData")
LABELED_NER_DATA = os.path.join(NER_DATA_DIR, "labeled_"+TEXT_PREPROCESS_CHOICE)
UNLABELED_NER_DATA = os.path.join(NER_DATA_DIR, "unlabeled_"+TEXT_PREPROCESS_CHOICE)

# NER training process
NER_DIR = "./NER"
NER_TRAINING_DATASET_DIR = os.path.join(NER_TRAINING_DIR, "data")

# KG
KG = "./StructureInfoExtraction/dls.gexf"
KG_NODE_EMBEDDING = "./StructureInfoExtraction/KGE/KE/ent_embedding_vec.txt"
KG_RELATION_EMBEDDING = "./StructureInfoExtraction/KGE/KE/rel_embedding_vec.txt"

# HIN
HIN = "./NonstructureInfoExtraction/hin.gexf"
HIN_EMBEDDING = "./NonstructureInfoExtraction/parsed_vec.txt"

# Verification data
VERIFICATION_DATA_DIR = "./Data/Verification/test_samples.txt"