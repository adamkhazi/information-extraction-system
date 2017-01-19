import pdb
import numpy as np
from numpy import array
from generate_dataset import GenerateDataset
from document import TokenIterable
from tagger import Tagger
from extractor import Extractor
from dataset import Dataset

gd = GenerateDataset()
te = Extractor()
tagger = Tagger()
dataset = Dataset()

document_example = [
        ["1A", "1B", "1C", "1D"],
        ["2A", "2B", "2C", "2D"],
        ["3A", "3B", "3C", "3D"],
        ["4A", "4B", "4C", "4D"],
        ]

te.prepare_dataset()
for document_idx, document in enumerate(te.resume_content):
    print("doc nr: " + str(document_idx))
    if not te.resume_labels[document_idx].NewDataSet.Jobs[0]:
        continue
        label_len = len(te.resume_labels[document_idx].NewDataSet.Jobs[0].job_position.cdata.strip())
        if label_len == 0 or label_len == 1:
            continue
    label = te.resume_labels[document_idx].NewDataSet.Jobs[0].job_position.cdata.strip()
    dataset.resume_content.append(tagger.match_label(document, label, "POS"))

dataset.save()
#tagged_doc = tagger.match_label(te.resume_content[0], te.resume_labels[0].NewDataSet.Jobs[0].job_position.cdata.strip(), "TEST")



