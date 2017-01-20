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
    job_title_list = te.get_job_titles(te.resume_labels[document_idx])
    prepared_doc = tagger.prepare_doc(document)

    for job_title in job_title_list:
        prepared_doc = tagger.match_label(prepared_doc, job_title, "POS")
    prepared_doc = tagger.pos_tag(prepared_doc)

    dataset.resume_content.append(prepared_doc)

dataset.save()
#tagged_doc = tagger.match_label(te.resume_content[0], te.resume_labels[0].NewDataSet.Jobs[0].job_position.cdata.strip(), "TEST")



