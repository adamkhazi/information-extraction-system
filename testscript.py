import pdb
import numpy as np
from numpy import array
from generate_dataset import GenerateDataset
from document import TokenIterable
from tagger import Tagger
from text_extractor import TextExtractor

gd = GenerateDataset()
te = TextExtractor()
tagger = Tagger()

document_example = [
        ["1A", "1B", "1C", "1D"],
        ["2A", "2B", "2C", "2D"],
        ["3A", "3B", "3C", "3D"],
        ["4A", "4B", "4C", "4D"],
        ]

te.prepare_dataset()
tagger.match_label(te.pdf_content[0],
        te.xml_labels[0].NewDataSet.Jobs[0].job_position.cdata.strip(), "TEST")
#dataset = gd.read_tagged_tokens()

#dataset = array(dataset[0])


