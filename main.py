import sys

from generate_dataset import GenerateDataset
from crf_suite import CrfSuite
from annotator import Annotator

__argument_train = "-t"
__argument_annotate_dataset = "-a"

def main():
    if sys.argv[1] == __argument_train:
        train_model()
    elif sys.argv[1] == __argument_annotate_dataset:
        if len(sys.argv) > 2:
            annotate_data(int(sys.argv[2]))
        else:
            annotate_data()
    else:
        print("Commands accepted:")
        print("train: -t")
        print("annotate dataset: -a <number_of_documents>")

def annotate_db_data():
    """
    gd = GenerateDataset()
    gd.pull_db_records(2, 1000)
    gd.tokenize_text()
    gd.pos_tag_tokens()
    gd.ner_tag_tokens()
    gd.nonlocal_ner_tag_tokens()
    gd.save_tagged_tokens()
    """
    raise NotImplementedError

def annotate_data(nr_docs=-1):
    annotator = Annotator()
    annotator.prepare_dataset(nr_docs)

def train_model():
    cs = CrfSuite()
    cs.get_dataset()
    #cs.load_embeddings()
    cs.generate_embeddings()
    #cs.encode_dataset()

    cs.split_dataset()
    cs.generate_features()
    cs.train_model()
    cs.test_model()


if __name__ == '__main__':
  main()
