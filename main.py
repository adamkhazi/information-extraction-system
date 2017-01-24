import sys
import pdb

from generate_dataset import GenerateDataset
from crf_suite import CrfSuite
from annotator import Annotator
from api import API

__argument_train = "-t"
__argument_annotate_dataset = "-a"
__argument_api = "-rn"

def main():
    command_arg = sys.argv[1]
    if command_arg == __argument_train:
        train_model()

    elif command_arg == __argument_annotate_dataset:
        if len(sys.argv) > 2:
            annotate_data(nr_docs=int(sys.argv[2]))
        else:
            annotate_data()

    elif command_arg == __argument_api:
        run_api()

    else:
        print("Commands accepted:")
        print("train: -t")
        print("annotate dataset: -a <number_of_documents>")
        print("run api: -rn")

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

def run_api():
    api = API()
    api.run()

if __name__ == '__main__':
  main()
