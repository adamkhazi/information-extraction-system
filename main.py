from generate_dataset import GenerateDataset
from crf_suite import CrfSuite


def main():
    gd = GenerateDataset()
    gd.pull_db_records(2, 400)
    gd.tokenize_text()
    gd.pos_tag_tokens()
    gd.ner_tag_tokens()
    gd.nonlocal_ner_tag_tokens()
    gd.save_tagged_tokens()

    cs = CrfSuite()
    cs.get_dataset()
    cs.split_dataset()
    cs.generate_features()
    cs.train_model()
    cs.test_model()

if __name__ == '__main__':
  main()
