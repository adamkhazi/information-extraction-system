from generate_dataset import GenerateDataset
#from crf_suite import 


def main():
    gd = GenerateDataset()
    gd.pull_db_records(0, 5)
    gd.tokenize_text()
    gd.pos_tag_tokens()
    gd.ner_tag_tokens()
    gd.save_tagged_tokens()

if __name__ == '__main__':
  main()
