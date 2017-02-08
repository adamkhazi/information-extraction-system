import os
import io
import csv
import glob
import math

from logger import Logger

class Dataset():
    # saves and read annotated files to this folder
    __dataset_folder = "dataset_files"

    resume_content = []
    resume_labels = []

    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("dataset created")

    def save(self):
        directory=self.__dataset_folder
        files=glob.glob('*.txt')
        for filename in files:
            os.unlink(filename)

        path = self.__dataset_folder + "/"

        for doc_idx, doc in enumerate(self.resume_content):
            doc_file = open(path + str(doc_idx) + '.txt', 'w', encoding='utf-8')
            for line_idx, line in enumerate(doc):
                for token_idx, token in enumerate(line):
                    doc_file.write("{}\t{}\t{}\t{}\n".format(token[0], token[1], token[2], token[3]))
                doc_file.write("\n")
            doc_file.close()

        self.__logger.println("saved dataset to: " + path)

    def read(self, nr_of_files=-1):
        dataset_docs = []
        count = 0
        wrongly_formed_files = set()
        for filename in os.listdir(self.__dataset_folder):
            count += 1
            current_file_path = self.__dataset_folder + "/" + filename
            if current_file_path.endswith(".txt"):
                with io.open(current_file_path, 'r', encoding='utf-8') as tsvin:
                    single_doc = []
                    single_line = []
                    tsvin = csv.reader(tsvin, delimiter='\t')
                    for row in tsvin:
                        if not row:
                            single_doc.append(single_line)
                            single_line = []
                        else:
                            single_line.append((row[0], row[1], row[2], row[3]))
                            if len(row[3]) < 1:
                                wrongly_formed_files.add(filename)
                    dataset_docs.append(single_doc)
            if count == nr_of_files and nr_of_files != -1:
                break
        self.resume_content = dataset_docs
        self.__logger.println("read %s files from: " % len(self.resume_content) + self.__dataset_folder)
        # temporary
        print(wrongly_formed_files)
        return self.resume_content

    def read_doc(self, filename):
        current_file_path = self.__dataset_folder + "/" + filename
        single_doc = []
        if current_file_path.endswith(".txt"):
            with io.open(current_file_path, 'r', encoding='utf-8') as tsvin:
                single_line = []
                tsvin = csv.reader(tsvin, delimiter='\t')
                for row in tsvin:
                    if not row:
                        single_doc.append(single_line)
                        single_line = []
                    else:
                        single_line.append((row[0], row[1], row[2], row[3]))
        return single_doc


    # TODO add with shuffle option
    # takes in data and splits at a ratio
    # first set is training and second test
    def split_dataset(self, data, split_point=0.75):
        split_point = math.ceil(len(data) * split_point)
        train_set = data[0:split_point]
        test_set = data[split_point:]

        self.__logger.println("%s split on dataset: %s training, %s test" % (split_point, len(train_set), len(test_set)))
        return train_set, test_set

    # generate word counts and unique word idxs
    def encode_dataset(self, data):
        word2idx = {}
        word2count = {}
        word_idx = 0

        for doc_idx, doc in enumerate(data):
            for line_idx, line in enumerate(doc):
                for token_idx, token in enumerate(line):
                    word = token[0].lower()
                    if word not in word2idx:
                        word2idx[word] = word_idx
                        word2count[word] = 1
                        word_idx += 1
                    else:
                        word2count[word] += 1

        self.__logger.println("encode_dataset: " + str(word_idx) + " unique words")
        return word2count, word2idx

    # flatten a doc list into lines
    def docs2lines(self, docs):
        lines = []
        for doc in docs:
            for line in doc:
                lines.append(line)
        return lines
