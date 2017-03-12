import os
import io
import csv
import glob
import math
import copy
import pdb
from random import shuffle

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
                            if self.read_error(row):
                                print("read error in file %s" % filename)
                            single_line.append((row[0], row[1], row[2], row[3]))
                    dataset_docs.append(single_doc)
            if count == nr_of_files and nr_of_files != -1:
                break
        self.resume_content = dataset_docs
        self.__logger.println("read %s files from: " % len(self.resume_content) + self.__dataset_folder)
        self.resume_content = self.shuffle_data(self.resume_content)
        return self.resume_content

    def read_error(self, line):
        if len(line[0]) == 0 or len(line[1]) == 0 or len(line[2]) == 0 or len(line[3]) == 0:
            return True
        else:
            return False

    def shuffle_data(self, data):
        self.__logger.println("shuffling dataset")
        shuffle(data)
        return data

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

    def obtain_default_tags(self, number, data):
        default_tag_lines_list = []
        line_count = 0
        for doc_idx, doc in enumerate(data):
            default_tag_lines_list.append([])
            for line_idx, line in enumerate(doc):
                add_line = True
                for token_idx, token in enumerate(line):
                    if token[3] != "O": 
                        delete_idx = False 
                if add_line:
                    line_count += 1
                    default_tag_lines_list[doc_idx].append(line)
                if line_count >= number:
                    print("returning %s lines with default tags" % line_count)
                    return default_tag_lines_list

    def filter_for_filled_tags(self, data):
        data_copy = copy.deepcopy(data)
        to_delete_idx = []
        for doc_idx, doc in enumerate(data_copy):
            to_delete_idx.append([])
            for line_idx, line in enumerate(doc):
                delete_idx = True
                for token_idx, token in enumerate(line):
                    if token[3] != "O": 
                        delete_idx &= False 
                if delete_idx:
                    to_delete_idx[doc_idx].append(line_idx)

        line_count = 0
        for doc_idx, doc in enumerate(to_delete_idx):
            delete_count = 0
            for idx, del_line_idx in enumerate(doc):
                del data_copy[doc_idx][del_line_idx-delete_count]
                delete_count += 1
            line_count += len(data_copy[doc_idx])

        print("returning %s filled lines" % line_count)
        return line_count, data_copy

    def save_doc_lines(self, doc_lines, filenames, folder_name):
        path = folder_name + "/"

        pdb.set_trace()
        for doc_idx, doc in enumerate(doc_lines):
            doc_file = open(path + str(filenames[doc_idx][0]) + '.txt', 'w', encoding='utf-8')
            for line_idx, line in enumerate(doc):
                if not line:
                    pass
                else:
                    doc_file.write("{}\n".format(line))
            doc_file.close()

        self.__logger.println("saved dataset to: " + path)
