import os
import io
import csv
import glob

from logger import Logger

class Dataset:
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
                            single_line.append((row[0], row[1], row[2], row[3]))
                    dataset_docs.append(single_doc)
            if count == nr_of_files and nr_of_files != -1:
                break
        self.resume_content = dataset_docs
        self.__logger.println("read %s files from: " % len(self.resume_content) + self.__dataset_folder)
