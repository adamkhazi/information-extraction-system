import glob

class Dataset:
    __dataset_folder = "db_generated_datasets"

    resume_content = []
    resume_labels = []

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
                    doc_file.write("{}\t{}\n".format(token[0], token[3]))
                doc_file.write("\n")
            doc_file.close()

        print("Saved tagged tokens to: " + path)

    def read(self):
        dataset_docs = []
        for filename in os.listdir(self.__dataset_folder):
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
                            single_line.append((row[0], row[1]))
                    dataset_docs.append(single_doc)
        self.resume_content = dataset_docs
