import nltk
import sklearn
import pycrfsuite
import csv
import math
import io
import time
import logging
import copy
import gensim

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from gensim.models import word2vec

from generate_dataset import GenerateDataset
from dataset import Dataset
from logger import Logger
from tags import Tags

class CrfSuite(Tags):
    __seperator = "/"
    __pre_trained_models_folder = "pre_trained_models"
    __google_news_word2vec = "GoogleNews-vectors-negative300.bin.gz"

    __default_ner_tag = "O"

    __w2v_model_name = "generated_w2v_model"

    # pre-trained embeddings
    def load_embeddings(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        path = self.__pre_trained_models_folder + self.__seperator + self.__google_news_word2vec
        self.w2v_model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

    def generate_embeddings(self):
        #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        copy_total_sents = copy.deepcopy(self.total_sents)
        all_lines = []
        for doc_idx, doc in enumerate(copy_total_sents):
            for line_idx, line in enumerate(doc):
                all_lines.append(line)

        for line_idx, line in enumerate(all_lines):
            for token_idx, token in enumerate(line):
                all_lines[line_idx][token_idx] = token[0].lower()

        self.w2v_model = word2vec.Word2Vec(all_lines, size=30, iter=10)
        self.w2v_model.save(self.__w2v_model_name)

    # n-gram generation
    def encode_dataset(self):
        self.word2idx = {}
        self.word2count = {}
        self.word_idx = 0

        for doc_idx, doc in enumerate(self.total_sents):
            for line_idx, line in enumerate(doc):
                for token_idx, token in enumerate(line):
                    word = token[0].lower()

                    if word not in self.word2idx:
                        self.word2idx[word] = self.word_idx
                        self.word2count[word] = 1
                        self.word_idx += 1
                    else:
                        self.word2count[word] += 1


        print("encode_dataset: " + str(self.word_idx) + " unique words")

    def get_dataset(self):
        dataset = Dataset()
        dataset.read()
        self.total_sents = dataset.resume_content
        print("Read " + str(len(self.total_sents)) + " documents")

    def split_dataset(self):
        split_point = math.ceil(len(self.total_sents) * 0.75)
        self.train_sents = self.total_sents[0:split_point]
        self.test_sents = self.total_sents[split_point+1:]
        print("Split dataset")

    def word2features(self, line, token_idx, line_idx, doc_idx, doc_size):
        word = line[token_idx][0]
        postag = line[token_idx][1]
        nonlocalnertag = line[token_idx][2]

        features = {
                "bias": 1.0,
                "word": word.lower(),
                'word[-3:]': word[-3:],
                'word[-2:]': word[-2:],
                'word.isupper': word.isupper(),
                'word.istitle': word.istitle(),
                'word.isdigit': word.isdigit(),
                'word.idx': float(token_idx),
                'line.idx': float(line_idx),
                'line.size': float(len(line)),
                'pos': postag,
                'pos[-3:]': postag[-3:],
                'pos[-2:]': postag[-2:],
                'nonlocalner': nonlocalnertag
        }

        try:
            for d_idx, dimension in enumerate(self.w2v_model[word.lower()]):
                features["we_dimen_"+str(d_idx)] = dimension
        except KeyError:
            features["we_dimen_"+str(0)] = "UNKNOWN"

        if token_idx > 0:
            word1 = line[token_idx-1][0]
            postag1 = line[token_idx-1][1]
            nonlocalnertag1 = line[token_idx-1][2]

            features['word-1'] = word1.lower()
            features['pos-1'] = postag1
            features['posbigram-1'] = postag1 + postag
            features['pos[-3:]'] = postag1[-3:]
            features['pos[-2:]'] = postag1[-2:]
            features['bigram-1'] = word1.lower() + word.lower()
            features['word-1.isupper'] = word1.isupper()
            features['word-1.istitle'] = word1.istitle()
            features['word-1.isdigit'] = word1.isdigit()
            features['word-1[-3:]'] = word1[-3:],
            features['word-1[-2:]'] = word1[-2:],
            features['word-1.idx']= float(token_idx-1)
        else:
            features['bigram-1'] = "BOL" + word.lower()
            features['BOL'] = 1.0

        if token_idx < len(line)-1:
            word1 = line[token_idx+1][0]
            postag1 = line[token_idx+1][1]
            nonlocalnertag1 = line[token_idx+1][2]

            features['word+1'] = word1.lower()
            features['pos+1'] = postag1
            features['posbigram+1'] = postag + postag1
            features['pos[-3:]'] = postag1[-3:]
            features['pos[-2:]'] = postag1[-2:]
            features['bigram+1'] = word.lower() + word1.lower()
            features['word+1.isupper'] = word1.isupper()
            features['word+1.istitle'] = word1.istitle()
            features['word+1.isdigit'] = word1.isdigit()
            features['word+1[-3:]'] = word1[-3:]
            features['word+1[-2:]'] = word1[-2:]
        else:
            features['bigram+1'] = word.lower() + "EOL"
            features['EOL'] = 1.0

        if token_idx > 0 and token_idx < len(line)-1:
            word1behind = line[token_idx-1][0]
            word1ahead = line[token_idx+1][0]
            features['trigram-1+1'] = word1behind.lower() + word.lower() + word1ahead.lower()

        if line_idx == 0:
            features['BOD'] = 1.0
        else:
            features['BOD'] = 0.0

        if line_idx == doc_size:
            features['EOD'] = 1.0
        else:
            features['EOD'] = 0.0

        return features

    def sent2features(self, line, line_idx, doc_idx, doc_size):
        return [self.word2features(line, token_idx, line_idx, doc_idx, doc_size) for token_idx in range(len(line))]

    def sent2labels(self, sent):
        labels = []
        for token, pos_tag, nonlocalne, label in sent:
            labels.append(label)
        return labels

    def sent2tokens(self, sent):
        return [token for token, pos_tag, nonlocalne, label in sent]

    def doc2features(self, doc_idx, doc):
        return [self.sent2features(doc[line_idx], line_idx, doc_idx, len(doc)-1) for line_idx in range(len(doc))]
        #return [self.sent2features(line) for line in doc]

    def doc2labels(self, doc):
        return [self.sent2labels(sent) for sent in doc]

    def generate_features(self):
        self.current_dataset = self.train_sents
        self.X_train = [self.doc2features(doc_idx, d) for doc_idx, d in enumerate(self.train_sents)]
        self.y_train = [self.doc2labels(d) for d in self.train_sents]

        self.current_dataset = self.test_sents
        self.X_test = [self.doc2features(doc_idx, s) for doc_idx, s in enumerate(self.test_sents)]
        self.y_test = [self.doc2labels(s) for s in self.test_sents]
        print("Features created for train and test data")

    def train_model(self):
        trainer = pycrfsuite.Trainer(verbose=True)
        print("pycrfsuite Trainer init")

        # transform data structure to group tokens by lines
        for doc_x, doc_y in zip(self.X_train, self.y_train):
            for line_idx, line in enumerate(doc_x):
                trainer.append(line, doc_y[line_idx])

        print("pycrfsuite Trainer has data")

        trainer.set_params({
            'c1': 0.1,   # coefficient for L1 penalty
            'c2': 0.1,  # coefficient for L2 penalty
            'max_iterations': 300,  # stop earlier

            # include states features that do not even occur in the training
            # data, crfsuite creates all possible associations between
            # attirbutes and labels - enabling improves label accuracy
            'feature.possible_states': True,
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        print(trainer.params())
        trainer.train('test_NER.crfsuite')

    def basic_classification_report(self, y_true, y_pred):
        lb = LabelBinarizer()

        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_)
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(y_true_combined, y_pred_combined, labels = [class_indices[cls] for cls in tagset], target_names = tagset)

    def load_tagger(self):
        self.__trained_tagger = pycrfsuite.Tagger()
        self.__trained_tagger.open('test_NER.crfsuite')

        self.w2v_model = word2vec.Word2Vec.load(self.__w2v_model_name)

    # doc: in format of tagged tuples
    def tag_doc(self, doc):
        feature_input = [self.doc2features(doc_idx, d) for doc_idx, d in enumerate([doc])]
        xseq = []
        for line_idx, line in enumerate(feature_input[0]):
            for token_idx, token in enumerate(line):
                xseq.append(token)

        predicted_tags = self.__trained_tagger.tag(xseq)

        # TODO change this to take in doc and not xseq (convert predicted tags
        # to the structure of doc)
        return self.interpret_predicted_tags(xseq, predicted_tags)

    def interpret_predicted_tags(self, doc, tags):
        identified_entities = []
        for tag_idx, tag in enumerate(tags):
            if tag in Tags.__start_tagset:

                entity_found = ""
                while True:
                    if tags[tag_idx] == Tags.__outside_tag:
                        break
                    entity_found = entity_found + " " + doc[token_idx]['word']
                    tag_idx += 1

                identified_entities.append(entity_found, tags[token_idx])

        return identified_entities

    def test_model(self):
        #predictions
        tagger = pycrfsuite.Tagger()
        tagger.open('test_NER.crfsuite')

        # train data
        docs_x_test = []
        docs_y_test_true = []

        for doc_x, doc_y in zip(self.X_train, self.y_train):
            xseq = []
            yseq = []
            for line_idx, line in enumerate(doc_x):
                for token_idx, token in enumerate(line):
                    xseq.append(token)
                    yseq.append(doc_y[line_idx][token_idx])
            docs_x_test.append(xseq)
            docs_y_test_true.append(yseq)

        y_pred = [tagger.tag(doc) for doc in docs_x_test]

        print("Training set:")
        print(self.basic_classification_report(docs_y_test_true, y_pred))

        # test data
        docs_x_test = []
        docs_y_test_true = []

        for doc_x, doc_y in zip(self.X_test, self.y_test):
            xseq = []
            yseq = []
            for line_idx, line in enumerate(doc_x):
                for token_idx, token in enumerate(line):
                    xseq.append(token)
                    yseq.append(doc_y[line_idx][token_idx])
            docs_x_test.append(xseq)
            docs_y_test_true.append(yseq)

        y_pred = [tagger.tag(doc) for doc in docs_x_test]

        print("Test set:")
        print(self.basic_classification_report(docs_y_test_true, y_pred))
