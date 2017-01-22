from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import csv
import math
import io
import time
import logging
import copy
import gensim
from gensim.models import word2vec

from generate_dataset import GenerateDataset
from dataset import Dataset

class CrfSuite:
    __seperator = "/"
    __pre_trained_models_folder = "pre_trained_models"
    __google_news_word2vec = "GoogleNews-vectors-negative300.bin.gz"

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

    # n-gram generation
    def encode_dataset(self):
        self.word2idx = {}
        self.pos_tag2idx = {}
        self.nonlocal_ner_tag2idx = {}
        self.ner_tag2idx = {}

        self.word_idx = 0
        #self.pos_tag_idx = 0
        #self.nonlocal_ner_tag_idx = 0
        self.tag_idx = 0

        for doc_idx, doc in enumerate(self.total_sents):
            for line_idx, line in enumerate(doc):
                for token_idx, token in enumerate(line):
                    word = token[0].lower()
                    #pos_tag = token[1]
                    #nonlocal_ner_tag = token[2]
                    ner_tag = token[1]

                    if word not in self.word2idx:
                        self.word2idx[word] = self.word_idx
                        self.word_idx += 1
                    """
                    if pos_tag not in self.pos_tag2idx:
                        self.pos_tag2idx[pos_tag] = self.pos_tag_idx
                        self.pos_tag_idx += 1
                    if nonlocal_ner_tag not in self.nonlocal_ner_tag2idx:
                        self.nonlocal_ner_tag2idx[nonlocal_ner_tag] = self.nonlocal_ner_tag_idx
                        self.nonlocal_ner_tag_idx += 1
                        """
                    if ner_tag not in self.ner_tag2idx:
                        self.ner_tag2idx[ner_tag] = self.tag_idx
                        self.tag_idx += 1

        """
        self.bigram2idx = {}
        self.bigram_idx = 0

        for doc_idx, doc in enumerate(self.total_sents):
            bigram_prev_word = "START"
            for line_idx, line in enumerate(doc):
                for token_idx, token in enumerate(line):
                    current_word = token[0].lower()
                    bigram = bigram_prev_word + ' ' + current_word
                    bigram_prev_word = current_word
                    if bigram in self.bigram2idx:
                        self.bigram2idx[bigram] += 1
                    else:
                        self.bigram2idx[bigram] = 1
                    if bigram not in self.bigram2idx:
                        self.bigram2idx[bigram] = self.bigram_idx
                        self.bigram_idx += 1
                        """
        print("encode_dataset: " + str(self.word_idx) + " unique words")
        #print("encode_dataset: " + str(self.bigram_idx) + " unique bigrams")

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

    def first_letter_upper(self, token):
        return token[0].isupper()

    def word2features(self, line, token_idx, line_idx, doc_idx, doc_size):
        word = line[token_idx][0]
        postag = line[token_idx][1]
        #nonlocalnertag = line[token_idx][2]

        """
        bigram = ''
        if line_idx == 0 and token_idx == 0:
            bigram = "START" + ' ' + word.lower()
        elif token_idx == 0 and line_idx != 0:
            # previous line last token
            prev_word = self.current_dataset[doc_idx][line_idx-1][len(self.current_dataset[doc_idx][line_idx-1])-1][0].lower()
            bigram = prev_word + ' ' + word.lower()
        else:
            prev_word = line[token_idx-1][0].lower()
            bigram = prev_word + ' ' + word.lower()
        """

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
                #'word_idx': float(self.word2idx[word.lower()])
                #"word_idx": self.model[word.lower()],
                #"bigram_idx_count": self.bigram2idx[bigram],
                #"pos_idx": self.pos_tag2idx[postag]
        }


        try:
            #print("trying" + " " + word)
            for d_idx, dimension in enumerate(self.w2v_model[word.lower()]):
                features["we_dimen_"+str(d_idx)+":"] = dimension
        except KeyError:
            features["we_dimen_"+str(0)+":"] = "UNKNOWN"


        #print("word: " + word.lower() + " bigram: " + bigram.lower())
        #time.sleep(1)

        if token_idx > 0:
            word1 = line[token_idx-1][0]
            postag1 = line[token_idx-1][1]
            #nonlocalnertag1 = line[token_idx-1][2]

            #features['nl-1'] = self.nonlocal_ner_tag2idx[nonlocalnertag1]
            #features['p-1'] = self.pos_tag2idx[postag1]
            #features['word-1'] = float(self.word2idx[word1.lower()])
            features['word-1'] = word1.lower()
            features['pos-1'] = postag1
            features['pos[-3:]'] = postag1[-3:]
            features['pos[-2:]'] = postag1[-2:]
            features['BOL'] = 0.0
            """
            features['word-1[-3:]'] = word1[-3:],
            features['word-1[-2:]'] = word1[-2:],
            features['word-1.isupper'] = word1.isupper(),
            features['word-1.istitle'] = word1.istitle(),
            features['word-1.isdigit'] = word1.isdigit(),
            features['word-1.idx']= float(token_idx-1)
            """
        else:
            features['BOL'] = 1.0

        if token_idx < len(line)-1:
            word1 = line[token_idx+1][0]
            postag1 = line[token_idx+1][1]
            #nonlocalnertag1 = line[token_idx+1][2]

            #features['nl+1'] = self.nonlocal_ner_tag2idx[nonlocalnertag1]
            #features['p+1'] = self.pos_tag2idx[postag1]
            #features['word+1'] = float(self.word2idx[word1.lower()])
            features['word+1'] = word1.lower()
            features['pos+1'] = postag1
            features['pos[-3:]'] = postag1[-3:]
            features['pos[-2:]'] = postag1[-2:]
            features['EOL'] = 0.0
            """
            features['word+1[-3:]'] = word1[-3:],
            features['word+1[-2:]'] = word1[-2:],
            features['word+1.isupper'] = word1.isupper(),
            features['word+1.istitle'] = word1.istitle(),
            features['word+1.isdigit'] = word1.isdigit(),
            features['word+1.idx']= float(token_idx+1)
            """
        else:
            features['EOL'] = 1.0

        """
        if token_idx > 1:
            word2 = line[token_idx-2][0]
            postag2 = line[token_idx-2][1]
            nonlocalnertag2 = line[token_idx-2][2]

            features['nl-2'] = self.nonlocal_ner_tag2idx[nonlocalnertag2]
            features['p-2'] = self.pos_tag2idx[postag2]
            features['w-2'] = self.word2idx[word2.lower()]

        if token_idx < len(line)-2:
            word2 = line[token_idx+2][0]
            postag2 = line[token_idx+2][1]
            nonlocalnertag2 = line[token_idx+2][2]

            features['nl+2'] = self.nonlocal_ner_tag2idx[nonlocalnertag2]
            features['p+2'] = self.pos_tag2idx[postag2]
            features['w+2'] = self.word2idx[word2.lower()]
            """

        if line_idx == 0:
            features['BOD'] = 1.0
        else:
            features['BOD'] = 0.0

        if line_idx == doc_size:
            features['EOD'] = 1.0
        else:
            features['EOD'] = 0.0

        """
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'word.idx=' + str(token_idx),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
            'nonlocalnertag=' + nonlocalnertag,
            'line.idx='+ str(line_idx)
        ]

        if token_idx > 0:
            word1 = line[token_idx-1][0]
            postag1 = line[token_idx-1][1]
            nonlocalnertag1 = line[token_idx-1][2]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1word.firstletterupper=%s' % self.first_letter_upper(word1),
                '-1word.idx=' + str(token_idx-1),
                '-1:postag=' + postag1,
                '-1:postag[:2]=' + postag1[:2],
                '-1:nonlocalnertag=' + nonlocalnertag,
            ])
        else:
            features.append('BOL')

        if token_idx < len(line)-1:
            word1 = line[token_idx+1][0]
            postag1 = line[token_idx+1][1]
            nonlocalnertag1 = line[token_idx+1][2]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                '+1word.firstletterupper=%s' % self.first_letter_upper(word1),
                '+1word.idx=' + str(token_idx+1),
                '+1:postag=' + postag1,
                '+1:postag[:2]=' + postag1[:2],
                '+1:nonlocalnertag=' + nonlocalnertag,
            ])

        else:
            features.append('EOL')

        if line_idx == doc_size:
            features.append('EOD')
        """
        return features

    def sent2features(self, line, line_idx, doc_idx, doc_size):
        return [self.word2features(line, token_idx, line_idx, doc_idx, doc_size) for token_idx in range(len(line))]

    def sent2labels(self, sent):
        labels = []
        for token, pos_tag, label in sent:
            labels.append(label)
        return labels

    def sent2tokens(self, sent):
        return [token for token, pos_tag, label in sent]

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

        for doc_x, doc_y in zip(self.X_train, self.y_train):
            xseq = []
            yseq = []
            for line_idx, line in enumerate(doc_x):
                for token_idx, token in enumerate(line):
                    xseq.append(token)
                    yseq.append(doc_y[line_idx][token_idx])
            trainer.append(xseq, yseq)
        print("pycrfsuite Trainer has data")

        trainer.set_params({
            'c1': 0.01,   # coefficient for L1 penalty
            'c2': 0.01,  # coefficient for L2 penalty
            'max_iterations': 300,  # stop earlier

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


"""
print("printint last iteration")
print(trainer.logparser.last_iteration)

print(str(len(trainer.logparser.iterations)) + ", " +
        str(trainer.logparser.iterations[-1]))

#predictions
tagger = pycrfsuite.Tagger()
tagger.open('test_NER.crfsuite')

# example tag
example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))

    #evaluate model
    def bio_classification_report(y_true, y_pred):

        
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.
        
        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
       

        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
            
        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
        
        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
        )

y_pred = [tagger.tag(xseq) for xseq in X_test]

def basic_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    return classification_report(y_true_combined, y_pred_combined)

print(basic_classification_report(y_test, y_pred))

#print(bio_classification_report(y_test, y_pred))
"""
