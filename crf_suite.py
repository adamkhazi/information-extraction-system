from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import csv
import math
import io

from generate_dataset import GenerateDataset

class CrfSuite:

    def get_dataset(self):
        gd = GenerateDataset()
        self.total_sents = gd.read_tagged_tokens()
        print("Read " + str(len(self.total_sents)) + " documents")

    def split_dataset(self):
        split_point = math.ceil(len(self.total_sents) * 0.7)
        self.train_sents = self.total_sents[0:split_point]
        self.test_sents = self.total_sents[split_point+1:]
        print("Split dataset")

    def first_letter_upper(self, token):
        return token[0].isupper()

    def word2features(self, line, token_idx, line_idx, doc_size):
        word = line[token_idx][0]
        postag = line[token_idx][1]
        nonlocalnertag = line[token_idx][2]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'word.firstletterupper=%s' % self.first_letter_upper(word),
            'word.idx=' + str(token_idx),
            'postag=' + postag,
            'postag[:2]=' + postag[:2],
            'nonlocalnertag=' + nonlocalnertag,
            'line.idx=' + str(line_idx),
        ]

        if line_idx == 0:
            features.append('BOD')

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

        return features

    def sent2features(self, line, line_idx, doc_size):
        return [self.word2features(line, token_idx, line_idx, doc_size) for token_idx in range(len(line))]

    def sent2labels(self, sent):
        return [label for token, postag, nonlocalnertag, label in sent]

    def sent2tokens(self, sent):
        return [token for token, postag, nonlocalnertag, label in sent]

    def doc2features(self, doc):
        return [self.sent2features(doc[line_idx], line_idx, len(doc)-1) for line_idx in range(len(doc))]
        #return [self.sent2features(line) for line in doc]

    def doc2labels(self, doc):
        return [self.sent2labels(sent) for sent in doc]

    def generate_features(self):
        self.X_train = [self.doc2features(d) for d in self.train_sents]
        self.y_train = [self.doc2labels(d) for d in self.train_sents]

        self.X_test = [self.doc2features(s) for s in self.test_sents]
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
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        print(trainer.params())
        trainer.train('test_NER.crfsuite')

    def basic_classification_report(self, y_true, y_pred):
        lb = LabelBinarizer()

        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        return classification_report(y_true_combined, y_pred_combined)

    def test_model(self):
        #predictions
        tagger = pycrfsuite.Tagger()
        tagger.open('test_NER.crfsuite')
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
