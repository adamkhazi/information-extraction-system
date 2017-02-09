import nltk
import sklearn
import pycrfsuite
import csv
import math
import io
import time
import logging
import copy
import sklearn_crfsuite
import scipy.stats
import pdb

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import matplotlib.pyplot as plt
import numpy as np
import sklearn_crfsuite

from generate_dataset import GenerateDataset
from dataset import Dataset
from logger import Logger
from tags import Tags

class CrfSuite(Tags):
    __seperator = "/"

    def __init__(self):
        self.logger = Logger()

    def train_model(self, X, y, model_name):
        trainer = pycrfsuite.Trainer(verbose=True)
        self.logger.println("pycrfsuite Trainer init")

        count = 0
        # transform data structure to group tokens by lines
        for doc_x, doc_y in zip(X, y):
            for line_idx, line in enumerate(doc_x):
                trainer.append(line, doc_y[line_idx])
            count+=1
            self.logger.print("Added %s documents to trainer" % count)

        self.logger.new_line()
        self.logger.println("pycrfsuite Trainer has data")

        trainer.set_params({
            'c1': 0.001,   # coefficient for L1 penalty
            'c2': 0.001,  # coefficient for L2 penalty
            'max_iterations': 400,  # stop earlier

            # include states features that do not even occur in the training
            # data, crfsuite creates all possible associations between
            # attirbutes and labels - enabling improves label accuracy
            #'feature.possible_states': True,
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        print(trainer.params())
        trainer.train(model_name)
        return trainer

    def score_model_temp(self, est, y_true, y_pred):
        lb = LabelBinarizer()

        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return f1_score(y_true_combined, y_pred_combined, average="weighted", labels = [class_indices[cls] for cls in tagset])

    def score_model(self, y_true, y_pred):
        lb = LabelBinarizer()

        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return f1_score(y_true_combined, y_pred_combined, average="weighted", labels = [class_indices[cls] for cls in tagset])

    def print_classification_report(self, y_true, y_pred):
        lb = LabelBinarizer()

        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_)
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        # TODO return f1 score or other wise here: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        print(classification_report(y_true_combined, y_pred_combined, labels = [class_indices[cls] for cls in tagset], target_names = tagset))

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
                    if tag_idx >= len(tags) or tags[tag_idx] == Tags.__outside_tag:
                        break
                    entity_found = entity_found + " " + doc[token_idx]['word']
                    tag_idx += 1

                identified_entities.append(entity_found, tags[token_idx])

        return identified_entities

    # use an existing model to tag data
    def test_model(self, model_name, features):
        tagger = pycrfsuite.Tagger()
        tagger.open(model_name)
        # transformed data
        docs_x_test = []

        for doc_x in features:
            xseq = []
            for line_idx, line in enumerate(doc_x):
                for token_idx, token in enumerate(line):
                    xseq.append(token)
            docs_x_test.append(xseq)

        y_pred = [tagger.tag(doc) for doc in docs_x_test]
        return y_pred

    def optimise_model(self):
        # prepare data structure
        count = 0
        xseq = []
        yseq = []
        # transform data structure to group tokens by lines
        for doc_x, doc_y in zip(self.X_train, self.y_train):
            for line_idx, line in enumerate(doc_x):
                xseq.append(line)
                yseq.append(doc_y[line_idx])

            count+=1
            self.logger.print("Added %s documents to trainer" % count)

        # define fixed parameters and parameters to search
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        labels = Tags.tag_list
        labels.remove('O')
        # use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

        # search
        rs = RandomizedSearchCV(crf, params_space, cv=3, verbose=1, n_jobs=-1, n_iter=50, scoring=f1_scorer)
        rs.fit(xseq, yseq)
    
    def plot_learning_curve(self, X, y):
        train_sizes=np.linspace(.1, 1.0, 5)
        n_jobs=8
        title = "Learning Curves"
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plt.figure()
        plt.title(title)
        ylim = (0.01, 1.01)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        X_lines = []
        y_lines = []
        for doc_x, doc_y in zip(X, y):
            for line_idx, line in enumerate(doc_x):
                X_lines.append(line)
                y_lines.append(doc_y[line_idx])

        estimator = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, verbose=True)
        custom_scorer = make_scorer(self.score_model, greater_is_better=True)

        #train_sizes, train_scores, test_scores = learning_curve(estimator, X_lines, y_lines, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_lines, y_lines, cv=cv, scoring=custom_scorer, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        return plt
