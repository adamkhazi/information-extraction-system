import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from logger import Logger
from dataset import Dataset
from crf_suite import CrfSuite
from we_model import WeModel
from feature_generator import FeatureGenerator

# Class evaluates an already trained model using ROC analysis.
# Can also used bootstrapping to sample the dataset and train the model
class Evaluator():
    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("created evaluator")

    # used an already trained model and perform ROC analysis with the dataset
    def perform_roc_analysis(self):
        raise NotImplementedError

    def perform_bootstrapping(self, dataset, sample_size, iterations):
        training_scores = []
        test_scores = []
        for x in range(0, iterations):
            sampled_train_set, oob_test_set = self.resample_data(dataset, sample_size, return_leftovers=True)
            cs = CrfSuite()
            ds = Dataset()
            we_model = WeModel()
            w2v_model = we_model.train(sampled_train_set) # optionally load a pretrained model here 
            word2count, word2idx = ds.encode_dataset(sampled_train_set)

            f_generator = FeatureGenerator(w2v_model, word2count, word2idx)

            train_features = f_generator.generate_features_docs(sampled_train_set)
            y_train = f_generator.generate_true_outcome(sampled_train_set)

            test_features = f_generator.generate_features_docs(oob_test_set)
            y_test = f_generator.generate_true_outcome(oob_test_set)

            model_name = "test_NER.crfsuite"
            trainer = cs.train_model(train_features, y_train, model_name)
            y_train_pred = cs.test_model(model_name, train_features, y_train)
            y_test_pred = cs.test_model(model_name, test_features, y_test)

            score_train = cs.score_model(ds.docs2lines(y_train), y_train_pred)
            score_test = cs.score_model(ds.docs2lines(y_test), y_test_pred)

            training_scores.append(score_train)
            test_scores.append(score_test)

        return training_scores, test_scores

    # returns resampled training and test set
    def resample_data(self, dataset, nr_samples, return_leftovers=False):
        data_rows = [i for i in range(len(dataset))]
        random_rows = [np.random.choice(data_rows) for row in range(nr_samples)]
        if not return_leftovers:
            return dataset[random_rows]
        leftover_rows = filter(lambda x: not x in random_rows, data_rows)
        return [dataset[idx] for idx in random_rows], [dataset[idx] for idx in leftover_rows]
        
