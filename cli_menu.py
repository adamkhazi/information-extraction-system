import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np

from feature_generator import FeatureGenerator
from crf_suite import CrfSuite
from generate_dataset import GenerateDataset
from annotator import Annotator
from api import API
from logger import Logger
from evaluator import Evaluator
from dataset import Dataset
from we_model import WeModel

class CliMenu():
    __argument_train = "-t"
    __argument_optimise = "-o"
    __argument_annotate_dataset = "-a"
    __argument_api = "-rn"
    __argument_evaluate = "-e"
    __argument_train_w_learning_curve = "-lc"
    __argument_evaluate_zylon = "-e_zylon"
    __argument_accuracy_normal_ies = "-an"
    __argument_draw_roc_curve_saved_model = "-rsm"

    def __init__(self):
        self.logger = Logger()

    def perform_command(self):
        command_arg = sys.argv[1]

        if command_arg == self.__argument_train:
            if len(sys.argv) > 2:
                self.train_model(nr_of_files=int(sys.argv[2]))
            else:
                self.train_model()

        elif command_arg == self.__argument_train_w_learning_curve:
            self.train_model_learning_curve(int(sys.argv[2]))

        elif command_arg == self.__argument_optimise:
            self.optimise_model(int(sys.argv[2]))

        elif command_arg == self.__argument_annotate_dataset:
            if len(sys.argv) > 2:
                self.annotate_data(nr_docs=int(sys.argv[2]))
            else:
                self.annotate_data()

        elif command_arg == self.__argument_api:
            self.run_api()

        elif command_arg == self.__argument_draw_roc_curve_saved_model:
            self.draw_roc_curve_saved_model()

        elif command_arg == self.__argument_evaluate:
            if len(sys.argv) > 2:
                self.evaluate_model(sys.argv[2])
            else:
                print("evaluate model option needs an extra parameter")

        elif command_arg == self.__argument_evaluate_zylon:
            self.evaluate_zylon()

        elif command_arg == self.__argument_accuracy_normal_ies:
            self.ies_normal_accuracy_scores()

        else:
            print("Commands accepted:")
            print("train: -t <number_of_documents>(default is all available documents)")
            print("hyperparameter optimisation: -o")
            print("annotate dataset: -a <number_of_documents>(default is all available documents")
            print("run api: -rn")
            print("evaluate model: -e [-b train and analyse performance using bootstrapping]|[-r perform roc analysis]")

    def annotate_db_data(self):
        raise NotImplementedError

    def annotate_data(self, nr_docs=-1):
        self.logger.println("data annotator called")
        start_time = timeit.default_timer()
        annotator = Annotator()
        annotator.prepare_dataset(nr_docs)
        elapsed_seconds = timeit.default_timer() - start_time
        self.logger.print_time_taken("data annotation operation took", elapsed_seconds)

    def optimise_model(self, argv):
        self.logger.println("optimise model called")
        start_time = timeit.default_timer()

        cs = CrfSuite()

        dataset = Dataset()
        data = dataset.read(nr_of_files=argv)

        we_model = WeModel()
        w2v_model = we_model.train(data) # optionally load a pretrained model here 
        we_model.save(w2v_model)

        word2count, word2idx = dataset.encode_dataset(data)

        f_generator = FeatureGenerator(w2v_model, word2count, word2idx)
        train_features = f_generator.generate_features_docs(data)
        y_train = f_generator.generate_true_outcome(data)

        cs.optimise_model(train_features, y_train)

        elapsed_seconds = timeit.default_timer() - start_time
        self.logger.print_time_taken("optimise model operation took", elapsed_seconds)

    def train_model(self, nr_of_files=-1):
        self.logger.println("train model called")
        start_time = timeit.default_timer()
        cs = CrfSuite()

        dataset = Dataset()
        data = dataset.read(nr_of_files=nr_of_files)
        nr_of_filled_lines, data1 = dataset.filter_for_filled_tags(data)
        data2 = dataset.obtain_default_tags(nr_of_filled_lines*3, data)
        data = data1 + data2
        data = dataset.shuffle_data(data)
        train_set, test_set = dataset.split_dataset(data)

        we_model = WeModel()
        w2v_model = we_model.train(data) # optionally load a pretrained model here 
        we_model.save(w2v_model)
        we_model = None

        word2count, word2idx = dataset.encode_dataset(train_set)

        f_generator = FeatureGenerator(w2v_model, word2count, word2idx)
        w2v_model = None
        train_features = f_generator.generate_features_docs(train_set)
        y_train = f_generator.generate_true_outcome(train_set)

        test_features = f_generator.generate_features_docs(test_set)
        y_test = f_generator.generate_true_outcome(test_set)
        f_generator = None

        model = cs.train_model(train_features, y_train)
        cs.save_model(model)
        y_train_pred = cs.test_model(model, train_features)
        y_test_pred = cs.test_model(model, test_features)

        print("printing training results")
        cs.print_classification_report(dataset.docs2lines(y_train), y_train_pred)
        score_train = cs.score_model(dataset.docs2lines(y_train), y_train_pred)
        print("training f1 score: %s" % score_train)

        print("printing test results")
        cs.print_classification_report(dataset.docs2lines(y_test), y_test_pred)
        score_test = cs.score_model(dataset.docs2lines(y_test), y_test_pred)
        print("test f1 score: %s" % score_test)

        elapsed_seconds = timeit.default_timer() - start_time
        self.logger.print_time_taken("train model operation took", elapsed_seconds)

        evaluator = Evaluator()
        evaluator.perform_roc_analysis(dataset.docs2lines(y_train), y_train_pred)
        evaluator.perform_roc_analysis(dataset.docs2lines(y_test), y_test_pred)

    def run_api(self):
        self.logger.println("api called")
        api = API()
        api.run()

    def evaluate_model(self, arg):
        self.logger.println("train model called")
        start_time = timeit.default_timer()

        self.logger.println("evaluate model called")
        evaluator = Evaluator()
        dataset = Dataset()
        data = dataset.read(-1)
        nr_of_filled_lines, data1 = dataset.filter_for_filled_tags(data)
        data2 = dataset.obtain_default_tags(nr_of_filled_lines*3, data)
        data = data1 + data2
        data = dataset.shuffle_data(data)

        emp_pos, emp_comp, edu_inst, edu_major = evaluator.perform_bootstrapping(data, len(data), 100)

        #print("test scores")
        print("saving scores to results:")

        np.savetxt('results/emp_pos.txt', emp_pos)
        np.savetxt('results/emp_comp.txt', emp_comp)
        np.savetxt('results/edu_inst.txt', edu_inst)
        np.savetxt('results/edu_major.txt', edu_major)

        elapsed_seconds = timeit.default_timer() - start_time
        self.logger.print_time_taken("train model operation took", elapsed_seconds)

    def train_model_learning_curve(self, arg):
        self.logger.println("train model called")
        start_time = timeit.default_timer()

        cs = CrfSuite()

        dataset = Dataset()
        data = dataset.read(nr_of_files=arg)
        nr_of_filled_lines, data1 = dataset.filter_for_filled_tags(data)
        data2 = dataset.obtain_default_tags(nr_of_filled_lines*3, data)
        data = data1 + data2
        data = dataset.shuffle_data(data)
        train_set, test_set = dataset.split_dataset(data)

        we_model = WeModel()
        w2v_model = we_model.train(data) # optionally load a pretrained model here 
        #w2v_model = we_model.load_pretrained_model() # optionally load a pretrained model here 
        word2count, word2idx = dataset.encode_dataset(train_set)

        f_generator = FeatureGenerator(w2v_model, word2count, word2idx)
        train_features = f_generator.generate_features_docs(train_set)
        y_train = f_generator.generate_true_outcome(train_set)

        cs.plot_learning_curve(train_features, y_train)
        plt.show()

        elapsed_seconds = timeit.default_timer() - start_time
        self.logger.print_time_taken("train model operation took", elapsed_seconds)

    def evaluate_zylon(self):
        self.logger.println("train model called")
        start_time = timeit.default_timer()

        evaluator = Evaluator()
        scores = evaluator.get_zylon_parser_scores()
        print(scores)

        elapsed_seconds = timeit.default_timer() - start_time
        self.logger.print_time_taken("train model operation took", elapsed_seconds)

    def ies_normal_accuracy_scores(self):
        self.logger.println("normal accuracy scores ies called")
        evaluator = Evaluator()
        evaluator.get_ies_scores()

    def draw_roc_curve_saved_model(self):
        self.logger.println("drawing roc curve from saved model")
        start_time = timeit.default_timer()
        cs = CrfSuite()
        crf = cs.load_model("current_crf_model.pkl")

        dataset = Dataset()
        data = dataset.read(nr_of_files=1000)
        nr_of_filled_lines, data1 = dataset.filter_for_filled_tags(data)
        data2 = dataset.obtain_default_tags(nr_of_filled_lines*3, data)
        data = data1 + data2
        data = dataset.shuffle_data(data)
        train_set, test_set = dataset.split_dataset(data)

        we_model = WeModel()
        w2v_model = we_model.read()
        we_model = None

        word2count, word2idx = dataset.encode_dataset(train_set)

        f_generator = FeatureGenerator(w2v_model, word2count, word2idx)
        w2v_model = None
        train_features = f_generator.generate_features_docs(train_set)
        y_train = f_generator.generate_true_outcome(train_set)

        test_features = f_generator.generate_features_docs(test_set)
        y_test = f_generator.generate_true_outcome(test_set)
        f_generator = None

        evaluator = Evaluator()
        evaluator.draw_roc_proba(crf, test_features, y_test)


if __name__ == '__main__':
    CliMenu().perform_command()
