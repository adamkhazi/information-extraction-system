import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import chain

from logger import Logger
from dataset import Dataset
from crf_suite import CrfSuite
from we_model import WeModel
from feature_generator import FeatureGenerator
from tags import Tags
from extractor import Extractor
from tokeniser import Tokeniser
from annotator import Annotator

# Class evaluates an already trained model using ROC analysis.
# Can also used bootstrapping to sample the dataset and train the model
class Evaluator(Tags):
    __ies_accuracy_test = "ies_accuracy_test"
    __zylon_parser_labels_folder = "zylon_parser_xml"
    __dataset_raw_folder = "dataset_raw_data"
    __seperator = "/"
    __file_ext_xml = ".xml"

    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("created evaluator")

    # used an already trained model and perform ROC analysis with the dataset
    def perform_roc_analysis(self, y_true, y_pred):
        lb = LabelBinarizer()

        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

        n_classes = y_true_combined.shape[1]

        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        fpr = dict()
        tpr = dict()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_combined[:, i], y_pred_combined[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_combined.ravel(), y_pred_combined.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        lw =  2
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    def entity_scorer(self, y_true, y_pred, entity_tag):
        """
        returns [precision, recall, f1-score] for an entity
        """
        lb = LabelBinarizer()

        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        for tag in Tags.tag_list:
            if tag.split('-', 1)[::-1][0] != entity_tag: # remove tag if not currently asked for
                tagset = tagset - {tag}

        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
        f1_s = f1_score(y_true_combined, y_pred_combined, average="weighted", labels = [class_indices[cls] for cls in tagset])
        precision = precision_score(y_true_combined, y_pred_combined, average="weighted", labels = [class_indices[cls] for cls in tagset])
        recall = recall_score(y_true_combined, y_pred_combined, average="weighted", labels = [class_indices[cls] for cls in tagset])

        return np.array([precision, recall, f1_s], dtype='float64')

    def perform_bootstrapping(self, dataset, sample_size, iterations):
        """
        bootstraps a sample n times. Averages the precision, recall, f1, tpr and
        fpr for each of the entities. Prints results of precision, recall and
        f1. Plots roc curves for tpr and fpr of each entity.
        """
        training_scores = []
        test_scores = []

        emp_pos_scores = np.empty(shape=(0,3),dtype='float64')
        emp_comp_scores = np.empty(shape=(0,3),dtype='float64')
        edu_major_scores = np.empty(shape=(0,3),dtype='float64')
        edu_inst_scores = np.empty(shape=(0,3),dtype='float64')

        mean_fpr = np.linspace(0, 1, 100)
        lb = LabelBinarizer()

        emp_pos_tpr = np.empty(shape=(0,3),dtype='float64')
        emp_pos_fpr = np.empty(shape=(0,3),dtype='float64')
        emp_comp_tpr = np.empty(shape=(0,3),dtype='float64')
        emp_comp_fpr = np.empty(shape=(0,3),dtype='float64')
        edu_major_tpr = np.empty(shape=(0,3),dtype='float64')
        edu_major_fpr = np.empty(shape=(0,3),dtype='float64')
        edu_inst_tpr = np.empty(shape=(0,3),dtype='float64')
        edu_inst_fpr = np.empty(shape=(0,3),dtype='float64')

        for x in range(0, iterations):
            sampled_train_set, oob_test_set = self.resample_data(dataset, sample_size, return_leftovers=True)
            cs = CrfSuite()
            ds = Dataset()
            we_model = WeModel()
            w2v_model = we_model.train(dataset) # optionally load a pretrained model here 
            word2count, word2idx = ds.encode_dataset(sampled_train_set)

            f_generator = FeatureGenerator(w2v_model, word2count, word2idx)

            train_features = f_generator.generate_features_docs(sampled_train_set)
            y_train = f_generator.generate_true_outcome(sampled_train_set)

            test_features = f_generator.generate_features_docs(oob_test_set)
            y_test = f_generator.generate_true_outcome(oob_test_set)

            trainer = cs.train_model(train_features, y_train)
            y_train_pred = cs.test_model(trainer, train_features)
            y_test_pred = cs.test_model(trainer, test_features)

            score_train = cs.score_model(ds.docs2lines(y_train), y_train_pred)
            score_test = cs.score_model(ds.docs2lines(y_test), y_test_pred)

            y_true_combined = lb.fit_transform(list(chain.from_iterable(ds.docs2lines(y_test))))
            y_pred_combined = lb.transform(list(chain.from_iterable(y_test_pred)))

            class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

            # fpr and tpr for one class
            temp_fpr, temp_tpr, _ = roc_curve(y_true_combined[:, class_indices["B-EMP-POS"]], y_pred_combined[:, class_indices["B-EMP-POS"]], pos_label=1)
            temp_fpr1, temp_tpr1, _ = roc_curve(y_true_combined[:, class_indices["I-EMP-POS"]], y_pred_combined[:, class_indices["I-EMP-POS"]], pos_label=1)
            temp_fpr = np.vstack([temp_fpr, temp_fpr1])
            temp_tpr = np.vstack([temp_tpr, temp_tpr1])
            emp_pos_tpr = np.vstack([emp_pos_tpr, temp_tpr.mean(axis=0)])
            emp_pos_fpr = np.vstack([emp_pos_fpr, temp_fpr.mean(axis=0)])
            temp_fpr = temp_tpr = temp_fpr1 = temp_tpr1 = np.empty(shape=(0,3),dtype='float64')

            temp_fpr, temp_tpr, _ = roc_curve(y_true_combined[:, class_indices["B-EMP-COMP"]], y_pred_combined[:, class_indices["B-EMP-COMP"]], pos_label=1)
            temp_fpr1, temp_tpr1, _ = roc_curve(y_true_combined[:, class_indices["I-EMP-COMP"]], y_pred_combined[:, class_indices["I-EMP-COMP"]], pos_label=1)
            temp_fpr = np.vstack([temp_fpr, temp_fpr1])
            temp_tpr = np.vstack([temp_tpr, temp_tpr1])
            emp_comp_tpr = np.vstack([emp_comp_tpr, temp_tpr.mean(axis=0)])
            emp_comp_fpr = np.vstack([emp_comp_fpr, temp_fpr.mean(axis=0)])
            temp_fpr = temp_tpr = temp_fpr1 = temp_tpr1 = np.empty(shape=(0,3),dtype='float64')

            temp_fpr, temp_tpr, _ = roc_curve(y_true_combined[:, class_indices["B-EDU-MAJOR"]], y_pred_combined[:, class_indices["B-EDU-MAJOR"]], pos_label=1)
            temp_fpr1, temp_tpr1, _ = roc_curve(y_true_combined[:, class_indices["I-EDU-MAJOR"]], y_pred_combined[:, class_indices["I-EDU-MAJOR"]], pos_label=1)
            temp_fpr = np.vstack([temp_fpr, temp_fpr1])
            temp_tpr = np.vstack([temp_tpr, temp_tpr1])
            edu_major_tpr = np.vstack([edu_major_tpr, temp_tpr.mean(axis=0)])
            edu_major_fpr = np.vstack([edu_major_fpr, temp_fpr.mean(axis=0)])
            temp_fpr = temp_tpr = temp_fpr1 = temp_tpr1 = np.empty(shape=(0,3),dtype='float64')

            temp_fpr, temp_tpr, _ = roc_curve(y_true_combined[:, class_indices["B-EDU-INST"]], y_pred_combined[:, class_indices["B-EDU-INST"]], pos_label=1)
            temp_fpr1, temp_tpr1, _ = roc_curve(y_true_combined[:, class_indices["I-EDU-INST"]], y_pred_combined[:, class_indices["I-EDU-INST"]], pos_label=1)
            temp_fpr = np.vstack([temp_fpr, temp_fpr1])
            temp_tpr = np.vstack([temp_tpr, temp_tpr1])
            edu_inst_tpr = np.vstack([edu_inst_tpr, temp_tpr.mean(axis=0)])
            edu_inst_fpr = np.vstack([edu_inst_fpr, temp_fpr.mean(axis=0)])

            emp_pos_scores = np.vstack([emp_pos_scores, self.entity_scorer(ds.docs2lines(y_test), y_test_pred, "EMP-POS")])
            emp_comp_scores = np.vstack([emp_comp_scores, self.entity_scorer(ds.docs2lines(y_test), y_test_pred, "EMP-COMP")])
            edu_major_scores = np.vstack([edu_major_scores, self.entity_scorer(ds.docs2lines(y_test), y_test_pred, "EDU-MAJOR")])
            edu_inst_scores = np.vstack([edu_inst_scores, self.entity_scorer(ds.docs2lines(y_test), y_test_pred, "EDU-INST")])

        print("EMP-POS")
        print("precision %s" % np.mean(emp_pos_scores[:,0]))
        print("recall %s" % np.mean(emp_pos_scores[:,1]))
        print("f1 %s" % np.mean(emp_pos_scores[:,2]))

        print("EMP-COMP")
        print("precision %s" % np.mean(emp_comp_scores[:,0]))
        print("recall %s" % np.mean(emp_comp_scores[:,1]))
        print("f1 %s" % np.mean(emp_comp_scores[:,2]))

        print("EDU-MAJOR")
        print("precision %s" % np.mean(edu_major_scores[:,0]))
        print("recall %s" % np.mean(edu_major_scores[:,1]))
        print("f1 %s" % np.mean(edu_major_scores[:,2]))

        print("EDU-INST")
        print("precision %s" % np.mean(edu_inst_scores[:,0]))
        print("recall %s" % np.mean(edu_inst_scores[:,1]))
        print("f1 %s" % np.mean(edu_inst_scores[:,2]))


        emp_pos_tpr = emp_pos_tpr.mean(axis=0)
        emp_pos_fpr = emp_pos_fpr.mean(axis=0)

        emp_comp_tpr = emp_comp_tpr.mean(axis=0)
        emp_comp_fpr = emp_comp_fpr.mean(axis=0)

        edu_major_tpr = edu_major_tpr.mean(axis=0)
        edu_major_fpr = edu_major_fpr.mean(axis=0)

        edu_inst_tpr = edu_inst_tpr.mean(axis=0)
        edu_inst_fpr = edu_inst_fpr.mean(axis=0)

        lw=2
        plt.subplot(221)
        plt.plot(emp_pos_fpr, emp_pos_tpr, color='g', linestyle='--', label='EMP-POS', lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")

        plt.subplot(222)
        plt.plot(emp_comp_fpr, emp_comp_tpr, color='g', linestyle='--', label='EMP-COMP', lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")

        plt.subplot(223)
        plt.plot(edu_major_fpr, edu_major_tpr, color='g', linestyle='--', label='EDU-MAJOR', lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")

        plt.subplot(224)
        plt.plot(edu_inst_fpr, edu_inst_tpr, color='g', linestyle='--', label='EDU-INST', lw=lw)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")

        plt.show()

        return emp_pos_scores

    def resample_data(self, dataset, nr_samples, return_leftovers=False):
        """
        Returns resampled training and test set. Uses the OOB Bootstrapping
        method.
        """
        data_rows = [i for i in range(len(dataset))]
        random_rows = [np.random.choice(data_rows) for row in range(nr_samples)]
        if not return_leftovers:
            return dataset[random_rows]
        leftover_rows = filter(lambda x: not x in random_rows, data_rows)
        return [dataset[idx] for idx in random_rows], [dataset[idx] for idx in leftover_rows]

    def get_zylon_parser_scores(self):
        """
        parameters: none

        Extracts labelled entities from zylon's xml output and true xml
        output. Compares the entity lists and returns a score, higher is
        better.
        
        return: edu_insts_match_score, edu_majors_match_score, emp_names_match_score, emp_jtitles_match_score
        """
        extractor = Extractor()
        zylon_filenames = extractor.populate_file_names(self.__zylon_parser_labels_folder)

        zylon_xml_trees = extractor.read_resume_labels(self.__zylon_parser_labels_folder, zylon_filenames)
        true_xml_trees = extractor.read_resume_labels(self.__dataset_raw_folder, zylon_filenames)

        true_edu_insts = [extractor.get_edu_institutions(xml_tree) for xml_tree in true_xml_trees]
        true_edu_majors = [extractor.get_edu_majors(xml_tree) for xml_tree in true_xml_trees]
        true_emp_names = [extractor.get_company_names(xml_tree) for xml_tree in true_xml_trees]
        true_emp_jtitles = [extractor.get_job_titles(xml_tree) for xml_tree in true_xml_trees]

        zylon_edu_insts = [extractor.get_edu_institutions_zy(xml_tree) for xml_tree in zylon_xml_trees]
        zylon_edu_majors = [extractor.get_edu_majors_zy(xml_tree) for xml_tree in zylon_xml_trees]
        zylon_emp_names = [extractor.get_company_names_zy(xml_tree) for xml_tree in zylon_xml_trees]
        zylon_emp_jtitles = [extractor.get_job_titles_zy(xml_tree) for xml_tree in zylon_xml_trees]

        tokeniser = Tokeniser()
        true_edu_insts = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_edu_insts))
        true_edu_majors = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_edu_majors))
        true_emp_names = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_emp_names))
        true_emp_jtitles = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_emp_jtitles))

        zylon_edu_insts = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(zylon_edu_insts))
        zylon_edu_majors = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(zylon_edu_majors))
        zylon_emp_names = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(zylon_emp_names))
        zylon_emp_jtitles = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(zylon_emp_jtitles))

        edu_insts_match_score = self.score_matches(zylon_edu_insts, true_edu_insts)
        edu_majors_match_score = self.score_matches(zylon_edu_majors, true_edu_majors)
        emp_names_match_score = self.score_matches(zylon_emp_names, true_emp_names)
        emp_jtitles_match_score = self.score_matches(zylon_emp_jtitles, true_emp_jtitles)

        return edu_insts_match_score, edu_majors_match_score, emp_names_match_score, emp_jtitles_match_score

    def score_matches(self, obtained_list, true_list):
        """
        parameters:  obtained_list - is the predicted or obtained entity list for each document 
        parameters:  true_list - is the true or correct list of entities for each document

        A point is awarded for each entity in the obtained list that occurs in
        the true list for each document.

        return: number of entities matched correctly/total entities in true list
        """
        matches = [True for idx, doc in enumerate(obtained_list) for entity in doc for true_entity in true_list[idx] if true_entity == entity]
        total_score = [True for doc in true_list for entity in doc]
        return len(matches)/len(total_score)

    def get_ies_scores(self):
        extractor = Extractor()
        ies_filenames = extractor.populate_file_names(self.__ies_accuracy_test)
        ies_filenames = extractor.filter_by_valid_exts(ies_filenames)
        filenames, resume_content = extractor.read_resume_content_tika_api(ies_filenames, self.__ies_accuracy_test)
        filenames, resume_content = extractor.remove_empty_resumes(filenames, resume_content)
        resume_labels = extractor.read_resume_labels(self.__ies_accuracy_test, filenames)

        true_edu_insts = [extractor.get_edu_institutions(xml_tree) for xml_tree in resume_labels]
        true_edu_majors = [extractor.get_edu_majors(xml_tree) for xml_tree in resume_labels]
        true_emp_names = [extractor.get_company_names(xml_tree) for xml_tree in resume_labels]
        true_emp_jtitles = [extractor.get_job_titles(xml_tree) for xml_tree in resume_labels]

        cs = CrfSuite()
        cs.load_tagger()
        annotator = Annotator()
        annotated_resumes = [annotator.annotate_using_trained_model(self.__ies_accuracy_test +self.__seperator + filename[0] + filename[1]) for filename in filenames]
        predicted_entity_list = [cs.tag_doc(resume) for resume in annotated_resumes]

        ies_edu_insts = [extractor.get_edu_institutions_from_list(entity_list) for entity_list in predicted_entity_list]
        ies_edu_majors = [extractor.get_edu_major_from_list(entity_list) for entity_list in predicted_entity_list]
        ies_emp_names = [extractor.get_company_names_from_list(entity_list) for entity_list in predicted_entity_list]
        ies_emp_jtitles = [extractor.get_company_position_from_list(entity_list) for entity_list in predicted_entity_list]

        tokeniser = Tokeniser()
        true_edu_insts = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_edu_insts))
        true_edu_majors = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_edu_majors))
        true_emp_names = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_emp_names))
        true_emp_jtitles = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(true_emp_jtitles))

        ies_edu_insts = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(ies_edu_insts))
        ies_edu_majors = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(ies_edu_majors))
        ies_emp_names = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(ies_emp_names))
        ies_emp_jtitles = tokeniser.docs_tolower(tokeniser.tokenise_doclines_to_words(ies_emp_jtitles))

        edu_insts_match_score = self.score_matches(ies_edu_insts, true_edu_insts)
        edu_majors_match_score = self.score_matches(ies_edu_majors, true_edu_majors)
        emp_names_match_score = self.score_matches(ies_emp_names, true_emp_names)
        emp_jtitles_match_score = self.score_matches(ies_emp_jtitles, true_emp_jtitles)
        print(edu_insts_match_score)
        print(edu_majors_match_score)
        print(emp_names_match_score)
        print(emp_jtitles_match_score)
