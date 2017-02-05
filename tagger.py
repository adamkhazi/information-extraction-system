import untangle
import pdb
import nltk
import copy
import os

from os.path import expanduser
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger
from nltk.internals import find_jars_within_path
from nltk.corpus import stopwords

from logger import Logger

class Tagger:
    __outside_tag = "O"
    __begin_tag_prefix = "B-"
    __inside_tag_prefix = "I-"
    __empty_string = ""

    __pos_tag_tuple_idx = 1
    __nonlocal_ne_tag_tuple_idx = 2
    __ner_tag_tuple_idx = 3

    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("tagger created")

        # setup st tagger once so pickle file not reloaded
        self.setup_nonlocal_tagger()

    # add tuples in place of tokens
    def prepare_doc(self, doc):
        for line_idx, line in enumerate(doc):
            for token_idx, token in enumerate(line):
                # token, postag, nonlocal tag, ner tag
                doc[line_idx][token_idx] = (token, "", "", "")
        return doc

    # match a single label against a document
    def match_label(self, doc, label_str, match_tag):
        content_flat_list, length_of_lines = self.flat_token_list_transform(doc)
        content_flat_list_len = len(content_flat_list)

        label_str = label_str.lower() #lowercase for comparison
        label_str = word_tokenize(label_str)
        label_str = [word for word in label_str if word not in stopwords.words('english')]
        label_str_len = len(label_str)

        idx = max(0, label_str_len-1)
        while idx < content_flat_list_len:
            trailing_window_len = idx - (label_str_len-1)
            comparison = content_flat_list[trailing_window_len:idx+1]
            comparison = [item[0].lower() for item in comparison] # first position is token

            idx += 1
            if comparison == label_str:
                # found
                is_existing_tag = False
                for matches_idx in range(trailing_window_len, idx):
                    if content_flat_list[matches_idx][3] != self.__outside_tag and content_flat_list[matches_idx][3] != self.__empty_string:
                        is_existing_tag |= True

                if not is_existing_tag:
                    for matches_idx in range(trailing_window_len, idx):
                        current_tag = ""
                        if matches_idx == trailing_window_len:
                            current_tag = self.__begin_tag_prefix + match_tag
                        else:
                            current_tag = self.__inside_tag_prefix + match_tag

                        content_flat_list[matches_idx] = self.replace_ner_tag(content_flat_list[matches_idx], current_tag)
                    break
                    idx += max(1, (label_str_len-1))

        for token_idx, token in enumerate(content_flat_list):
            if content_flat_list[token_idx][3] == self.__empty_string:
                content_flat_list[token_idx] = self.replace_ner_tag(content_flat_list[token_idx], self.__outside_tag)

        return self.line_list_transform(content_flat_list, length_of_lines)

    def replace_ner_tag(self, original_tuple, new_tag):
        return (original_tuple[0], original_tuple[1], original_tuple[2], new_tag)

    # convert list of lines to flat list of tokens
    def flat_token_list_transform(self, doc_list):
        line_list = [item for sublist in doc_list for item in sublist]
        length_of_lines = [len(line) for line in doc_list]
        return line_list, length_of_lines

    # convert a list of tokens back to list of lines
    def line_list_transform(self, line_list, length_of_lines):
        start_idx = 0
        doc_list = []
        for list_len in length_of_lines:
            doc_list.append(line_list[start_idx:start_idx+list_len])
            start_idx += list_len
        return doc_list

    # takes tokens and adds pos tags
    # doc: is a prepared token list with doc structure
    def pos_tag(self, doc):
        copy_doc = copy.deepcopy(doc)
        plain_doc = self.tuple_to_plain(copy_doc)
        pos_doc = nltk.pos_tag_sents(plain_doc)
        return self.add_pos_tags(pos_doc, doc)

    # change tuples to only text strings
    def tuple_to_plain(self, doc):
        for line_idx, line in enumerate(doc):
            for token_idx, token in enumerate(line):
                doc[line_idx][token_idx] = token[0]
        return doc

    # add pos tags to idx slot 1
    def add_pos_tags(self, pos_doc, original_doc):
        for line_idx, line in enumerate(original_doc):
            for token_idx, token in enumerate(line):
                new_pos_tag = pos_doc[line_idx][token_idx][self.__pos_tag_tuple_idx]
                original_doc[line_idx][token_idx] = self.replace_pos_tag(original_doc[line_idx][token_idx], new_pos_tag)
        return original_doc

    def replace_pos_tag(self, original_tuple, new_tag):
        tuple = (original_tuple[0], new_tag, original_tuple[2], original_tuple[3])
        return tuple

    def setup_nonlocal_tagger(self):
        home = expanduser("~")
        os.environ['CLASSPATH'] = home + '/stanford-ner-2015-12-09'
        os.environ['STANFORD_MODELS'] = home + '/stanford-ner-2015-12-09/classifiers'

        self.__stanford_tagger = StanfordNERTagger("english.all.3class.distsim.crf.ser.gz", java_options='-mx4000m')

        stanford_dir = self.__stanford_tagger._stanford_jar[0].rpartition('/')[0]
        stanford_jars = find_jars_within_path(stanford_dir)
        self.__stanford_tagger._stanford_jar = ':'.join(stanford_jars)
        self.__logger.println("nonlocal tagger loaded")

    # nonlocal named entity recoginition tags assignment function
    # resumes: list resumes documents in doc/line/token structure
    def nonlocal_ner_tag(self, resumes):
        self.__logger.println("nonlocal tagging resumes")
        # do not tokenise text
        nltk.internals.config_java(options='-tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerOptions "tokenizeNLs=true"')

        copied_resumes = copy.deepcopy(resumes)
        for doc_idx, doc in enumerate(copied_resumes):
            copied_resumes[doc_idx] = self.tuple_to_plain(doc)

        # list of all lines
        copied_resumes, doc_lengths = self.flat_token_list_transform(copied_resumes)
        tagged_resumes = self.__stanford_tagger.tag_sents(copied_resumes)
        # list of all docs
        tagged_resumes = self.line_list_transform(tagged_resumes, doc_lengths)

        # add nonlocal ner tag in tuple slots to return in same structure as inputted
        for tagged_doc_idx, tagged_doc in enumerate(tagged_resumes):
            self.__logger.println("adding nonlocal tags to resume %s/%s" % (tagged_doc_idx+1, len(tagged_resumes)))
            resumes[tagged_doc_idx] = self.add_nonlocal_ner_tags(tagged_doc, resumes[tagged_doc_idx])

        self.__logger.println("completed nonlocal tagging resumes")

        return resumes

    def add_nonlocal_ner_tags(self, nonlocal_ner_doc, original_doc):
        for line_idx, line in enumerate(original_doc):
            for token_idx, token in enumerate(line):
                try:
                    new_nonlocal_ne_tag = nonlocal_ner_doc[line_idx][token_idx][1]
                    original_doc[line_idx][token_idx] = self.replace_nonlocal_ne_tag(original_doc[line_idx][token_idx], new_nonlocal_ne_tag)
                except IndexError:
                    self.__logger.println("index out of range error while non local ner tagging")
                    self.__logger.println("line_idx %s token_idx %s" % (line_idx, token_idx))
        return original_doc

    def replace_nonlocal_ne_tag(self, original_tuple, new_tag):
        tuple = (original_tuple[0], original_tuple[1], new_tag, original_tuple[3])
        return tuple
