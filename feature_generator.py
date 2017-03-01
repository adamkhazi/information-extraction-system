import copy 

from logger import Logger

# generates features for tokens in the dataset
class FeatureGenerator():
    def __init__(self, w2v_model, word2count, word2idx):
        self.__logger = Logger()
        self.__logger.println("feature generator created")
        self.__we_model = w2v_model
        self.word2count = word2count
        self.word2idx = word2idx

    def generate_features_docs(self, data):
        feature_data = copy.deepcopy(data)
        X = [self.doc2features(doc_idx, d) for doc_idx, d in enumerate(feature_data)]
        self.__logger.println("generated features")
        return X

    def generate_true_outcome(self, data):
        true_outcome_data = copy.deepcopy(data)
        y = [self.doc2labels(d) for d in true_outcome_data]
        return y

    def word2features(self, line, token_idx, line_idx, doc_idx, doc_size):
        word = line[token_idx][0]
        postag = line[token_idx][1]
        nonlocalnertag = line[token_idx][2]

        features = dict()

        features["bias"] = 1.0
        features["word"] = word.lower()
        features['word[-3:]'] = word[-3:].lower()
        features['word[-2:]'] = word[-2:].lower()
        features['word.isupper'] = word.isupper()
        features['word.istitle'] = word.istitle()
        features['word.isdigit'] = word.isdigit()
        features['word.freq'] = self.word_to_count(word)
        features['word.idx'] = float(token_idx)
        features['line.idx'] = float(line_idx)
        features['line.size'] = float(len(line))
        features['pos'] = postag
        features['pos[-3:]'] = postag[-3:]
        features['pos[-2:]'] = postag[-2:]
        features['nonlocalner'] = nonlocalnertag

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
            features['word-1[-3:]'] = word1[-3:]
            features['word-1[-2:]'] = word1[-2:]
            features['word.freq'] = self.word_to_count(word1)
            features['word-1.idx']= float(token_idx-1)
            features['nonlocalner'] = nonlocalnertag1

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
            features['word.freq'] = self.word_to_count(word1)
            features['word+1.idx']= float(token_idx+1)
            features['nonlocalner'] = nonlocalnertag1
        else:
            features['bigram+1'] = word.lower() + "EOL"
            features['EOL'] = 1.0

        """
        if token_idx > 0 and token_idx < len(line)-1:
            word1behind = line[token_idx-1][0]
            word1ahead = line[token_idx+1][0]
            features['trigram-1+1'] = word1behind.lower() + word.lower() + word1ahead.lower()
        """

        features = self.add_we_vector(features, word)

        if line_idx == 0:
            features['BOD'] = 1.0

        if line_idx == doc_size:
            features['EOD'] = 1.0

        return features

    def sent2features(self, line, line_idx, doc_idx, doc_size):
        return [self.word2features(line, token_idx, line_idx, doc_idx, doc_size) for token_idx in range(len(line))]

    def sent2tokens(self, sent):
        return [token for token, pos_tag, nonlocalne, label in sent]

    def doc2features(self, doc_idx, doc):
        return [self.sent2features(doc[line_idx], line_idx, doc_idx, len(doc)-1) for line_idx in range(len(doc))]

    def sent2labels(self, sent):
        labels = []
        for token, pos_tag, nonlocalne, label in sent:
            labels.append(label)
        return labels

    def doc2labels(self, doc):
        return [self.sent2labels(sent) for sent in doc]

    def word_to_count(self, word):
        try:
            return float(self.word2count[word.lower()])
        except KeyError:
            return "UNKNOWN_WORD_COUNT"

    def add_we_vector(self, features, word):
        try:
            for d_idx, dimension in enumerate(self.__we_model[word.lower()]):
                features["we_dimen_"+str(d_idx)] = dimension
        except KeyError:
            features["we_dimen_"+str(0)] = "UNKNOWN_WE"
        return features

    def similarity_between(self, word1, word2):
        try:
            return self.__we_model.similarity(word1.lower(), word2.lower())
        except:
            return "NO_SIMILARITY"


