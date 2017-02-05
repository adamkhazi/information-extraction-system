import gensim

from gensim.models import word2vec

from logger import Logger

# Word Embedding Model
class WeModel():

    __seperator = "/"
    __w2v_model_name = "w2v_model"
    __pre_trained_models_folder = "pre_trained_models"
    __google_news_word2vec = "GoogleNews-vectors-negative300.bin.gz"

    def __init__(self):
        self.__logger = Logger()
        self.__logger.println("word embedding model created")

    def train(self, data, dimen=30):
        all_lines = []
        for doc_idx, doc in enumerate(data):
            for line_idx, line in enumerate(doc):
                all_lines.append(line)

        for line_idx, line in enumerate(all_lines):
            for token_idx, token in enumerate(line):
                all_lines[line_idx][token_idx] = token[0].lower()

        w2v_model = word2vec.Word2Vec(all_lines, size=dimen, iter=10, min_count=1)
        return w2v_model

    def save(self, w2v_model):
        w2v_model.save(self.__w2v_model_name)

    def read(self):
        return word2vec.Word2Vec.load(self.__w2v_model_name)

    def load_pretrained_model(self, model_name=__google_news_word2vec):
        path = self.__pre_trained_models_folder + self.__seperator + model_name
        return gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
