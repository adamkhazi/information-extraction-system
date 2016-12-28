from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import csv
import math
import io

train_sents = []
test_sents = []

total_sents = []

print("reading in dataset")
with io.open('db_generated_datasets/ner_training_data.txt', 'r',
encoding='utf-8') as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tsvin:
        total_sents.append((row[0], row[1], row[2])) #append tuples

split_point = math.ceil(len(total_sents) * 0.7)
train_sents = [list(total_sents[0:split_point])]
test_sents = [list(total_sents[split_point+1:])]
print("done reading and splitting dataset")

#nltk.corpus.conll2002.fileids()

#train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
#test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
#print(train_sents)
#print(test_sents)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    """
    if i > 0:
        word1 = sent[i-1][0]
#        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
#            '-1:postag=' + postag1,
#            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
#        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
#            '+1:postag=' + postag1,
#            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
    """
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent] 

sent2features(train_sents[0])[0]
print("sent2features")
X_train = [sent2features(s) for s in train_sents]
print("X train done")

y_train = [sent2labels(s) for s in train_sents]
print("y train done")

print("sent2features")
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]
print("X and y test done")

trainer = pycrfsuite.Trainer(verbose=True)
print("pycrfsuite Trainer init")

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
print("pycrfsuite Trainer has data")

trainer.select("l2sgd")
trainer.set_params({
    #'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1,  # coefficient for L2 penalty
    'max_iterations': 300,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


print(trainer.params())

print("starting to train...")
trainer.train('test_NER.crfsuite')

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
"""
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

print(bio_classification_report(y_test, y_pred))
"""
