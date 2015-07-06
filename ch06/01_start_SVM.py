# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#
# This script trains multinomial Naive Bayes on the tweet corpus
# to find two different results:
# - How well can we distinguis positive from negative tweets?
# - How well can we detect whether a tweet contains sentiment at all?
#

import time
start_time = time.time()

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit
from sklearn import svm

from utils import plot_pr
from utils import load_sanders_data
from utils import tweak_labels


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

import nltk.stem
import sys
import scipy as sp
import math

from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
from sklearn import neighbors

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
vectorizer = StemmedTfidfCountVectorizer(min_df = 1, stop_words = 'english')

def create_ngram_model():
    tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3),
                                   analyzer="word", binary=False)
    #clf = MultinomialNB()
    clf = svm.SVC(kernel='rbf')
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    return pipeline


def train_model(SVMType, X, Y, name="SVM ngram", plot=False):
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, random_state=0)
    vectorizer = StemmedTfidfCountVectorizer(min_df = 1, stop_words = 'english')
    X = vectorizer.fit_transform(X)
    #cv = KFold(n=len(X), n_folds=10, indices=True)
    train_errors = []
    test_errors = []

    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []
    Fmeasures = []

    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]
        #print X_train
        #print y_train
        #clf = clf_factory()
        #clf = svm.SVC(kernel='rbf')
        #clf = svm.SVC(kernel='poly')
        #clf = svm.SVC(kernel='linear')
        clf = svm.SVC(kernel= SVMType)
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        #print train_score, test_score
        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)
        scores.append(test_score)
        
        proba = clf.predict(X_test)
        #print proba
        """
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        """
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test, proba)
        Fmeasure = ((2 * precision * recall)/(precision + recall))
        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)
        Fmeasures.append(Fmeasure)
    scores_to_sort = pr_scores
    median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
    
    if plot:
        plot_pr(pr_scores[median], name, "01", precisions[median],
                recalls[median], label=name)

        summary = (np.mean(scores), np.std(scores),
                   np.mean(pr_scores), np.std(pr_scores))
        print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)
        print("F-measure:" + str(np.mean(Fmeasures)))
    return np.mean(train_errors), np.mean(test_errors)


def print_incorrect(clf, X, Y):
    Y_hat = clf.predict(X)
    wrong_idx = Y_hat != Y
    X_wrong = X[wrong_idx]
    Y_wrong = Y[wrong_idx]
    Y_hat_wrong = Y_hat[wrong_idx]
    for idx in range(len(X_wrong)):
        print("clf.predict('%s')=%i instead of %i" %
              (X_wrong[idx], Y_hat_wrong[idx], Y_wrong[idx]))


if __name__ == "__main__":
    X_orig, Y_orig = load_sanders_data()
    classes = np.unique(Y_orig)
    for c in classes:
        print("#%s: %i" % (c, sum(Y_orig == c)))

    print("== Pos vs. neg ==")
    pos_neg = np.logical_or(Y_orig == "positive", Y_orig == "negative")
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]
    Y = tweak_labels(Y, ["positive"])

    train_model('rbf', X, Y, name="pos vs neg", plot=True)
    train_model('poly', X, Y, name="pos vs neg", plot=True)
    train_model('linear', X, Y, name="pos vs neg", plot=True)

    print("== Pos/neg vs. irrelevant/neutral ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])
    
    train_model('rbf', X, Y, name="pos vs neg", plot=True)
    train_model('poly', X, Y, name="pos vs neg", plot=True)
    train_model('linear', X, Y, name="pos vs neg", plot=True)

    print("== Pos vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    
    train_model('rbf', X, Y, name="pos vs neg", plot=True)
    train_model('poly', X, Y, name="pos vs neg", plot=True)
    train_model('linear', X, Y, name="pos vs neg", plot=True)

    print("== Neg vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    
    train_model('rbf', X, Y, name="pos vs neg", plot=True)
    train_model('poly', X, Y, name="pos vs neg", plot=True)
    train_model('linear', X, Y, name="pos vs neg", plot=True)

    print("time spent:", time.time() - start_time)
