from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


from sklearn import svm
from sklearn.svm import SVC
from time import time
import numpy as np
from tqdm import tqdm
import argparse


def KNN_precision(args):

    npzfile_train = np.load(args.train_file)
    npzfile_test = np.load(args.test_file)


    X_train = npzfile_train['embeddings'][1:]
    y_train = npzfile_train['lebels'][1:]
    filenames_train = npzfile_train['filenames'][1:]

    #print (X_train.shape, y_train.shape)

    X_test = npzfile_test['embeddings'][1:]
    y_test = npzfile_test['lebels'][1:]
    filenames_test = npzfile_test['filenames'][1:]

    #neigh = MLPClassifier(n_neighbors=100,n_jobs=-1)

    #neigh = KNeighborsClassifier(n_neighbors=5)
    #neigh.fit(X_train, y_train)

    # print("kNN done - 5 neighbors")
    #
    # total = len(y_test-1)
    # correct = 0
    #
    #
    # correct += (neigh.predict(X_test) == y_test).sum()
    #
    # print (total)
    # print("KNN - Precision: ", 100.*correct/total )


    #===============================================
    # linear svm

    # clf = svm.LinearSVC()#SVC(kernel='rbf', gamma='auto')
    # clf.fit(X_train, y_train)
    #
    # print(" SVM done - 5 neighbors")
    #
    # total = len(y_test - 1)
    # correct = 0
    #
    # correct += (clf.predict(X_test) == y_test).sum()
    #
    # print (total)
    # print("SVM - Precision: ", 100. * correct / total)

    #==================================================

    # clf = DecisionTreeClassifier(max_depth=100)
    # clf.fit(X_train, y_train)
    #
    # print(" Decision Tree max depth - 5 ")
    #
    # total = len(y_test - 1)
    # correct = 0
    #
    # correct += (clf.predict(X_test) == y_test).sum()
    #
    # print (total)
    # print("Decision Tree - Precision: ", 100. * correct / total)

    # =======================================================


    # clf = RandomForestClassifier(max_depth=120)
    # clf.fit(X_train, y_train)
    #
    # print(" Random Forest ")
    #
    # total = len(y_test - 1)
    # correct = 0
    #
    # correct += (clf.predict(X_test) == y_test).sum()
    #
    # print (total)
    # print(" Random Forest: ", 100. * correct / total)
    #
    #=======================================================

    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X_train, y_train)

    print(" logistic regression ")

    total = len(y_test - 1)
    correct = 0

    correct += (clf.predict(X_test) == y_test).sum()

    print (total)
    print(" logistic regression: ", 100. * correct / total)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--train_file', nargs='?', type=str, default="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/softmax/3/train_set_triplet_ae_softmax_novel_core50.npz",
                        help='Path of the input image')
    parser.add_argument('--test_file', nargs='?', type=str, default="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/softmax/3/test_set_triplet_ae_softmax_novel_core50.npz",
                        help='Path of the input image')


    args = parser.parse_args()
    KNN_precision(args)
