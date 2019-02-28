from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import time
import numpy as np
from tqdm import tqdm
import argparse
import os, sys


def KNN_precision(args):


    train_file = os.path.split(args.ckpt_path)[0]+"/train_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"
    test_file  = os.path.split(args.ckpt_path)[0]+"/test_" + os.path.split(args.ckpt_path)[1][0:-4] + "_" + args.instances + ".npz"


    npzfile_train = np.load(train_file)
    npzfile_test = np.load(test_file)


    X_train = npzfile_train['embeddings'][1:]
    y_train = npzfile_train['lebels'][1:]
    filenames_train = npzfile_train['filenames'][1:]

    #print (X_train.shape, y_train.shape)

    X_test = npzfile_test['embeddings'][1:]
    y_test = npzfile_test['lebels'][1:]
    filenames_test = npzfile_test['filenames'][1:]

    #neigh = MLPClassifier(n_neighbors=100,n_jobs=-1)
    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-4)
    neigh.fit(X_train, y_train)

    #print("kNN done - 5 neighbors")

    total = len(y_test-1)
    correct = 0


    correct += (neigh.predict(X_test) == y_test).sum()

    #print (total)
    #print("Precision KNN : ", 100.*correct/total )
    sys.stdout.write("{}".format(100.*correct/total))
    sys.stdout.flush()
    sys.exit(0)
    #return 100.0 * correct / total
    
    #sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--ckpt_path', nargs='?', type=str, default="/home/mikelf/Desktop/v4/ae_softmax_hinterstoisser_v4.pkl",
                        help='Path of the input image')
    parser.add_argument('--instances', nargs='?', type=str, default="full",
                        help='Path of the input image')

    args = parser.parse_args()
    KNN_precision(args)