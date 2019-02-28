from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import time
import numpy as np
from tqdm import tqdm


npzfile_train = np.load("/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/core50_novel/1/train_set_triplet_cnn_l2_core50.npz")
npzfile_test = np.load("/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/core50_novel/1/test_set_triplet_cnn_l2_core50.npz")


X_train = npzfile_train['embeddings'][1:]
y_train = npzfile_train['lebels'][1:]
filenames_train = npzfile_train['filenames'][1:]


X_test = npzfile_test['embeddings'][1:]
y_test = npzfile_test['lebels'][1:]
filenames_test = npzfile_test['filenames'][1:]


#=============================================================================
#                Temporal Modelling

# X_test_temporal = np.zeros((1,128))
#
# logit_average = 0
#
# for idx, logit in enumerate(X_test):
#
#     print (idx)
#
#     if idx == 0:
#         X_test_temporal = np.concatenate((X_test_temporal, [logit]), axis=0)
#     else:
#         logit_average = (X_test_temporal[idx-1] + logit) / 2
#         X_test_temporal = np.concatenate((X_test_temporal, [logit_average]), axis=0)
#
# X_test_temporal = X_test_temporal[1:]
#
# print("Temporal modelling Done")

#============================================================================


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)

print(" kNN done  ")

total = len(y_test)
correct = 0

print(" =========================== ")


#=============================================================================
#                    Label smoothing
predictions = neigh.predict(X_test)

print (predictions[0:60])
print (y_test[0:60])

print((predictions[0:60] == y_test[0:60]).sum()/60.0)

counts = np.bincount(predictions[0:60].astype(int))
print np.argmax(counts)
#predictions_temporal = neigh.predict(logits_temporal_filter)

#for idx, prediction in enumerate(predictions):

#    if idx >0 : predictions_temporal[idx] = predictions[idx]


#correct += (predictions == y_test).sum()

#correct += (neigh.predict(logits_temporal_filter) == y_test).sum()




#=============================================================================

print (total)
print( 1.*correct/total )
