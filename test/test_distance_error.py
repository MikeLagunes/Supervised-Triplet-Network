from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from time import time
import numpy as np
from tqdm import tqdm 
from sklearn.metrics.pairwise import manhattan_distances


train_set = np.load("/media/mikelf/media_rob/experiments/metric_learning/siamese/far_view/train_set_segnet_tless.npz")
test_set = np.load("/media/mikelf/media_rob/experiments/metric_learning/siamese/far_view/test_set_segnet_tless.npz")
print(train_set.files)
print (train_set['arr_1'].shape)
print (train_set['arr_0'].shape)

#idx = np.random.choice(np.arange(len(train_set['arr_1'])), 3000, replace=False)

# X = npzfile['arr_0'][0:2000]
# y = npzfile['arr_1'][0:2000]

x = train_set['arr_0']
y = train_set['arr_1']
x_test = test_set['arr_0']
y_test = test_set['arr_1']


print(manhattan_distances([x[1]],[x[2]]))


# #neigh = MLPClassifier(n_neighbors=100,n_jobs=-1)
# neigh = MLPClassifier(alpha=1)
# neigh.fit(x, y) 

# print("NN done")

# total = 0
# correct = 0

# for test_point, label in tqdm(zip(x_test,y_test),total=len(y_test)):

#   if neigh.predict([test_point]) == label:
#     correct += 1

#   total += 1

# print (total)
# print( 1.*correct/total )
