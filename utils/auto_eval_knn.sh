#!/bin/sh

# KNN clssifier 

#python test/nearest_neighbours.py --train_file="" --test_file=""

# python test/nearest_neighbours.py --train_file="" \
#                                   --test_file=""

# python test/nearest_neighbours.py --train_file="/media/mikelf/media_rob/experiments/metric_learning/toybox/novel/triplet_ae/nby_0/x2/train_set_triplet_ae_toybox.npz" \
#                                   --test_file="/media/mikelf/media_rob/experiments/metric_learning/toybox/novel/triplet_ae/nby_0/x2/test_set_triplet_ae_toybox.npz"  

# # --- toybox
echo "toybox - l 0.1"
                                                                                               #toybox/striplet/lambda_0.1/1/train_set_triplet_cnn_softmax_all_toybox.npz
python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/1/train_set_triplet_cnn_softmax_all_toybox.npz" \
                                  --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/1/test_set_triplet_cnn_softmax_all_toybox.npz"   

python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/2/train_set_triplet_cnn_softmax_all_toybox.npz" \
                                  --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/2/test_set_triplet_cnn_softmax_all_toybox.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/train_set_triplet_cnn_softmax_all_toybox.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/test_set_triplet_cnn_softmax_all_toybox.npz"   


# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.01/2/train_set_triplet_cnn_softmax_all_toybox.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.01/2/test_set_triplet_cnn_softmax_all_toybox.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.01/3/train_set_triplet_cnn_softmax_all_toybox.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.01/3/test_set_triplet_cnn_softmax_all_toybox.npz"   

echo "toybox - l 0.0001"

python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/1/train_set_triplet_cnn_softmax_all_toybox.npz" \
                                  --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/1/test_set_triplet_cnn_softmax_all_toybox.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/2/train_set_triplet_cnn_softmax_all_toybox.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/2/test_set_triplet_cnn_softmax_all_toybox.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/3/train_set_triplet_cnn_softmax_all_toybox.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/3/test_set_triplet_cnn_softmax_all_toybox.npz"   

echo "toybox - l 0.01"                                                                          #toybox/striplet/lambda_0.1/2/train_set_triplet_cnn_softmax_novel_toybox.npz

python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.01/1/train_set_triplet_cnn_softmax_all_toybox.npz" \
                                  --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.01/1/test_set_triplet_cnn_softmax_all_toybox.npz"   
                                                                                                                                                                                                                                                                                                                                                                                                                                                     

# # --- core50
# echo "core50 - l 0.1"/home/mikelf/deepNetwork/alexaDATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.1/1/test_set_triplet_cnn_softmax_novel_core50.npz


# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.1/1/train_set_triplet_cnn_softmax_known_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.1/1/test_set_triplet_cnn_softmax_known_core50.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.1/2/train_set_triplet_cnn_softmax_novel_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.1/2/test_set_triplet_cnn_softmax_novel_core50.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.1/2/train_set_triplet_cnn_softmax_all_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.1/2/test_set_triplet_cnn_softmax_all_core50.npz"   


# echo "core50 - l 0.01"

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.01/1/train_set_triplet_cnn_softmax_all_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.01/1/test_set_triplet_cnn_softmax_all_core50.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.01/2/train_set_triplet_cnn_softmax_all_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.01/2/test_set_triplet_cnn_softmax_all_core50.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.01/3/train_set_triplet_cnn_softmax_all_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.01/3/test_set_triplet_cnn_softmax_all_core50.npz"   

# echo "core50 - l 0.0001"

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.0001/1/train_set_triplet_cnn_softmax_all_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.0001/1/test_set_triplet_cnn_softmax_all_core50.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.0001/2/train_set_triplet_cnn_softmax_all_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.0001/2/test_set_triplet_cnn_softmax_all_core50.npz"   

# python test/nearest_neighbours.py --train_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.0001/3/train_set_triplet_cnn_softmax_all_core50.npz" \
#                                   --test_file="/media/alexa/DATA/Miguel/evaluation/ICRA/novel/core50/striplet/lambda_0.0001/3/test_set_triplet_cnn_softmax_all_core50.npz"   



# python test/nearest_neighbours.py --train_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_0/x1/train_set_triplet_ae_known_core50.npz" \
#                                   --test_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_0/x1/test_set_triplet_ae_known_core50.npz"

# python test/nearest_neighbours.py --train_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_0/x2/train_set_triplet_ae_known_core50.npz" \
#                                   --test_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_0/x2/test_set_triplet_ae_known_core50.npz"

# python test/nearest_neighbours.py --train_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_0/x3/train_set_triplet_ae_known_core50.npz" \
#                                   --test_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_0/x3/test_set_triplet_ae_known_core50.npz"

# python test/nearest_neighbours.py --train_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_3/x1/train_set_triplet_ae_known_core50.npz" \
#                                   --test_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_3/x1/test_set_triplet_ae_known_core50.npz"

# python test/nearest_neighbours.py --train_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_3/x2/train_set_triplet_ae_known_core50.npz" \
#                                   --test_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_3/x2/test_set_triplet_ae_known_core50.npz"      

# python test/nearest_neighbours.py --train_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_3/x3/train_set_triplet_ae_known_core50.npz" \
#                                   --test_file="/media/mikelf/media_rob/experiments/metric_learning/core50/novel/triplet_ae/nby_3/x3/test_set_triplet_ae_known_core50.npz"

                                                                                                                                                             

# ARC:

# python test/nearest_neighbours_arc.py --test_item_dir="" \
#                                       --test_imgs=""


# python test/nearest_neighbours_arc.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x2/test-item" \
#                                       --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x2/test-img/test_set_cnn_novel_known_arc.npz"
                                                                                          

# python test/nearest_neighbours_arc.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x3/test-item" \
#                                       --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x3/test-img/test_set_cnn_novel_known_arc.npz"

# python test/nearest_neighbours_arc.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/2/test-item" \
#                                       --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/2/test-img/test_set_triplet_ae_known_arc.npz" 
                                      
# python test/nearest_neighbours_arc.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/3/test-item" \
#                                       --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/3/test-img/test_set_triplet_ae_known_arc.npz"


#                                       python test/nearest_neighbours_arc.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/1/test-item" \
#                                       --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/1/test-img/test_set_triplet_ae_known_arc.npz"

#python test/nearest_neighbours_arc_kn.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x2/test-item" \
#                                      --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x2/test-img/test_set_triplet_ae_arc.npz" 
                                      
#python test/nearest_neighbours_arc_kn.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x1/test-item" \
#                                      --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x1/test-img/test_set_triplet_ae_arc.npz"

#python test/nearest_neighbours_arc_kn.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x3/test-item" \
#                                      --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x3/test-img/test_set_triplet_ae_known_arc.npz"

#python test/nearest_neighbours_arc_kn.py --test_item_dir="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x4/test-item" \
#                                      --test_imgs="/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x4/test-img/test_set_triplet_ae_known_arc.npz"  