#!/bin/sh

# TLESS

# s-triplet

# all


python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/triplet_softmax_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/ --split=train --dataset=toybox
python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/triplet_softmax_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/ --split=test --dataset=toybox


#Core50

# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/triplet_softmax_sgd_all_l1_core50_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/ --split=train --dataset=core50
# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/triplet_softmax_sgd_all_l1_core50_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/ --split=test --dataset=core50

# # TOybox

# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/triplet_softmax_sgd_all_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/ --split=train --dataset=toybox
# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/triplet_softmax_sgd_all_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/ --split=test --dataset=toybox


# CORE50

# s-triplet

#  novel 
`
# - all


