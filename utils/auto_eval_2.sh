#!/bin/sh

# TLESS

# s-triplet

#  novel 

# - novel - known


python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/2/triplet_softmax_l1_toybox_ckpt_60.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/2/ --split=train --dataset=toybox
python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/2/triplet_softmax_l1_toybox_ckpt_60.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/2/ --split=test --dataset=toybox

python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/triplet_softmax_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/ --split=train --dataset=toybox
python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/triplet_softmax_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.1/3/ --split=test --dataset=toybox

python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/3/triplet_softmax_l4_toybox_ckpt_90.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/3/ --split=train --dataset=toybox
python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/3/triplet_softmax_l4_toybox_ckpt_90.pkl --test_path=/media/alexa/DATA/Miguel/evaluation/ICRA/novel/toybox/striplet/lambda_0.0001/3/ --split=test --dataset=toybox

# CORE50

# s-triplet

#  novel 
# - all


