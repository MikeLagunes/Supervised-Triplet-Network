#!/bin/sh


#Triplet CNN - baseline

#python test/test_triplet_cnn_embed_arc_items.py  --model_path=/media/mikelf/media_rob/experiments/arc/triplet/all_vs_all_2/triplet_cnn_novel_x1_allvsall2_cont_arc_temp.pkl --test_path=/media/mikelf/media_rob/experiments/arc/triplet/all_vs_all_2/test-item/ --split=test-item --dataset=arc
#python test/test_triplet_cnn_embed_arc_items.py  --model_path=/media/mikelf/media_rob/experiments/arc/triplet/img_vs_img/triplet_cnn_novel_x2_img_vs_img_arc_temp.pkl --test_path=/media/mikelf/media_rob/experiments/arc/triplet/img_vs_img/train-img/ --split=train --dataset=arc
#python test/test_triplet_cnn_embed_arc_imgs.py   --model_path=/media/mikelf/media_rob/experiments/arc/triplet/img_vs_img/triplet_cnn_novel_x2_img_vs_img_arc_temp.pkl --test_path=/media/mikelf/media_rob/experiments/arc/triplet/img_vs_img/test/ --split=test --dataset=arc
#python test/test_triplet_ae_embed.py  --model_path=/media/mikelf/media_rob/experiments/arc/tae/arc/allvsall/1/triplet_ae_nby_x1_arc_temp.pkl --test_path=/media/mikelf/media_rob/experiments/arc/tae/arc/allvsall/1/train-img/ --split=train --dataset=arc

# CNN embedding

python test/test_cnn_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x2/cnn_arc_items_plus_imgs__arc_best_119.pkl --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x2/test-item/ --split=test-item --dataset=arc
python test/test_cnn_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x2/cnn_arc_items_plus_imgs__arc_best_119.pkl --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x2/test-img/ --split=test --dataset=arc

python test/test_cnn_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x3/cnn_arc_items_plus_imgs__arc_100.pkl --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x3/test-item/ --split=test-item --dataset=arc
python test/test_cnn_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x3/cnn_arc_items_plus_imgs__arc_100.pkl --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/cnn/x3/test-img/ --split=test --dataset=arc


#python test/test_triplet_cnn_embed_arc_items.py --model_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/all_vs_all/triplet_cnn_x1_arc_1200.pkl --test_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/all_vs_all/test-item/ --split=test-item --dataset=arc
#python test/test_triplet_cnn_embed_arc_imgs.py  --model_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/all_vs_all/triplet_cnn_x1_arc_1200.pkl --test_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/all_vs_all/test-img/ --split=test --dataset=arc
                                                             
#python test/test_triplet_cnn_embed_arc_imgs.py  --model_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/img_vs_img/2/triplet_cnn_novel_x2_img_vs_img_arc_temp.pkl --test_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/img_vs_img/2/train-img/ --split=train --dataset=arc
#python test/test_triplet_cnn_embed_arc_imgs.py  --model_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/img_vs_img/2/triplet_cnn_novel_x2_img_vs_img_arc_temp.pkl --test_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/img_vs_img/2/test-img/ --split=test --dataset=arc
#python test/test_triplet_cnn_embed_arc_imgs.py  --model_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/img_vs_item/triplet_cnn_x1_arc_temp.pkl --test_path=/home/mikelf/deepNetwork/alexaDATA/Miguel/checkpoints/arc/img_vs_item/test-img/ --split=test --dataset=arc


#python test/test_triplet_ae_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/arc/tae/allvsall2/1/triplet_ae_nby_allvsall2_x1_arc_temp.pkl --test_path=/media/mikelf/media_rob/experiments/arc/tae/allvsall2/1/test-item/  --split=test-item --dataset=arc



#python test/test_triplet_cnn_embed_arc_items.py  --model_path=/media/mikelf/media_rob/experiments/arc/tae/arc/allvsall/1/triplet_ae_nby_x1_arc_temp.pkl --test_path=/media/mikelf/media_rob/experiments/arc/tae/arc/allvsall/1/test-img/ --split=test --dataset=arc

# Triplet AE

# python test/test_triplet_ae_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x2/triplet_ae_novel_fin_hm_x1_arc_temp.pkl \
#                                                --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x2/test-item/ --split=test-item --dataset=arc
# python test/test_triplet_ae_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x2/triplet_ae_novel_fin_hm_x1_arc_temp.pkl \
#                                     --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x2/test-img/ --split=test --dataset=arc


# python test/test_triplet_ae_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x3/triplet_ae_novel_fin_hm_x1_arc_1700.pkl \
#                                                --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x3/test-item/ --split=test-item --dataset=arc
# python test/test_triplet_ae_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x3/triplet_ae_novel_fin_hm_x1_arc_1700.pkl \
#                                     --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x3/test-img/ --split=test --dataset=arc


# python test/test_triplet_ae_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x4/triplet_ae_novel_fin_hm_x1_arc_1600.pkl \
#                                                --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x4/test-item/ --split=test-item --dataset=arc
# python test/test_triplet_ae_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x4/triplet_ae_novel_fin_hm_x1_arc_1600.pkl \
#                                     --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_1/4/x4/test-img/ --split=test --dataset=arc


# python test/test_triplet_ae_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/1/triplet_ae_novel_same_hm_x1_arc_temp.pkl \
#                                                --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/1/test-item/ --split=test-item --dataset=arc

# python test/test_triplet_ae_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/1/triplet_ae_novel_same_hm_x1_arc_temp.pkl \
#                                      --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/1/test-img/ --split=test --dataset=arc                                                              


# python test/test_triplet_ae_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/2/triplet_ae_same_x1_arc_1500.pkl \
#                                                --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/2/test-item/ --split=test-item --dataset=arc

#python test/test_triplet_ae_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/2/triplet_ae_same_x1_arc_1500.pkl  \
#                                     --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/2/test-img/ --split=test --dataset=arc  

# python test/test_triplet_ae_embed_arc_items.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/3/triplet_ae_same_x1_arc_temp.pkl \
#                                                --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/3/test-item/ --split=test-item --dataset=arc

# python test/test_triplet_ae_embed.py --model_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/3/triplet_ae_same_x1_arc_temp.pkl  \
#                                      --test_path=/media/mikelf/media_rob/experiments/metric_learning/arc/novel/triplet_ae/nby_0/3/test-img/ --split=test --dataset=arc                                      
