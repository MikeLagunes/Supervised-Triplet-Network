import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

import sys, os
sys.path.append('.')

from models import get_model
from loader import get_loader, get_data_path
from utils import convert_state_dict

import csv

known_classes = ["avery_binder","balloons","band_aid_tape","bath_sponge","black_fashion_gloves","burts_bees_baby_wipes",
"colgate_toothbrush_4pk","composition_book","crayons","duct_tape","empty","epsom_salts","expo_eraser","fiskars_scissors",
"flashlight","glue_sticks","hand_weight","hanes_socks","hinged_ruled_index_cards","ice_cube_tray","irish_spring_soap",
"laugh_out_loud_jokes","marbles","measuring_spoons","mesh_cup","mouse_traps","pie_plates","plastic_wine_glass","poland_spring_water",
"reynolds_wrap","robots_dvd","robots_everywhere","scotch_sponges","speed_stick","table_cloth","tennis_ball_container","ticonderoga_pencils",
"tissue_box","toilet_brush","white_facecloth","windex"]

items_test = ['avery_binder','balloons','band_aid_tape','bath_sponge','black_fashion_gloves','burts_bees_baby_wipes',
'cherokee_easy_tee_shirt','cloud_b_plush_bear','colgate_toothbrush_4pk','composition_book','cool_shot_glue_sticks',
'crayons','creativity_chenille_stems','dove_beauty_bar','dr_browns_bottle_brush','duct_tape',
'easter_turtle_sippy_cup','elmers_washable_no_run_school_glue','empty','epsom_salts','expo_eraser',
'fiskars_scissors','flashlight','folgers_classic_roast_coffee','glue_sticks','hand_weight',
'hanes_socks','hinged_ruled_index_cards','i_am_a_bunny_book','ice_cube_tray','irish_spring_soap',
'jane_eyre_dvd','kyjen_squeakin_eggs_plush_puppies','laugh_out_loud_jokes','marbles','measuring_spoons',
'mesh_cup','mouse_traps','oral_b_toothbrush_red','peva_shower_curtain_liner','pie_plates',
'plastic_wine_glass','platinum_pets_dog_bowl','poland_spring_water','rawlings_baseball','reynolds_wrap',
'robots_dvd','robots_everywhere','scotch_bubble_mailer','scotch_sponges','speed_stick',
'staples_index_cards','table_cloth','tennis_ball_container','ticonderoga_pencils','tissue_box',
'toilet_brush','up_glucose_bottle','white_facecloth','windex','woods_extension_cord']

def test(args):

    # Setup image
    # train/eval
    # Setup Dataloader

    root_dir = args.test_path #"/media/alexa/DATA/Miguel/results/" + args.dataset +"/triplet_cnn/" 

    data_loader = get_loader("cnn_" + args.dataset)
    data_path = get_data_path(args.dataset)
    

    for items in known_classes: # Using only images [known_classes]

        try:

            t_loader = data_loader(data_path, is_transform=True, split=args.split, img_size=(args.img_rows, args.img_cols), augmentations=None, class_id=items)

            n_classes = t_loader.n_classes

            trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=0, shuffle=False)
         
            # Setup Model
            # Setup Model
            model = get_model(args.arch, n_classes)
            #### ..............................................................................................................

            # # Keys to remove:
            # with open('/home/mikelf/Desktop/remove_keys_asiamese.csv', 'rb') as f:
            #     reader = csv.reader(f)
            #     your_list = list(reader)

            # keys_to_remove = []

            # for items_list in your_list:
            #     keys_to_remove.append(items_list[0])

            # weights = torch.load(args.model_path)['model_state']

            # #for key, value in weights.iteritems():
            # #    print key

            # for key in keys_to_remove:
            #     del weights[key]

            ### .............................................................................................................


            weights = torch.load(args.model_path)['model_state']

            #state = convert_state_dict(weights)
            model.load_state_dict(weights)
            model.eval()

            #model.load_state_dict(weights)
            print ("Model Loaded, Epoch: ", torch.load(args.model_path)['epoch'])

            print ("Projecting: " + args.dataset + " | " + args.split + " set" + " | " + items)

            #print(model)
            
            model = model.cuda()
            model.eval()
            #images = Variable(img.cuda(0), volatile=True)

            output_embedding = np.array([])
            outputs_embedding = np.zeros((1,128))
            labels_embedding = np.zeros((1))
            path_imgs = []

            for i, (images, labels, path_img) in enumerate(tqdm(trainloader)):
                    
                images = Variable(images.cuda())
                labels = labels.view(len(labels))
                labels = labels.cpu().numpy()
                #labels = Variable(labels.cuda())
                outputs, _, _ = model(images, images, images)

                output_embedding = outputs.data
                output_embedding = output_embedding.cpu().numpy()

                outputs_embedding = np.concatenate((outputs_embedding,output_embedding), axis=0)
                labels_embedding = np.concatenate((labels_embedding,labels), axis=0)

                path_imgs.extend(path_img)

              
            np.savez(root_dir + args.split + "_set_triplet_cnn_" + items +"-item_" + args.dataset,  embeddings=outputs_embedding, lebels=labels_embedding, filenames=path_imgs)

        except ValueError:
            pass


    print ('Done: ')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch', nargs='?', type=str, default='triplet_cnn', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='', 
                        help='Path to the saved model')
    parser.add_argument('--test_path', nargs='?', type=str, default='', 
                        help='Path to saving results')
    parser.add_argument('--dataset', nargs='?', type=str, default='core50', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=5, 
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test-item', 
                        help='Dataset split to use [\'train, eval\']')

    args = parser.parse_args()
    test(args)
