import json

from loader.cnn_tless import cnn_tless
from loader.triplet_resnet_tless import triplet_resnet_tless
from loader.triplet_resnet_tless_softmax import triplet_resnet_tless_softmax

from loader.cnn_toybox import cnn_toybox
from loader.triplet_resnet_toybox import triplet_resnet_toybox
from loader.triplet_resnet_toybox_softmax import triplet_resnet_toybox_softmax

from loader.cnn_arc import cnn_arc
from loader.triplet_resnet_arc import triplet_resnet_arc
from loader.triplet_resnet_arc_softmax import triplet_resnet_arc_softmax

from loader.cnn_core50 import cnn_core50
from loader.triplet_resnet_core50 import triplet_resnet_core50
from loader.triplet_resnet_core50_softmax import triplet_resnet_core50_softmax



def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'cnn_tless':cnn_tless,
        'triplet_resnet_tless': triplet_resnet_tless,
        'triplet_resnet_tless_softmax': triplet_resnet_tless_softmax,

        'cnn_toybox':cnn_toybox,
        'triplet_resnet_toybox': triplet_resnet_toybox,
        'triplet_resnet_toybox_softmax': triplet_resnet_toybox_softmax,
         
        'cnn_arc':cnn_arc,
        'triplet_resnet_arc': triplet_resnet_arc,
        'triplet_resnet_arc_softmax': triplet_resnet_arc_softmax,
        
        'cnn_core50':cnn_core50,
        'triplet_resnet_core50': triplet_resnet_core50,
        'triplet_resnet_core50_softmax': triplet_resnet_core50_softmax,

    }[name]


#def get_data_path(name, config_file='../config.json'):
def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']
