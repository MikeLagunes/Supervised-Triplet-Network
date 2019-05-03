import json

from loader.cnn_core50 import cnn_core50
from loader.triplet_resnet_core50 import triplet_resnet_core50
from loader.triplet_resnet_core50_softmax import triplet_resnet_core50_softmax



def get_loader(name):
    """get_loader

    :param name:
    """
    return {
      
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
