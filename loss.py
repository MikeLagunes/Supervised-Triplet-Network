import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxLoss(nn.Module):
    """
    Softmax loss
    Takes logits and class labels
    """

    def __init__(self, margin=128.0, size_average=True):
        super(SoftmaxLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()

    def forward(self, prediction, labels ):

        loss_softmax = self.xentropy(input=prediction, target=labels)

        return  loss_softmax

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=4.0, size_average=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
                    #embedding1, embedding2, embedding3, rec1, rec2, rec3, images, images_pos, images_neg
    def forward(self, anchor, positive, negative ):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
    
        return  losses.sum()


class TripletSoftmaxLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample, a negative sample, logits and class labels
    """

    def __init__(self, margin=0.0, size_average=True, lambda_factor=0.001 ):
        super(TripletSoftmaxLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = 0.01
                    
    def forward(self, anchor, positive, negative, outputs, labels ):
        distance_positive = torch.abs(anchor - positive).sum(1)  # .pow(.5)
        distance_negative = torch.abs(anchor - negative).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total = self.lambda_factor*losses.sum() + loss_softmax
        #return losses.mean() if size_average else losses.sum()

        return loss_total, losses.sum(), loss_softmax # 