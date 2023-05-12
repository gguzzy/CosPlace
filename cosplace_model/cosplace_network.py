
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
#import timm 

from cosplace_model.layers import Flatten, L2Norm, GeM

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "vit_base_patch16_224": 768,
}


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone : str, fc_output_dim : int):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18' or 'vit_deit_base_patch16_224'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        if backbone_name.startswith("vit"):
            # I can create the model accordingly, based on pretrained weights
            #model = timm.create_model(backbone_name, pretrained=True)
            model = torchvision.models.vit_base_patch16_224(pretrained=True)
        else: #Non VIT architectures
            weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
            model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        if backbone_name.startswith("vit"):
            model = torchvision.models.vit_base_patch16_224(pretrained=True)
        else:
            model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model

def get_backbone(backbone_name : str) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
        backbone = torch.nn.Sequential(*layers) 
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    #UPDATE: Handle the case for ViT
    elif backbone_name == "vit_base_patch16_224":
        import torchvision
        backbone = torchvision.models.vit_base_patch16_224(pretrained=True)
        for p in backbone.parameters():
            p.requires_grad = False
        logging.debug(f"Train the last layers of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove the head of the model.
        backbone = torch.nn.Sequential(*layers)
        features_dim = 768

    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
        backbone = torch.nn.Sequential(*layers) 
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    else:
        raise ValueError(f"Invalid backbone name: {backbone_name}")
    
    return backbone, features_dim
