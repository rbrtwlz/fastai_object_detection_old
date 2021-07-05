import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection import FasterRCNN
from torchvision.ops.misc import FrozenBatchNorm2d
from functools import partial
from fastai_object_detection.models.swin_transformer_backbone import SwinTransformer
from fastai.vision.all import default_device


__all__ = ['get_FasterRCNN', 'fasterrcnn_resnet18', 'fasterrcnn_resnet34', 'fasterrcnn_resnet50', 'fasterrcnn_resnet101', 'fasterrcnn_resnet152',
          'get_FasterRCNN_SWIN', 'fasterrcnn_swinT', 'fasterrcnn_swinS', 'fasterrcnn_swinB', 'fasterrcnn_swinL']


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
    'fasterrcnn_mobilenet_v3_large_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth',
    'swin_tiny_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    'swin_small_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    'swin_base_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
    'swin_base_384': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',
    'swin_large_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',
    'swin_large_384': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth'
}

def get_FasterRCNN(arch_str, num_classes, pretrained=True, pretrained_backbone=True, 
                   trainable_layers=5, **kwargs):
    
    #if pretrained == True: pretrained_backbone=False
        
    backbone = resnet_fpn_backbone(arch_str, pretrained=pretrained_backbone, trainable_layers=trainable_layers)
    
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_fg_iou_thresh=0.5,
                       box_bg_iou_thresh=0.5,
                       image_mean = [0.0, 0.0, 0.0], # already normalized by fastai
                       image_std = [1.0, 1.0, 1.0],
                       #min_size = 1,
                       #box_score_thresh=0.6,
                       **kwargs
                      )
    

    if pretrained:
        try:
            pretrained_dict = load_state_dict_from_url(model_urls['fasterrcnn_'+arch_str+'_fpn_coco'], progress=True)
            model_dict = model.state_dict()
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
                     
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            #overwrite_eps(model, 0.0)
            for module in model.modules():
                if isinstance(module, FrozenBatchNorm2d):
                    module.eps = 0.0
            
        except Exception as e: 
            #print(e)
            print("No pretrained coco model found for fasterrcnn_"+arch_str)
            print("This does not affect the backbone.")
            
    return model.train()


def get_FasterRCNN_SWIN(arch_str, num_classes, pretrained=False, pretrained_backbone=True, **kwargs):
    anchor_sizes = ((32,), (64,), (128,), (256,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1','2','3'],
    #                                                output_size=7,
    #                                                sampling_ratio=2)
    
    img_size = 224 if arch_str in "swin_tiny swin_small".split() else 384
    window_size = 7 if arch_str in "swin_tiny swin_small".split() else 12
    depths = [2, 2, 6, 2] if arch_str=="swin_tiny" else [2, 2, 18, 2]
    
    scale_factors = {"swin_tiny":1.0, "swin_small":1.5, "swin_base":2.0, "swin_large":2.0}
    sf = scale_factors[arch_str]
    embed_dim = int(96*sf)
    fpn_cin = [int(96*sf*2**i) for i in range(4)]
    #fpn_cin = [int(i*sf) for i in [96, 192, 384, 768]]
    
    backbone = SwinTransformerFPN(img_size=img_size, window_size=window_size, embed_dim=embed_dim, 
                                  depths=depths, fpn_cin=fpn_cin, fpn_cout=256)
    
    if pretrained_backbone:
        sd = load_state_dict_from_url(model_urls[f'{arch_str}_{img_size}'], 
                                      progress=True, map_location=default_device())['model']
        sd_model = backbone.state_dict()
        sd = {k: v for k, v in sd.items() if k in sd_model.keys()}
        sd_model.update(sd)
        backbone.load_state_dict(sd_model)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       #box_roi_pool=roi_pooler,
                       box_fg_iou_thresh=0.5,
                       box_bg_iou_thresh=0.5,
                       image_mean = [0.0, 0.0, 0.0], # already normalized by fastai
                       image_std = [1.0, 1.0, 1.0],
                       #min_size=IMG_SIZE,
                       #max_size=IMG_SIZE,
                       **kwargs
                      )
                       
    return model.train()


class SwinTransformerFPN(nn.Module):
    def __init__(self, img_size=224, window_size=7, embed_dim=96, depths=[2, 2, 6, 2], fpn_cin=[96, 192, 384, 768], fpn_cout=256):
        super().__init__()
        self.body = SwinTransformer(pretrain_img_size=img_size, patch_size=4, in_chans=3, 
                                    embed_dim=embed_dim, depths=depths, num_heads=[3, 6, 12, 24],
                                    window_size=window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, 
                                    attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=torch.nn.modules.normalization.LayerNorm,
                                    ape=False,patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False)
        
        self.fpn = FeaturePyramidNetwork(in_channels_list=fpn_cin,  out_channels=fpn_cout)
        self.out_channels = fpn_cout
    
    def forward(self, x):
        x = self.body(x)
        features = {f"{i}":v for i,v in enumerate(x)}
        return self.fpn(features)


fasterrcnn_resnet18 = partial(get_FasterRCNN, arch_str="resnet18")
fasterrcnn_resnet34 = partial(get_FasterRCNN, arch_str="resnet34")
fasterrcnn_resnet50 = partial(get_FasterRCNN, arch_str="resnet50")
fasterrcnn_resnet101 = partial(get_FasterRCNN, arch_str="resnet101")
fasterrcnn_resnet152 = partial(get_FasterRCNN, arch_str="resnet152")
fasterrcnn_swinT = partial(get_FasterRCNN_SWIN, arch_str="swin_tiny")
fasterrcnn_swinS = partial(get_FasterRCNN_SWIN, arch_str="swin_small")
fasterrcnn_swinB = partial(get_FasterRCNN_SWIN, arch_str="swin_base")
fasterrcnn_swinL = partial(get_FasterRCNN_SWIN, arch_str="swin_large")

