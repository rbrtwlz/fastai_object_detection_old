from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.ops.misc import FrozenBatchNorm2d
from functools import partial

__all__ = ['get_FasterRCNN', 'fasterrcnn_resnet18', 'fasterrcnn_resnet34', 'fasterrcnn_resnet50', 'fasterrcnn_resnet101', 'fasterrcnn_resnet152']


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'fasterrcnn_mobilenet_v3_large_320_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
    'fasterrcnn_mobilenet_v3_large_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth'
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


fasterrcnn_resnet18 = partial(get_FasterRCNN, arch_str="resnet18")
fasterrcnn_resnet34 = partial(get_FasterRCNN, arch_str="resnet34")
fasterrcnn_resnet50 = partial(get_FasterRCNN, arch_str="resnet50")
fasterrcnn_resnet101 = partial(get_FasterRCNN, arch_str="resnet101")
fasterrcnn_resnet152 = partial(get_FasterRCNN, arch_str="resnet152")

