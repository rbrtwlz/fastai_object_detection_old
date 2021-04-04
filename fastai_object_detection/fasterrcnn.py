from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from functools import partial

__all__ = ['get_FasterRCNN', 'fasterrcnn_resnet18', 'fasterrcnn_resnet34', 'fasterrcnn_resnet50', 'fasterrcnn_resnet101', 'fasterrcnn_resnet152']

def get_FasterRCNN(arch_str, num_classes, pretrained=True, pretrained_backbone=True, **kwargs):
    
    #if pretrained == True: pretrained_backbone=False
        
    backbone = resnet_fpn_backbone(arch_str, pretrained=pretrained_backbone, trainable_layers=5)
    
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_fg_iou_thresh=0.5,
                       box_bg_iou_thresh=0.5,
                       #box_score_thresh=0.6,
                       **kwargs
                      )
    
    if pretrained:
        try:
            state_dict = load_state_dict_from_url(model_urls['fasterrcnn_"+arch_str+"_fpn_coco'],
                                                  progress=progress)
            model.load_state_dict(state_dict)
        except Exception as e: 
            print(e)
            print("No pretrained model found for "+arch_str)
    
    return model.train()


fasterrcnn_resnet18 = partial(get_FasterRCNN, arch_str="resnet18")
fasterrcnn_resnet34 = partial(get_FasterRCNN, arch_str="resnet34")
fasterrcnn_resnet50 = partial(get_FasterRCNN, arch_str="resnet50")
fasterrcnn_resnet101 = partial(get_FasterRCNN, arch_str="resnet101")
fasterrcnn_resnet152 = partial(get_FasterRCNN, arch_str="resnet152")

