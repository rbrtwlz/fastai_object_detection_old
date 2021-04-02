from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN

def get_FasterRCNN(arch_str, num_classes, **kwargs):
    
    backbone = resnet_fpn_backbone(arch_str, pretrained=True, trainable_layers=5)
    
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_fg_iou_thresh=0.5,
                       box_bg_iou_thresh=0.5,
                       box_score_thresh=0.6,
                       **kwargs
                      )
    
    return model.train()
