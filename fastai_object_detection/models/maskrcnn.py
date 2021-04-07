from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torchvision.ops.misc import FrozenBatchNorm2d
from functools import partial

__all__ = ['get_MaskRCNN', 'maskrcnn_resnet18', 'maskrcnn_resnet34', 'maskrcnn_resnet50', 'maskrcnn_resnet101', 'maskrcnn_resnet152']

model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}

def get_MaskRCNN(arch_str, num_classes, pretrained=False, pretrained_backbone=True,
                 trainable_layers=5, **kwargs):
    
    #if pretrained: pretrained_backbone = False
        
    backbone = resnet_fpn_backbone(arch_str, pretrained=pretrained_backbone, trainable_layers=trainable_layers)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    
    if pretrained:
        try:
            
            pretrained_dict = load_state_dict_from_url(model_urls['maskrcnn_'+arch_str+'_fpn_coco'],
                                                       progress=True)
            model_dict = model.state_dict()
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
                     
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            
            for module in model.modules():
                if isinstance(module, FrozenBatchNorm2d):
                    module.eps = 0.0
                    
        except Exception as e: 
            #print(e)            
            print("No pretrained coco model found for maskrcnn_"+arch_str)
            print("This does not affect the backbone.")
            
    return model
  
  
maskrcnn_resnet18 = partial(get_MaskRCNN, arch_str="resnet18")
maskrcnn_resnet34 = partial(get_MaskRCNN, arch_str="resnet34")
maskrcnn_resnet50 = partial(get_MaskRCNN, arch_str="resnet50")
maskrcnn_resnet101 = partial(get_MaskRCNN, arch_str="resnet101")
maskrcnn_resnet152 = partial(get_MaskRCNN, arch_str="resnet152")
