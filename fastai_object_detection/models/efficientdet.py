import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms
from torchvision.models.utils import load_state_dict_from_url
from .efficientdet_source import FocalLoss, BBoxTransform, ClipBoxes, EfficientDetBackbone

__all__ = ['get_efficientdet_model', 'efficientdet_d0', 'efficientdet_d1', 'efficientdet_d2', 'efficientdet_d3', 
           'efficientdet_d4', 'efficientdet_d5', 'efficientdet_d6', 'efficientdet_d7']

class EffDetModelWrapper(nn.Module):
    def __init__(self, num_classes, compound_coef=0, pretrained=True, pretrained_backbone=True, 
                 nms_score_thresh=0.05, nms_iou_thresh=0.50, ratios='[(1.0,1.0),(1.4,0.7),(0.7,1.4)]', 
                 scales='[2**0, 2**(1.0/3.0), 2**(2.0/3.0)]', **kwargs):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = EfficientDetBackbone(num_classes=num_classes, compound_coef=compound_coef, ratios=eval(ratios), scales=eval(scales))
        self.model.train()

        self.training = True
        self.nms_score_thresh = nms_score_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, *x):

        imgs, targets = x if len(x)==2 else (x[0], None)
        imgs, targets = self.preprocess(imgs, targets)
        features, regression, classification, anchors = self.model(imgs)

        if targets is not None:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, targets)
            return {"cls_loss":cls_loss, "reg_loss":reg_loss}
        else:
            preds = self.postprocess(imgs, anchors, regression, classification)
            return preds

    def train(self):
        self.model.train()
        self.training = True

    def eval(self):
        self.model.eval()
        self.training = False

    def preprocess(self, imgs, targets=None):
        if targets is None:
            annotations = None
        else: 
            bboxes = [d["boxes"] for d in targets]
            labels = [d["labels"] - 1 for d in targets] # 0 is background in dataloader, but first class in model
            annotations = [torch.cat([b,l.unsqueeze(1)], dim=1) for b,l in zip(bboxes, labels)]
            # padding with -1
            max_len = max([len(b) for b in annotations])
            annotations = torch.stack([torch.cat([b, b.new_ones([max_len-len(b),5])*-1], dim=0) for b in annotations])
        return imgs, annotations

    def postprocess(self, x, anchors, regression, classification):
        # modified from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/utils/utils.py
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, x)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.nms_score_thresh)[:, :, 0]
        out = []
        for i in range(x.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                out.append({
                    'boxes': torch.tensor(()),
                    'labels': torch.tensor(()),
                    'scores': torch.tensor(()),
                })
                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)
            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=self.nms_iou_thresh)

            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx] + 1 # 0 is background and gets removed in metric, but is first class in model
                scores_ = scores_[anchors_nms_idx]
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]

                out.append({
                    'boxes': boxes_.cpu(),
                    'labels': classes_.cpu(),
                    'scores': scores_.cpu(),
                })
            else:
                out.append({
                    'boxes': torch.tensor(()),
                    'labels': torch.tensor(()),
                    'scores': torch.tensor(()),
                })

        return out
      
      
effdet_model_urls = {
    "efficientdet-d0": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth",
    "efficientdet-d1": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth",
    "efficientdet-d2": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth",
    "efficientdet-d3": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth",
    "efficientdet-d4": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth",
    "efficientdet-d5": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth",
    "efficientdet-d6": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth",
    "efficientdet-d7": "https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d7.pth"
}


def get_efficientdet_model(num_classes, compound_coef=0, pretrained_backbone=True, pretrained=True, nms_score_thresh=0.05, nms_iou_thresh=0.50 , **kwargs):

    arch_str = f"efficientdet-d{compound_coef}"
    model = EffDetModelWrapper(num_classes=num_classes, compound_coef=compound_coef, nms_score_thresh=nms_score_thresh, nms_iou_thresh=nms_iou_thresh, **kwargs)

    if pretrained or pretrained_backbone:
        try:
            pretrained_dict = load_state_dict_from_url(effdet_model_urls[arch_str], progress=True)
            model_dict = model.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            model_dict.update(pretrained_dict) 
            model.model.load_state_dict(model_dict)
        except:
            print("Error loading pretrained model")

    return model 
  

efficientdet_d0 = partial(get_efficientdet_model, compound_coef=0)  
efficientdet_d1 = partial(get_efficientdet_model, compound_coef=1)
efficientdet_d2 = partial(get_efficientdet_model, compound_coef=2)  
efficientdet_d3 = partial(get_efficientdet_model, compound_coef=3)
efficientdet_d4 = partial(get_efficientdet_model, compound_coef=4)  
efficientdet_d5 = partial(get_efficientdet_model, compound_coef=5)
efficientdet_d6 = partial(get_efficientdet_model, compound_coef=6)  
efficientdet_d7 = partial(get_efficientdet_model, compound_coef=7)
