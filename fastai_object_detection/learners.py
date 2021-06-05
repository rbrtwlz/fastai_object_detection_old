from fastprogress.fastprogress import progress_bar
from fastai.vision.all import *
from .callbacks import *

# it's intended to have two classes here, even if it would be easy to merge them

__all__ = ['fasterrcnn_learner', 'maskrcnn_learner']

def rcnn_split(m):
    "Default split of rcnn model between transform,backbone and rpn,roi_heads"
    m = L(c for c in m.children())
    return L(m[:2], m[2:]).map(params)
    #return L(m).map(params)


class fasterrcnn_learner(Learner):
    """ fastai-style learner to train fasterrcnn models """
    def __init__(self, dls, model, pretrained=True, pretrained_backbone=True, num_classes=None,
                 # learner args
                 loss_func=noop, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
                 model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95),
                 # other model args
                 **kwargs):
                
        if num_classes is None: num_classes = len(dls.vocab)
        
        if cbs is None: cbs = [RCNNAdapter()]
        else: cbs = L(RCNNAdapter())+L(cbs)
            
        model = model(num_classes=num_classes, pretrained=pretrained, pretrained_backbone=pretrained_backbone, **kwargs)
        
        if splitter is None: splitter = rcnn_split
            
        super().__init__(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn,
                   moms=moms)
        #learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=rcnn_split, cbs=cbs,
        #           metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn,
        #           moms=moms)
        #learn.splitter=splitter
        #return learn
        
    def get_preds(self, items=None, item_tfms=None, batch_tfms=None, box_score_thresh=0.05, max_n=None):
        
        if items is not None:
            if item_tfms is None: item_tfms = [Resize(800, method="pad", pad_mode="zeros")]
            dblock = DataBlock(
                blocks=(ImageBlock(cls=PILImage)),
                item_tfms=item_tfms,
                batch_tfms=batch_tfms)
            test_dl = dblock.dataloaders(items).test_dl(items, bs=self.dls.bs)
        else:
            test_dl = self.dls.valid
            
        inputs,preds = [],[]
        with torch.no_grad():
            for i,batch in enumerate(progress_bar(test_dl)):
                self.model.eval()
                preds.append(self.model(*batch))
                inputs.append(*batch)
                self.model.train()
                if max_n is not None:
                    if len(inputs)>=max_n:
                        break
        # preds: num_batches x bs x dict["boxes", "labels", "scores"]
        # flatten:
        preds = [i for p in preds for i in p]
        inputs = [i for inp in inputs for i in inp]
        
        preds = [torch.cat([p["boxes"],p["labels"].unsqueeze(1),p["scores"].unsqueeze(1)], dim=1) 
                 for p in preds]
        
        # only preds with score > box_score_thresh
        preds = [p[p[:,5]>box_score_thresh] for p in preds]
        
        boxes = [p[:,:4] for p in preds]
        labels = [p[:,4] for p in preds]
        scores = [p[:,5] for p in preds]
        
        return inputs, boxes, labels, scores
    
    
    def show_results(self, items=None, max_n=9, item_tfms=None, batch_tfms=None, box_score_thresh=0.6):
        inputs, bboxes, labels, scores  = self.get_preds(items=items, box_score_thresh=box_score_thresh, max_n=max_n)
        #idx = 10
        for idx in range(len(inputs)):
            if idx >= max_n: break
            fig, ax = plt.subplots(figsize=(8,8))
            TensorImage(inputs[idx]).show(ax=ax)
            LabeledBBox(TensorBBox(bboxes[idx]), [self.dls.vocab[int(l.item())] 
                                                  for l in labels[idx]]).show(ax)
        

class maskrcnn_learner(Learner):
    """ fastai-style learner to train maskrcnn models """
    def __init__(self, dls, model, pretrained=True, pretrained_backbone=True, num_classes=None,
                 # learner args
                 loss_func=noop, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
                 model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95),
                 # other model args
                 **kwargs):    
        
        if num_classes is None: num_classes = len(dls.vocab)
        
        if cbs is None: cbs = [RCNNAdapter()]
        else: cbs = L(RCNNAdapter())+L(cbs)
            
        model = model(num_classes=num_classes, pretrained=pretrained, pretrained_backbone=pretrained_backbone, **kwargs)
        
        super().__init__(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn,
                   moms=moms)
        
    def get_preds(self, items, item_tfms=None, batch_tfms=None, box_score_thresh=0.05, bin_mask_thresh=None):
        if item_tfms is None: item_tfms = [Resize(800, method="pad", pad_mode="zeros")]
        dblock = DataBlock(
            blocks=(ImageBlock(cls=PILImage)),
            item_tfms=item_tfms,
            batch_tfms=batch_tfms)
        test_dl = dblock.dataloaders(items).test_dl(items, bs=self.dls.bs)
        inputs,preds = [],[]
        with torch.no_grad():
            for i,batch in enumerate(progress_bar(test_dl)):
                self.model.eval()
                preds.append(self.model(*batch))
                inputs.append(*batch)
                self.model.train()
        # preds: num_batches x bs x dict["boxes", "labels", ...]
        # flatten:
        preds = [i for p in preds for i in p]
        inputs = [i.cpu() for inp in inputs for i in inp]

        # maskrcnn pred shapes
        # masks: [N, 1, H, W]
        # boxes: [N, 4]
        # labels: [N]
        # scores: [N]

        # filter out predictions under threshold
        filt = [p["scores"]>box_score_thresh for p in preds]

        masks = [p["masks"][filt[i]].cpu() for i,p in enumerate(preds)]
        boxes = [p["boxes"][filt[i]].cpu() for i,p in enumerate(preds)]
        labels = [p["labels"][filt[i]].cpu() for i,p in enumerate(preds)]
        scores = [p["scores"][filt[i]].cpu() for i,p in enumerate(preds)]
        
        print(len(masks))
        # by default returns masks in [N, 1, H, W] with activations
        # if you want binary masks in [N, H, W] set bin_mask_thresh 
        if bin_mask_thresh is not None:
            for i,m in enumerate(masks):
                masks[i] = torch.where(m > bin_mask_thresh, 1, 0).squeeze(1)

        return inputs, masks, boxes, labels, scores
    
    
    def show_results(self, items, max_n=9, box_score_thresh=0.6, bin_mask_thresh=0.5, **kwargs):
        inputs, masks, bboxes, labels, scores  = self.get_preds(items=items, box_score_thresh=box_score_thresh)
        
        for i,m in enumerate(masks):
            background = torch.ones([1,1,m.shape[-2],m.shape[-1]]) * bin_mask_thresh 
            m = torch.cat([background, m])
            masks[i] = torch.argmax(m, dim=0).squeeze(0)
        
        #idx = 10
        for idx in range(len(inputs)):
            if idx >= max_n: break
            fig, ax = plt.subplots(figsize=(8,8))
            TensorImage(inputs[idx]).show(ax=ax),
            TensorMask(masks[idx]).show(ax),
            LabeledBBox(TensorBBox(bboxes[idx]), [self.dls.vocab[int(l.item())] 
                                                  for l in labels[idx]]).show(ax)
        
