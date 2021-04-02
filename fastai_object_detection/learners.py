from fastprogress.fastprogress import progress_bar
from fastai.vision.all import *
from .callbacks import *


__all__ = ['fastrcnn_learner']


class fastrcnn_learner(Learner):
    def __init__(self, dls, model, cbs=None, **kwargs):
        if cbs is not None: cbs = L(FasterRCNNAdapter())+L(cbs)
        else: cbs = [FasterRCNNAdapter()]
        model = model(num_classes=len(dls.vocab))
        super().__init__(dls, model, loss_func=noop, cbs=cbs, **kwargs)
        
    def get_preds(self, items, dl=None, item_tfms=None, batch_tfms=None):
        if item_tfms is None: item_tfms = [Resize(800)]
        if dl is None:
            dblock = DataBlock(
                blocks=(ImageBlock(cls=PILImage)),
                item_tfms=item_tfms,
                batch_tfms=batch_tfms)
            test_dl = dblock.dataloaders(items).test_dl(items, bs=self.dls.bs)
        else: test_dl = dl
        inputs,preds = [],[]
        with torch.no_grad():
            for i,batch in enumerate(progress_bar(test_dl)):
                self.model.eval()
                preds.append(self.model(*batch))
                inputs.append(*batch)
                self.model.train()
        # preds: num_batches x bs x dict["boxes", "labels", "scores"]
        # flatten:
        preds = [i for p in preds for i in p]
        inputs = [i for inp in inputs for i in inp]
        
        preds = [torch.cat([p["boxes"],p["labels"].unsqueeze(1),p["scores"].unsqueeze(1)], dim=1) 
                 for p in preds]
        
        boxes = [p[:,:4] for p in preds]
        labels = [p[:,4] for p in preds]
        scores = [p[:,5] for p in preds]
        
        return inputs, boxes, labels, scores
    
    
    def show_results(self, items, max_n=9,  **kwargs):

        inputs, bboxes, labels, scores  = self.get_preds(items)
        #idx = 10
        for idx in range(len(inputs)):
            if idx >= max_n: break
            fig, ax = plt.subplots(figsize=(8,8))
            TensorImage(inputs[idx]).show(ax=ax)
            LabeledBBox(TensorBBox(bboxes[idx]), [self.dls.vocab[int(l.item())] 
                                                  for l in labels[idx]]).show(ax)
        

        
        
        
