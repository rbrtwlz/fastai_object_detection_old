from fastai.callback.all import *
from fastai.torch_basics import *
from fastai.torch_core import *

__all__ = ['RCNNAdapter']


class RCNNAdapter(Callback):
    
    def __init__(self, na_idx=0): self.na_idx = na_idx
        
    def after_create(self):
        self.learn.save_xb = []
        self.learn.save_yb = []

    def before_batch(self):
        self.learn.save_xb = self.learn.xb
        self.learn.save_yb = self.learn.yb
        
        xb,yb = self.transform_batch(self.learn.xb[0], *self.learn.yb)
        self.learn.xb = [xb[0],yb[0]]
        self.learn.yb = []        
        
    def after_pred(self):
        
        # leave yb empty to skip loss calc
        loss = sum(loss for loss in self.learn.pred.values())
        self.learn.loss_grad = loss
        self.learn.loss = self.learn.loss_grad.clone()
        
    def after_loss(self):
        
        # set yb for recorder to log train loss
        self.learn.yb = self.learn.save_yb

        if not self.learn.training:
            # set model to eval to get predictions
            self.learn.model.eval()
            # transform batch to fasterrcnn´s expected input
            xb,yb = self.transform_batch(self.learn.save_xb[0], *self.learn.save_yb)
            # save predictions
            self.learn.pred = self.learn.model(xb[0])
            self.learn.yb = yb
            self.learn.model.train()
            

    def after_batch(self):
        self.learn.model.train()

        
    def before_validate(self):
        # set model to train to get valid loss
        self.learn.model.train()  

        
    def transform_batch(self,x1,*yb):
        yb = [*yb]
        
        # check if with or without mask
        with_mask = len(yb) == 3

        bs,c,h,w = x1.shape
        dev = x1.device

        y={}
        
        keys = ["masks", "boxes", "labels"] if with_mask else ["boxes", "labels"]
        for i,k in enumerate(keys):
            y[k] = [e for e in yb[i]]
            
        y = [dict(zip(y,t)) for t in zip(*y.values())] # dict of lists to list of dicts

        #new_y = []
        for d in y:
            # remove padding
            filt = d["labels"]!=self.na_idx
            for k in keys:
                d[k] = d[k][filt]
                
            # remove empty bboxes
            filt = torch.eq(d["boxes"], tensor([[0.,0.,0.,0.]], device=dev)).all(dim=1)
            for k in keys:
                d[k] = d[k][~filt]
            
            # scale bboxes back
            d["boxes"] = (d["boxes"]+1.)*tensor([w,h,w,h], device=dev)*0.5

            if with_mask:
                # filter out objects with empty masks
                filt = d["masks"].sum(dim=-1).sum(dim=-1)==0 
                for k in keys:
                    d[k] = d[k][~filt]
    
            #new_y.append(d)
        return [x1],[y] # xb,yb
