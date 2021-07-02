from fastai.callback.all import *
from fastai.torch_basics import *
from fastai.torch_core import *

# temp bug fix
# https://github.com/fastai/fastai/issues/3384
#TensorMultiCategory.register_func(Tensor.__getitem__, TensorMultiCategory, TensorBBox)

__all__ = ['RCNNAdapter']


class ObjDetAdapter(Callback):
    '''Callback to convert batches from fastai's dataloader 
    to the expected input of object detection models'''
    
    def __init__(self, pad_idx=0, debug=False): 
        self.pad_idx = pad_idx
        self.debug = debug
        
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
            filt = d["labels"]!=self.pad_idx
            for k in keys:
                d[k] = d[k][filt]
                
            # remove empty bboxes
            filt = (d["boxes"][:,0]-d["boxes"][:,2])*(d["boxes"][:,1]-d["boxes"][:,3])!=0

            #filt = torch.eq(d["boxes"], tensor([[0.,0.,0.,0.]], device=dev)).all(dim=1)
            for k in keys:
                d[k] = d[k][filt]
            
            # scale bboxes back
            d["boxes"] = (d["boxes"]+1.)*tensor([w,h,w,h], device=dev)*0.5

            if with_mask:
                # filter out objects with empty masks
                filt = d["masks"].sum(dim=-1).sum(dim=-1)==0 
                for k in keys:
                    d[k] = d[k][~filt]
    
            #new_y.append(d)
        return [x1],[y] # xb,yb
    
    
class RCNNAdapter(Callback):
    '''Callback to convert batches from fastai's dataloader 
    to the expected input of fasterrcnn/maskrcnn'''
    
    def __init__(self, pad_idx=0): self.pad_idx = pad_idx
        
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
            filt = d["labels"]!=self.pad_idx
            for k in keys:
                d[k] = d[k][filt]
                
            # remove empty bboxes
            filt = (d["boxes"][:,0]-d["boxes"][:,2])*(d["boxes"][:,1]-d["boxes"][:,3])!=0

            #filt = torch.eq(d["boxes"], tensor([[0.,0.,0.,0.]], device=dev)).all(dim=1)
            for k in keys:
                d[k] = d[k][filt]
            
            # scale bboxes back
            d["boxes"] = (d["boxes"]+1.)*tensor([w,h,w,h], device=dev)*0.5

            if with_mask:
                # filter out objects with empty masks
                filt = d["masks"].sum(dim=-1).sum(dim=-1)==0 
                for k in keys:
                    d[k] = d[k][~filt]
    
            #new_y.append(d)
        return [x1],[y] # xb,yb
