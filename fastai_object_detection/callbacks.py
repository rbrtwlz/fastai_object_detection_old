from fastai.callback.all import *
from fastai.torch_basics import *
from fastai.torch_core import *

__all__ = ['RCNNAdapter']


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
        if len(yb) == 3: with_mask=True
        else: with_mask=False

        bs,c,h,w = x1.shape

        y={}
        if with_mask:
            y["masks"] = [m for m in yb[0]] # len: bs
            y["boxes"] = [b for b in yb[1]] # len: bs
            y["labels"] = [l for l in yb[2]] # len: bs
        else:
            y["boxes"] = [b for b in yb[0]] # len: bs
            y["labels"] = [l for l in yb[1]] # len: bs
            
        y = [dict(zip(y,t)) for t in zip(*y.values())] # dict of lists to list of dicts

        new_y = []
        for dict_ in y:
            #empty=False
            # remove padding
            a = dict_["boxes"]
            dict_["boxes"] = a[~torch.all(torch.eq(a,tensor([0.,0.,0.,0.], device=a.device)), dim=1)]
            a = dict_["labels"]
            dict_["labels"] = a[a!=self.na_idx]
            # scale back
            dict_["boxes"] = (dict_["boxes"]+1)* (h/2) 
            boxes = dict_["boxes"]
            if with_mask:
                if len(boxes) == 0:
                    dict_["masks"] = torch.empty([0,h,w], dtype=torch.uint8, device=boxes.device)
                # mask to stacked binary masks
                else:
                    m = dict_["masks"]
                    #print("mask shape")
                    #print(m.shape)
                    #print("mask unique")
                    #print(str(m.unique()))
                    m = torch.stack([torch.where(m==i+1,1,0) for i in range(len(boxes))]) # better pytorch solution?
                    dict_["masks"] = m
                    #print("binary masks shape")
                    #print(m.shape)

                    filt = m.sum(dim=-1).sum(dim=-1)!=0 # find empty binary segmentation masks
                    #print("filter:")
                    #print(filt)
                    #print("bbox before filter:")
                    #print(dict_["boxes"])
                    #print(dict_["boxes"].shape)
                    dict_["masks"] = dict_["masks"][filt]
                    dict_["labels"] = dict_["labels"][filt]
                    dict_["boxes"] = dict_["boxes"][filt]
                    #print("bbox after filter")
                    #print(dict_["boxes"])
                    #print(dict_["boxes"].shape)
                    #print("mask shape after filter")
                    #print(dict_["masks"].shape)
            new_y.append(dict_)
        return [x1],[new_y] # xb,yb
