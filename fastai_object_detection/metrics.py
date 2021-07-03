
from mean_average_precision import MetricBuilder
#from mean_average_precision import MeanAveragePrecision
#from fastai.metrics import AvgMetric
from fastai.metrics import Metric
from fastai.torch_basics import *
from fastai.torch_core import *
#from functools import partial

__all__ = ['mAP_at_IoU40', 'mAP_at_IoU50', 'mAP_at_IoU60', 'mAP_at_IoU70', 'mAP_at_IoU80', 
           'mAP_at_IoU90', 'mAP_at_IoU50_95', 'create_mAP_metric']        

class mAP_Metric():
    "Metric to calculate mAP for different IoU thresholds"
    def __init__(self, iou_thresholds, name, remove_background_class=True):
        self.__name__ = name
        self.iou_thresholds = iou_thresholds
        self.remove_background_class = remove_background_class
        
    def __call__(self, preds, targs, num_classes):
        if self.remove_background_class:
            num_classes=num_classes-1
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=num_classes)
        for sample_preds, sample_targs in self.create_metric_samples(preds, targs):
            metric_fn.add(sample_preds, sample_targs)
        metric_batch = metric_fn.value(iou_thresholds=self.iou_thresholds,
                                       recall_thresholds=np.arange(0., 1.01, 0.01), 
                                       mpolicy='soft')['mAP']
        return metric_batch
    
    def create_metric_samples(self, preds, targs):
        pred_samples = []
        for pred in preds:
            res = torch.cat([pred["boxes"], pred["labels"].unsqueeze(-1), pred["scores"].unsqueeze(-1)], dim=1) 
            pred_np = res.detach().cpu().numpy()
            if self.remove_background_class:
                # first idx is background
                try:
                    pred_np= pred_np-np.array([0,0,0,0,1,0])
           	    except: pass
            pred_samples.append(pred_np)

        targ_samples = []
        for targ in targs: # targs : yb[0]
            targ = torch.cat([targ["boxes"],targ["labels"].unsqueeze(-1)], dim=1)
            targ = torch.cat([targ, torch.zeros([targ.shape[0], 2], device=targ.device)], dim=1)
            targ_np = np.array(targ.detach().cpu())
            if self.remove_background_class:
                # first idx is background 
                try:
                    targ_np= targ_np-np.array([0,0,0,0,1,0,0])
                except: pass
            targ_samples.append(targ_np)

        return [s for s in zip(pred_samples, targ_samples)]
    
    
class AvgMetric_ObjectDetection(Metric):
    "Average the values of `func` taking into account potential different batch sizes"
    def __init__(self, func): self.func = func
    def reset(self): self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = len(learn.xb[0])
        self.total += learn.to_detach(self.func(learn.pred, *learn.yb, num_classes=learn.num_classes))*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self): return self.func.func.__name__ if hasattr(self.func, 'func') else  self.func.__name__
    
def create_mAP_metric(iou_tresh=np.arange(0.5, 1.0, 0.05), metric_name="mAP@IoU 0.5:0.95", remove_background_class=False):
    return AvgMetric_ObjectDetection(mAP_Metric(iou_tresh, metric_name, remove_background_class=remove_background_class)) 
    
    
mAP_at_IoU40 = AvgMetric_ObjectDetection(mAP_Metric(0.4, "mAP@IoU>0.4", remove_background_class=True))
mAP_at_IoU50 = AvgMetric_ObjectDetection(mAP_Metric(0.5, "mAP@IoU>0.5", remove_background_class=True))
mAP_at_IoU60 = AvgMetric_ObjectDetection(mAP_Metric(0.6, "mAP@IoU>0.6", remove_background_class=True))
mAP_at_IoU70 = AvgMetric_ObjectDetection(mAP_Metric(0.7, "mAP@IoU>0.7", remove_background_class=True))
mAP_at_IoU80 = AvgMetric_ObjectDetection(mAP_Metric(0.8, "mAP@IoU>0.8", remove_background_class=True))
mAP_at_IoU90 = AvgMetric_ObjectDetection(mAP_Metric(0.9, "mAP@IoU>0.9", remove_background_class=True))
mAP_at_IoU50_95 = AvgMetric_ObjectDetection(mAP_Metric(np.arange(0.5, 1.0, 0.05), "mAP@IoU 0.5:0.95", remove_background_class=True)) 

