
from mean_average_precision import MetricBuilder
from fastai.metrics import AvgMetric
from functools import partial

__all__ = ['mAP_at_IoU40', 'mAP_at_IoU50', 'mAP_at_IoU60', 'mAP_at_IoU70', 'mAP_at_IoU80', 'mAP_at_IoU90']

def create_metric_samples(preds, targs):
    pred_samples = []
    for pred in preds:
        res = torch.cat([pred["boxes"], pred["labels"].unsqueeze(-1), pred["scores"].unsqueeze(-1)], dim=1) 
        pred_np = np.array(res.detach().cpu())
        pred_samples.append(pred_np)

    targ_samples = []
    for targ in targs: # targs : yb[0]
        #print(targ["boxes"].shape,targ["labels"].shape)
        targ = torch.cat([targ["boxes"],targ["labels"].unsqueeze(-1)], dim=1)
        targ = torch.cat([targ, torch.zeros([targ.shape[0], 2], device=targ.device)], dim=1)
        targ_np = np.array(targ.detach().cpu())
        targ_samples.append(targ_np)

    return [i for i in zip(pred_samples, targ_samples)]


def m_ap_metric(preds, targs, iou_thresholds=0.4):
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=len(dls.vocab))
    for sample_preds, sample_targs in create_metric_samples(preds, targs):
        metric_fn.add(sample_preds, sample_targs)
    metric_batch =  metric_fn.value(iou_thresholds=iou_thresholds)['mAP']
    return metric_batch


@patch
def accumulate(x:AvgMetric, learn):
    bs = len(learn.yb[0])
    x.total += learn.to_detach(x.func(learn.pred, *learn.yb))*bs
    x.count += bs
    
    
mAP_at_IoU40 = AvgMetric(m_ap_metric)
mAP_at_IoU50 = AvgMetric(partial(m_ap_metric, iou_thresholds=0.5))
mAP_at_IoU60 = AvgMetric(partial(m_ap_metric, iou_thresholds=0.6))
mAP_at_IoU70 = AvgMetric(partial(m_ap_metric, iou_thresholds=0.7))
mAP_at_IoU80 = AvgMetric(partial(m_ap_metric, iou_thresholds=0.8))
mAP_at_IoU90 = AvgMetric(partial(m_ap_metric, iou_thresholds=0.9))
