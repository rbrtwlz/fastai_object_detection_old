# fastai_object_detection

Extension of the fastai library to include object recognition.

This package makes pytorch's FasterRCNN and MaskRCNN available for fastai users by using a callback which convers the batches to the required input for FastRCNN or MaskRCNN. It comes with a fastai's Dataloaders for object detection, prepared and easy to use models and some metrics to measure generated bounding boxes (mAP). So you can train models for object detection in the simple fastai way with one of the two included learner classes:

