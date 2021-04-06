# fastai_object_detection

Extension of the fastai library to include object recognition.

This package makes pytorch's FasterRCNN and MaskRCNN available for [fastai](https://www.fast.ai/) users by using a callback which convers the batches to the required input for FastRCNN or MaskRCNN. It comes with a fastai `Dataloaders` class for object detection, prepared and easy to use models and some metrics to measure generated bounding boxes (mAP). So you can train a model for object detection in the simple fastai way with one of the two included learner classes:

`from fastai.vision.all import *`

`from fastai_object_detection.all import *`

`dls = ObjectDetectionDataLoaders.from_df(df, bs=2, item_tfms=[Resize(800)], batch_tfms=[Normalize.from_stats(*imagenet_stats)])`
`dls.show_batch()`

`learn = fasterrcnn_learner(dls, fasterrcnn_resnet50, metrics=[mAP_at_IoU40, mAP_at_IoU60, mAP_at_IoU90])`
`learn.lr_find()`

`learn.fit_one_cycle(1, 1e-04)`
`
All you need is a pandas DataFrame containing the data for each object in the image. 
In default settings follwing columns are required:
For the image, which contains the object:
* `image_id`
* `image_path`
The objects bounding box:
* `x_min`
* `y_min`
* `x_max`
* `y_max`
The objects class/label:
* `class_name`

If you want to use MaskRCNN for instance segementation, following columns are additionally needed:
* `mask_path`
* `mask_pixel_idx` (background is 0, different objects in the same mask have different pixel index)

There are helper functions available for example for adding the `image_path` by `image_id` or to change the bbox format from xywh to x1y1x2y2.
