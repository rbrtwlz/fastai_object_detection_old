# fastai_object_detection

Extension of the fastai library to include object recognition.

Install it with

`pip install --upgrade git+https://github.com/rbrtwlz/fastai_object_detection`

This package makes FasterRCNN, MaskRCNN and EfficientDet available for [fastai](https://www.fast.ai/) users by using a callback which converts the batches to the required input for FasterRCNN or MaskRCNN. It comes with a fastai `Dataloaders` class for object detection, prepared and easy to use models and some metrics to measure generated bounding boxes (mAP). So you can train a model for object detection in the simple fastai way with one of two included learner classes:

```python
from fastai.vision.all import *
from fastai_object_detection.all import *

path, df = CocoData.create(ds_name="coco-cats-and-dogs", cat_list=["cat", "dog"], max_images=2000)

dls = ObjectDetectionDataLoaders.from_df(df, bs=2, 
                                         item_tfms=[Resize(800, method="pad", pad_mode="zeros")], 
                                         batch_tfms=[Normalize.from_stats(*imagenet_stats)])
dls.show_batch()

learn = fasterrcnn_learner(dls, fasterrcnn_resnet50, metrics=[mAP_at_IoU40, mAP_at_IoU60])
learn.lr_find()

learn.fit_one_cycle(1, 1e-04)
```

All you need is a pandas `DataFrame` containing the data for each object in the images. 
In default setting follwing columns are required:

For the image, which contains the object(s):
* `image_id`
* `image_path`

The object's bounding box:
* `x_min`
* `y_min`
* `x_max`
* `y_max`

The object's class/label:
* `class_name`

If you want to use MaskRCNN for instance segementation, following columns are additionally required:
* `mask_path` (path to the binary mask, which represents the object in the image)

There are helper functions available, for example for adding the `image_path` by `image_id` or to change the bbox format from `xywh` to `x1y1x2y2`.

Also there is a `CocoData` class provided to help you to download images from [COCO dataset](https://cocodataset.org/), create the corresponding masks and generate a `DataFrame`.
Simply use the following line to create a dataset for cat and dog detection:

```python
path, df = CocoData.create(ds_name="coco-cats-and-dogs", cat_list=["cat", "dog"], 
                           max_images=2000, with_mask=True)
```
By default, when no `data_path` is specified, it creates a new dataset in fastai's data path (like `untar_data()`)

You can always list all available dataset with 
```python
CocoData.ls()
```
get the path and the `DataFrame` of a dataset

```python
path,df = CocoData.get_path_df(ds_name="coco-cats-and-dogs")
```
remove a dataset

```python
CocoData.remove(ds_name="coco-cats-and-dogs")
```
or show some examples

```python
CocoData.show_examples("coco-cats-and-dogs", n=5)
```


