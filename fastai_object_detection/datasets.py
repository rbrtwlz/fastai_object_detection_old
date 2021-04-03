import contextlib
import io
from urllib.request import urlopen
from random import shuffle
from zipfile import ZipFile
from fastai.data.external import URLs
from fastai.imports import Path
from fastai.vision.all import PILImage, PILMask, TensorImage, TensorMask, LabeledBBox, TensorBBox
from fastprogress.fastprogress import master_bar, progress_bar
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

__all__ = ['CocoDatasetManager']

class CocoDatasetManager():
    def __init__(self, ds_name, cat_list=None, ds_path='./train', with_mask=False, max_images=1000):
        self.cat_list = cat_list
        self.with_mask = with_mask
        self.max_images = max_images

        if ds_path is None:
            self.path = Path(URLs.path(c_key='data'))/ds_name
        else: 
            self.path = Path(ds_path)/ds_name
        self.path_images = self.path/"images"
        self.path_masks = self.path/"masks"

        self.idx2cat,self.img_id2fn = {},{}
        self.df_train = pd.DataFrame()


    def get_dataset_path(self):
        if Path(self.path).is_dir():
            return self.path

        print("No dataset found in " + str(self.path))

        if self.cat_list is None:
            print("Specify categories to download.")
            return

        # create folders
        print("Creating folders.")
        self.path.mkdir(exist_ok=False, parents=True)
        self.path_images.mkdir()
        if self.with_mask: self.path_masks.mkdir()
        
        # download annotation files
        annotations = 'annotations/instances_train2017.json'
        if not (self.path/annotations).is_file():
            self._download_annotation_file()
        if not (self.path/annotations).is_file():
            print("Download was not successful. No annotation file found.")
            return
        self.coco = COCO(annotation_file=str(self.path/annotations))

        # download images
        self._download_images()

        # create dataframe
        self._create_dataframe()

        return self.path


    def get_dataframe(self):
        if (self.path/"df_train.csv").is_file():
            return pd.read_csv(self.path/"df_train.csv")
        else:
            print("No Dataframe found in "+self.path)
            return


    def _download_annotation_file(self):
        print("Downloading annotation files...")
        url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        zipresp = urlopen(url)
        zip_fn = self.path/'annotations_trainval2017.zip'
        with open(zip_fn, 'wb') as zip:
            zip.write(zipresp.read())
        zf = ZipFile(zip_fn)
        zf.extractall(path=str(self.path))
        zf.close()
        Path(zip_fn).unlink()


    def _download_images(self):
        cat_ids = self.coco.getCatIds(catNms=self.cat_list);
        self.idx2cat = {e['id']:e['name'] for e in self.coco.loadCats(self.coco.getCatIds())}
        self.img_id2fn = {}
        print("Found "+str(len(cat_ids))+" valid categories.")
        print([self.idx2cat[e] for e in cat_ids])
        print("Starting download.")
        mb = master_bar(range(len(cat_ids)))
        for i in mb: 
            c_id = cat_ids[i]
            print("Downloading images of category "+self.idx2cat[c_id])
            if self.max_images is None:
                img_ids = self.coco.getImgIds(catIds=c_id)
            else:
                img_ids = self.coco.getImgIds(catIds=c_id)[0:self.max_images+1] # downloads one less...
            for i in img_ids: 
                self.img_id2fn[i] = self.path_images/(str(i).zfill(12)+".jpg")
            for i in progress_bar(range(len(img_ids)), parent=mb):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.coco.download(self.path_images, [img_ids[i]])
        print(len([fn for fn in self.path_images.ls()]), "images downloaded.")


    def _create_dataframe(self):
        print("Creating Dataframe...")
        self.df_train = pd.DataFrame()
        img_ids = [i for i in self.img_id2fn.keys()]

        for i in progress_bar(range(len(img_ids))):
            img_id = img_ids[i]
            annos = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            # remove annotations of other labels 
            annos = [a for a in annos if self.idx2cat[a["category_id"]] in self.cat_list]

            n_objs = len(annos)

            df_x_mins = [a["bbox"][0] for a in annos]
            df_y_mins = [a["bbox"][1] for a in annos]
            widths = [a["bbox"][2] for a in annos]
            heights = [a["bbox"][3] for a in annos]
            df_x_maxs = [df_x_mins[ia]+widths[ia] for ia in range(len(annos))]
            df_y_maxs = [df_y_mins[ia]+heights[ia] for ia in range(len(annos))]
            df_class_names = [self.idx2cat[a["category_id"]] for a in annos]

            df_img_id = [img_id] * n_objs
            img_path = self.img_id2fn[img_id]
            df_img_path = [img_path] * n_objs

            if self.with_mask:
                mask_path = self.path_masks/(img_path.stem+".png") # save mask always as png
                df_mask_path = [mask_path] * n_objs
                df_mask_pixel_idx = [i for i in range(1,n_objs+1)]
                mask = np.zeros(self.coco.annToMask(annos[0]).shape, dtype=np.uint8)
                for j,p_idx in enumerate(df_mask_pixel_idx):
                    mask += self.coco.annToMask(annos[j]) * p_idx
                    Image.fromarray(mask).save(mask_path)

                df = pd.DataFrame({"image_id":df_img_id, "image_path":df_img_path, 
                                   "mask_path":df_mask_path, "mask_pixel_idx":df_mask_pixel_idx, 
                                   "x_min":df_x_mins, "y_min":df_y_mins, "x_max":df_x_maxs, "y_max":df_y_maxs,
                                   "class_name":df_class_names})
            else:
                df = pd.DataFrame({"image_id":df_img_id, "image_path":df_img_path, 
                                   "x_min":df_x_mins, "y_min":df_y_mins, "x_max":df_x_maxs, "y_max":df_y_maxs,
                                   "class_name":df_class_names})   
                    
            self.df_train = self.df_train.append(df)

        self.df_train.reset_index(inplace=True, drop=True)
        self.df_train.to_csv(str(self.path/"df_train.csv"), index=False)


    def show_examples(self, n=3):
        df = self.get_dataframe()
        img_ids = [i for i in df.image_id.unique()]
        shuffle(img_ids)
        for img_id in img_ids[:n]:
            filt = df.image_id == img_id

            img_path = df.loc[filt,"image_path"].values[0]
            img = PILImage.create(img_path)

            bboxes = [box for box in df.loc[filt,["x_min","y_min","x_max","y_max"]].values]
            labels = [label[0] for label in df.loc[filt,["class_name"]].values]

            if self.with_mask:
                mask_path = df.loc[filt,"mask_path"].values[0]
                mask = PILMask.create(mask_path)

            fig,ax = plt.subplots(figsize=(8,8))
            TensorImage(img).show(ax=ax)
            if self.with_mask:
                TensorMask(mask).show(ax)
            LabeledBBox(TensorBBox(bboxes), labels).show(ax)
