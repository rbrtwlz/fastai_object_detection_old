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
import numpy as np
from shutil import rmtree
from matplotlib import cm
from matplotlib.colors import ListedColormap, Colormap


__all__ = ['CocoData']

class CocoData():

    coco = None 

    @classmethod
    def create(cls, ds_name, cat_list, data_path=None, with_mask=False, max_images=1000, remove_crowded=True):

        path = Path(URLs.path(c_key='data'))/ds_name if data_path is None else Path(data_path)/ds_name
        path_images = path/"images"
        path_masks = path/"masks"

        if Path(path).is_dir():
            print("Dataset "+str(ds_name)+" already exists: "+str(path))
            return cls.get_path_df(ds_name, data_path=data_path)

        # create folders
        print("Creating folders.")
        path.mkdir(exist_ok=False, parents=True)
        path_images.mkdir()
        if with_mask: path_masks.mkdir()

        # download annotation files
        annotations = 'annotations/instances_train2017.json'
        if not (path/annotations).is_file():
            cls._download_annotation_file(path)
        if not (path/annotations).is_file():
            print("Download was not successful. No annotation file found.")
            return
        cls.coco = COCO(annotation_file=str(path/annotations))

        # download images
        cls._download_images(cat_list, path_images, max_images, remove_crowded)

        # create dataframe
        df = cls._create_dataframe(path, cat_list, with_mask)

        return path, df


    def get_path_df(ds_name, data_path=None):
        path = Path(URLs.path(c_key='data'))/ds_name if data_path is None else Path(data_path)/ds_name
        if path.is_dir():
            if (path/"df_train.csv").is_file():
                return (path, pd.read_csv(path/"df_train.csv"))
            else:
                print("No Dataframe found in "+str(path))
        else:
            print("No dataset '"+str(path)+"' found.")
            print("Create dataset first with CocoData.create(ds_name, cat_list) or list available datasets with CocoData.ls()")


    def ls(data_path=None):
        path = Path(URLs.path(c_key='data')) if data_path is None else Path(data_path)
        if path.is_dir():
            return list(path.ls())
        else: print("Path "+str(path)+" does not exist.")


    def remove(ds_name, data_path=None):
        path = Path(URLs.path(c_key='data'))/ds_name if data_path is None else Path(data_path)/ds_name
        if path.is_dir():
            rmtree(path)
            print(str(path)+" removed.")
        else:
            print("No dataset '"+str(path)+"' found.")


    def show_examples(ds_name, data_path=None, n=3):
        _, df = CocoData.get_path_df(ds_name, data_path=data_path) 
        img_ids = [i for i in df.image_id.unique()]
        shuffle(img_ids)
        with_mask = "mask_path" in df.columns
        from matplotlib import cm
        # transparent, blue, red, yellow, green, orange, black  
        # if more than 6 objects, rest is black
        colors_cmap = ["#ffffff99", "#0000ffcc", "#ff0000cc","#ffff00cc", "#4bdd75cc", "#bd6914cc", "#000000cc"] 
        cmap1 = ListedColormap(colors_cmap)
        for img_id in img_ids[:n]:
            filt = df.image_id == img_id
            img_path = df.loc[filt,"image_path"].values[0]
            img = PILImage.create(img_path)
            bboxes = [box for box in df.loc[filt,["x_min","y_min","x_max","y_max"]].values]
            labels = [label[0] for label in df.loc[filt,["class_name"]].values]
            if with_mask:
                mask_path = df.loc[filt,"mask_path"].values[0]
                mask = PILMask.create(mask_path)
            fig,ax = plt.subplots(figsize=(8,8))
            TensorImage(img).show(ax=ax)
            if with_mask:
                TensorMask(mask).show(ax, cmap=cmap1, vmin=0, vmax=6)
            LabeledBBox(TensorBBox(bboxes), labels).show(ax)


    def _download_annotation_file(path):
        print("Downloading annotation files...")
        url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        zipresp = urlopen(url)
        zip_fn = path/'annotations_trainval2017.zip'
        with open(zip_fn, 'wb') as zip:
            zip.write(zipresp.read())
        zf = ZipFile(zip_fn)
        zf.extractall(path=str(path))
        zf.close()
        Path(zip_fn).unlink()


    def _download_images(cat_list, path_images, max_images, remove_crowded):
        cat_ids = CocoData.coco.getCatIds(catNms=cat_list);
        idx2cat = {e['id']:e['name'] for e in CocoData.coco.loadCats(CocoData.coco.getCatIds())}
        img_id2fn = {}
        print("Found "+str(len(cat_ids))+" valid categories.")
        print([idx2cat[e] for e in cat_ids])
        print("Starting download.")
        mb = master_bar(range(len(cat_ids)))
        for i in mb: 
            c_id = cat_ids[i]
            print("Downloading images of category "+idx2cat[c_id])
            img_ids = CocoData.coco.getImgIds(catIds=c_id)
            # small function to filter images with crowded objects
            def _f(iid):
                annos = CocoData.coco.loadAnns(CocoData.coco.getAnnIds(imgIds=iid))
                annos = [a for a in annos if idx2cat[a["category_id"]] in cat_list]
                is_crowd = [a["iscrowd"] for a in annos]
                return 1 in is_crowd
            if remove_crowded:
                img_ids = [i for i in img_ids if not _f(i)]
            if max_images is not None:
                img_ids = img_ids[:max_images]
            for i in img_ids: 
                img_id2fn[i] = path_images/(str(i).zfill(12)+".jpg")
            for i in progress_bar(range(len(img_ids)), parent=mb):
                with contextlib.redirect_stdout(io.StringIO()):
                    CocoData.coco.download(path_images, [img_ids[i]])

        print(len([fn for fn in path_images.ls()]), "images downloaded.")


    def _create_dataframe(path, cat_list, with_mask,):
        print("Creating Dataframe...")
        path_images = path/"images"
        path_masks = path/"masks"
        df_train = pd.DataFrame()

        img_id2fn = {int(Path(fn).stem):fn for fn in path_images.ls()}
        img_ids = [i for i in img_id2fn.keys()]
        idx2cat = {e['id']:e['name'] for e in CocoData.coco.loadCats(CocoData.coco.getCatIds())}

        for i in progress_bar(range(len(img_ids))):
            img_id = img_ids[i]
            annos = CocoData.coco.loadAnns(CocoData.coco.getAnnIds(imgIds=img_id))
            # remove annotations of other labels 
            annos = [a for a in annos if idx2cat[a["category_id"]] in cat_list]
            # sort by area
            area_dict = {a["area"]:a for a in annos}
            annos = [area_dict[k] for k in sorted(area_dict, reverse=True)]

            n_objs = len(annos)

            df_x_mins = [a["bbox"][0] for a in annos]
            df_y_mins = [a["bbox"][1] for a in annos]
            widths = [a["bbox"][2] for a in annos]
            heights = [a["bbox"][3] for a in annos]
            df_x_maxs = [df_x_mins[ia]+widths[ia] for ia in range(len(annos))]
            df_y_maxs = [df_y_mins[ia]+heights[ia] for ia in range(len(annos))]
            df_class_names = [idx2cat[a["category_id"]] for a in annos]

            df_img_id = [img_id] * n_objs
            img_path = img_id2fn[img_id]
            df_img_path = [str(img_path)] * n_objs

            if with_mask:
                df_mask_path = [] 
                df_obj_ids = [i for i in range(n_objs)]
                mask = np.zeros(CocoData.coco.annToMask(annos[0]).shape, dtype=np.uint8)
                for o_id in df_obj_ids:
                    mask = CocoData.coco.annToMask(annos[o_id]) #* p_idx
                    #mask[mask>p_idx] = p_idx # for overlapping parts
                    mask_path = path_masks/(img_path.stem+"_"+str(o_id)+".png") # save mask always as png
                    Image.fromarray(mask).save(mask_path)
                    df_mask_path.append(str(mask_path))

                df = pd.DataFrame({"image_id":df_img_id, "image_path":df_img_path, 
                                   "mask_path":df_mask_path, "object_id":df_obj_ids, 
                                   "x_min":df_x_mins, "y_min":df_y_mins, "x_max":df_x_maxs, "y_max":df_y_maxs,
                                   "class_name":df_class_names})
            else:
                df = pd.DataFrame({"image_id":df_img_id, "image_path":df_img_path, 
                                   "x_min":df_x_mins, "y_min":df_y_mins, "x_max":df_x_maxs, "y_max":df_y_maxs,
                                   "class_name":df_class_names})   
                    
            df_train = df_train.append(df)

        df_train.reset_index(inplace=True, drop=True)
        df_train.to_csv(str(path/"df_train.csv"), index=False)
        return df_train
