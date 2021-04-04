from fastai.vision.all import *

__all__ = ['ObjectDetectionDataLoaders']


class ObjectDetectionDataLoaders(DataLoaders):
    
    df = pd.DataFrame()
    img_id_col, img_path_col, class_col = "","","" 
    bbox_cols = []

    "Basic wrapper around `DataLoader`s with factory methods for object dections problems"
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, valid_pct=0.2, img_id_col="image_id", img_path_col="image_path",
                bbox_cols=["x_min", "y_min", "x_max", "y_max"], class_col="class_name",
                seed=None, vocab=None, item_tfms=None, batch_tfms=None, **kwargs):
        
        if vocab is None :
                vocab = [c for c in df[class_col].unique()]

        cls.df = df
        cls.img_id_col,cls.img_path_col,cls.class_col = img_id_col,img_path_col,class_col
        cls.bbox_cols = bbox_cols
        
        if item_tfms is None:
            item_tfms = [Resize(800)]

        dblock = DataBlock(
            blocks=(ImageBlock(cls=PILImage), BBoxBlock, BBoxLblBlock(vocab=vocab, add_na=True)),
            n_inp=1,
            splitter=RandomSplitter(valid_pct),
            get_items=cls._get_images,
            get_y=[cls._get_bboxes, cls._get_labels],
            item_tfms=item_tfms,
            batch_tfms=batch_tfms)
        
        res = cls.from_dblock(dblock, df, path=".", **kwargs)
        return res
    
    def _get_images(df):
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        
        fns = L(fn for fn in df[img_path_col].unique())
        return fns

    def _get_bboxes(fn):
        df = ObjectDetectionDataLoaders.df
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        x_min, y_min, x_max, y_max = ObjectDetectionDataLoaders.bbox_cols
        
        filt = df[img_path_col] == fn #Path(fn)
        bboxes = [list(i) for i in zip(df.loc[filt,x_min], df.loc[filt,y_min], 
                                       df.loc[filt,x_max], df.loc[filt,y_max])]
        return bboxes

    def _get_labels(fn):
        df = ObjectDetectionDataLoaders.df
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        class_col = ObjectDetectionDataLoaders.class_col
        
        filt = df[img_path_col] == fn #Path(fn)
        labels = [l for l in df.loc[filt, class_col]]
        return labels
