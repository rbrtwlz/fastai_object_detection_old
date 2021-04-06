from fastai.vision.all import *

__all__ = ['ObjectDetectionDataLoaders']

class ObjectDetectionDataLoaders(DataLoaders):
    "Basic wrapper around `DataLoader`s with factory method for object dections problems"
 
    df = pd.DataFrame()
    img_id_col, img_path_col, class_col = "","","" 
    bbox_cols = []
    mask_path_col,mask_pixel_idx_col = "",""

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, valid_pct=0.2, img_id_col="image_id", img_path_col="image_path",
                bbox_cols=["x_min", "y_min", "x_max", "y_max"], class_col="class_name",
                mask_path_col="mask_path", mask_pixel_idx_col="mask_pixel_idx",
                seed=None, vocab=None, item_tfms=None, batch_tfms=None, **kwargs):
        
        if vocab is None :
                vocab = [c for c in df[class_col].unique()]

        cls.df = df
        cls.img_id_col,cls.img_path_col,cls.class_col = img_id_col,img_path_col,class_col
        cls.bbox_cols = bbox_cols
        cls.mask_path_col,cls.mask_pixel_idx_col = mask_path_col,mask_pixel_idx_col
        
        with_mask = mask_path_col in df.columns
        
        if item_tfms is None:
            item_tfms = [Resize(800, method="pad", pad_mode="zeros")]
            
        if not with_mask:
            dblock = DataBlock(
                blocks=(ImageBlock(cls=PILImage), BBoxBlock, BBoxLblBlock(vocab=vocab, add_na=True)),
                n_inp=1,
                splitter=RandomSplitter(valid_pct),
                get_items=cls._get_images,
                get_y=[cls._get_bboxes, cls._get_labels],
                item_tfms=item_tfms,
                batch_tfms=batch_tfms)
            res = cls.from_dblock(dblock, df, path=".", **kwargs)
            
        else:            
            dblock = DataBlock(
                blocks=(ImageBlock(cls=PILImage), MaskBlock, BBoxBlock, BBoxLblBlock(vocab=vocab, add_na=True)),
                n_inp=1,
                splitter=RandomSplitter(valid_pct),
                get_items=cls._get_images,
                get_y=[cls._get_mask, cls._get_bboxes, cls._get_labels],
                item_tfms=item_tfms,#method='pad', pad_mode='zeros'
                batch_tfms=batch_tfms)
            res = cls.from_dblock(dblock, df, path=".", before_batch=[cls._bb_pad_with_mask],**kwargs)
            
        return res
    
    def _get_images(df):
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        
        fns = L(fn for fn in df[img_path_col].unique())
        return fns

    def _get_bboxes(fn):
        df = ObjectDetectionDataLoaders.df
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        x_min_col, y_min_col, x_max_col, y_max_col = ObjectDetectionDataLoaders.bbox_cols
        
        filt = df[img_path_col] == fn #Path(fn)
        bboxes = [list(i) for i in zip(df.loc[filt,x_min_col], df.loc[filt,y_min_col], 
                                       df.loc[filt,x_max_col], df.loc[filt,y_max_col])]
        return bboxes

    def _get_labels(fn):
        df = ObjectDetectionDataLoaders.df
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        class_col = ObjectDetectionDataLoaders.class_col
        
        filt = df[img_path_col] == fn #Path(fn)
        labels = [l for l in df.loc[filt, class_col]]
        return labels
    
    def _get_mask(fn):
        df = ObjectDetectionDataLoaders.df
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        mask_path_col = ObjectDetectionDataLoaders.mask_path_col
        
        filt = df[img_path_col] == fn #Path(fn)
        mask_path = [m for m in df.loc[filt, mask_path_col]]
        #print(mask_path[0])
        return mask_path[0]
    
    def _bb_pad_with_mask(samples, pad_idx=0):
        samples = [(s[0],s[1], *clip_remove_empty(*s[2:])) for s in samples]
        max_len = max([len(s[3]) for s in samples])
        def _f(img,mask,bbox,lbl):
            bbox = torch.cat([bbox,bbox.new_zeros(max_len-bbox.shape[0], 4)])
            lbl  = torch.cat([lbl, lbl.new_zeros(max_len-lbl.shape[0])+pad_idx])
            return img,mask,bbox,lbl
        return [_f(*s) for s in samples]
    
"""    
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
        
"""
