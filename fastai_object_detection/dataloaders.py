from fastai.vision.all import *
#from fastai.torch_core import merge

# temp bug fix
# https://github.com/fastai/fastai/issues/3384
TensorMultiCategory.register_func(Tensor.__getitem__, TensorMultiCategory, TensorBBox)

__all__ = ['ObjectDetectionDataLoaders']


class TensorBinMasks(TensorImageBase):
    "Tensor class for binary mask representation"
    def show(self, ctx=None, **kwargs):
        return show_binmask(self,ctx=ctx, **{**self._show_args, **kwargs})

@delegates(plt.Axes.imshow, keep=True, but=['shape', 'imlim'])
def show_binmask(im, ax=None, figsize=None, title=None, ctx=None, **kwargs):
    "Function to show binary masks with matplotlib"
    if hasattrs(im, ('data','cpu','permute')):
        im = im.data.cpu()
    if not isinstance(im,np.ndarray): im=array(im)
    ax = ifnone(ax,ctx)
    if figsize is None: figsize = (_fig_bounds(im.shape[0]), _fig_bounds(im.shape[1]))
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    for m in im:
        c = (np.random.random(3) * 0.6 + 0.4) 
        #draw_mask(ax, m, c)
        color_mask = np.ones((*m.shape, 3)) * c
        ax.imshow(np.dstack((color_mask, m * 0.5)))
        ax.contour(m, colors=[color_mask[0, 0, :]], alpha=0.4)
    if title is not None: ax.set_title(title)
    ax.axis('off')
    return ax

def _fig_bounds(x):
    r = x//32
    return min(5, max(1,r))


def BinaryMasksBlock():
    "A `TransformBlock` for binary masks"
    return TransformBlock(type_tfms=lambda x: tuple(apply(PILMask.create,x)), batch_tfms=IntToFloatTensor)

def _bin_mask_stack_and_padding(t, pad_idx=0):
    "Function for padding to create batches when number of objects is different"
    stacked_masks = [torch.stack(t[i][1], dim=0) for i in range(len(t))]
    imgs = [t[i][0] for i in range(len(t))]
    bboxes = [t[i][2] for i in range(len(t))]
    labels = [t[i][3] for i in range(len(t))]
    samples = L(t for t in zip(imgs,stacked_masks,bboxes,labels))
    samples = [(s[0], *_clip_remove_empty_with_mask(*s[1:])) for s in samples]
    max_len = max([len(s[3]) for s in samples])
    def _f(img,bin_mask,bbox,lbl):
        bin_mask = torch.cat([bin_mask,bin_mask.new_zeros(max_len-bin_mask.shape[0], bin_mask.shape[-2], bin_mask.shape[-1])])
        bbox = torch.cat([bbox,bbox.new_zeros(max_len-bbox.shape[0], 4)])
        lbl  = torch.cat([lbl,lbl.new_zeros(max_len-lbl.shape[0])+pad_idx])
        return img,TensorBinMasks(bin_mask),bbox,lbl
    return [_f(*s) for s in samples]
    
def _clip_remove_empty_with_mask(bin_mask, bbox, label):
    bbox = torch.clamp(bbox, -1, 1)
    empty = ((bbox[...,2] - bbox[...,0])*(bbox[...,3] - bbox[...,1]) <= 0.)
    return (bin_mask[~empty], bbox[~empty], label[~empty])


class TensorBinMasks2TensorMask(Transform):
    "Class to transform binary masks to fastai's `TensorMask` class to make fastai's transforms available"
    def encodes(self, x:TensorBinMasks):
        return TensorMask(x)
    def decodes(self, x:TensorMask):
        return TensorBinMasks(x)

    
class ObjectDetectionDataLoaders(DataLoaders):
    "Basic wrapper around `DataLoader`s with factory method for object dections problems"
    df = pd.DataFrame()
    img_id_col, img_path_col, class_col = "","","" 
    bbox_cols = []
    mask_path_col,object_id_col = "",""

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, valid_pct=0.2, img_id_col="image_id", img_path_col="image_path",
                bbox_cols=["x_min", "y_min", "x_max", "y_max"], class_col="class_name",
                mask_path_col="mask_path", object_id_col="object_id",
                seed=None, vocab=None, add_na=True, item_tfms=None, batch_tfms=None, debug=False, **kwargs):
        
        if vocab is None :
                vocab = [c for c in df[class_col].unique()]

        cls.df = df
        cls.img_id_col,cls.img_path_col,cls.class_col = img_id_col,img_path_col,class_col
        cls.bbox_cols = bbox_cols
        cls.mask_path_col,cls.object_id_col = mask_path_col,object_id_col
        
        with_mask = mask_path_col in df.columns
        
        #if item_tfms is None: item_tfms = [Resize(800, method="pad", pad_mode="zeros")]
            
        if not with_mask:
            dblock = DataBlock(
                blocks=(ImageBlock(cls=PILImage), BBoxBlock, BBoxLblBlock(vocab=vocab, add_na=add_na)),
                n_inp=1,
                splitter=RandomSplitter(valid_pct),
                get_items=cls._get_images,
                get_y=[cls._get_bboxes, cls._get_labels],
                item_tfms=item_tfms,
                batch_tfms=batch_tfms)
            if debug: print(dblock.summary(df))
            res = cls.from_dblock(dblock, df, path=".", before_batch=[bb_pad], **kwargs)
            
        else:            
            dblock = DataBlock(
                blocks=(ImageBlock(cls=PILImage), BinaryMasksBlock, 
                        BBoxBlock, BBoxLblBlock(vocab=vocab, add_na=add_na)),
                n_inp=1,
                splitter=RandomSplitter(valid_pct),
                get_items=cls._get_images,
                get_y=[cls._get_masks, cls._get_bboxes, cls._get_labels],
                item_tfms=item_tfms,
                batch_tfms=[TensorBinMasks2TensorMask(), *batch_tfms])
            if debug: print(dblock.summary(df))
            res = cls.from_dblock(dblock, df, path=".", before_batch=[_bin_mask_stack_and_padding],**kwargs)
            
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
    
    def _get_masks(fn):
        df = ObjectDetectionDataLoaders.df
        img_path_col = ObjectDetectionDataLoaders.img_path_col
        mask_path_col = ObjectDetectionDataLoaders.mask_path_col
        
        filt = df[img_path_col] == fn
        mask_paths = [m for m in df.loc[filt, mask_path_col]]
        return mask_paths

