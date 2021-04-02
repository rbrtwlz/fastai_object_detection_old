from fastai.basics import Path

__all__ = ['add_image_path_by_image_id', 'add_bbox_cols_by_xywh']


def add_image_path_by_image_id(df, img_id_col="image_id", path="./train", extension=".jpg"):
    """Adds column `image_path` to dataframe by combining the image's id, 
    the path, where the files are located and a file extension."""
    
    df["image_path"] = df[img_id_col].apply(lambda x: Path(path)/(str(x)+extension))
    return df


def add_bbox_cols_by_xywh(df, xywh_cols=["x", "y", "w", "h"]):
    """Adds columns `x_min`, `y_min`, `x_max` and `y_max` to dataframe by using bbox_cols in xywh format"""
    
    df["x_min"],df["y_min"] = df[xywh_cols[0]], df[xywh_cols[1]]
    df["x_max"] = df["x_min"] + df[xywh_cols[2]]
    df["y_max"] = df["y_min"] + df[xywh_cols[3]]
    return df
