import os
from typing import Optional

from timm.data import ImageDataset

from .class_repeat_dataset import ClassRepeatDataset

_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)


def _search_split(root: str, split: str) -> str:
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root

def create_train_dataset(
        name: str,
        root: str,
        split: str = 'validation',
        search_split: bool = True,
        class_map: dict = None,
        load_bytes: bool = False,
        input_img_mode: str = 'RGB',
        repeat_class: Optional[int] = None,
        repeat_factor: Optional[int] = None,
        **kwargs,
) -> ImageDataset:
    if search_split and os.path.isdir(root):
            # look for split specific sub-folder in root
            root = _search_split(root, split)
    if repeat_class is not None and repeat_factor is not None and repeat_factor > 1:
        ds = ClassRepeatDataset(
            root=root,
            reader=name,
            class_map=class_map,
            load_bytes=load_bytes,
            input_img_mode=input_img_mode,
            class_to_repeat=[repeat_class],
            repeat_factor=repeat_factor,
            **kwargs,
        )
    else:
        ds = ImageDataset(
            root,
            reader=name,
            class_map=class_map,
            load_bytes=load_bytes,
            input_img_mode=input_img_mode,
            **kwargs,
        )   
    return ds