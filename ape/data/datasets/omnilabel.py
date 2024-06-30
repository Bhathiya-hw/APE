import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
import cv2
"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_omnilabel_json", "register_omnilabel_instances"]


def register_omnilabel_instances(name, metadata, json_file, image_root, anno_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_omnilabel_json(json_file, image_root, anno_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="omnilabel", **metadata
    )


def load_omnilabel_json(json_file, image_root, anno_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    timer = Timer()

    with open(json_file, 'r') as f:
        data_json= json.load(f)
    print(data_json['descriptions'])
    if timer.seconds() > 1:
        logger.info("Loading omnilabels takes {:.2f} seconds.".format(timer.seconds()))

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        images = data_json['images']
        imgid2img= {i['id']: i['file_name'] for i in images}

        descriptions = data_json['descriptions']
        id2desc = {d['id']:d['text'] for d in descriptions}
        desc2id = {v:k for k,v in id2desc.items()}

        thing_classes = [v for k, v in id2desc.items()]
        meta.thing_classes = thing_classes

        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(id2desc.keys())}
        meta.thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id

        img2desc = {}
        for desc in descriptions:
            for img_id in desc['image_ids']:
                if img_id in img2desc:
                    img2desc[img_id].append(desc['text'])
                else:
                    img2desc[img_id] = [desc['text']]

        img2annt = {}
        annt_id = 0
        if 'annotations' in data_json:
            for a in data_json['annotations']:
                image_id =  a['image_id']
                for desc_id in a['description_ids']:
                    obj = {"image_id": image_id, "category_id": desc_id, "bbox": a['bbox'], "annt_id":annt_id}
                    if image_id in a:
                        img2annt[image_id].append(obj)
                        annt_id +=1
                    else:
                        img2annt[image_id] = obj
                        annt_id += 1
        dataset_dicts = []
        for i, d in img2desc.items():
            record = {}
            record["file_name"] = os.path.join(image_root, imgid2img[i])
            print(record["file_name"] )
            im = cv2.imread(record["file_name"])
            h, w, c = im.shape
            record["height"] = h
            record["width"] = w
            record["image_id"] = i
            record['expressions'] = d
            record['sent_ids'] = [thing_dataset_id_to_contiguous_id[desc2id[exp]] for exp in record['expressions']]
            print(record['sent_ids'])
            record['annotations'] = img2annt[i]
            dataset_dicts.append(record)
        return dataset_dicts

# def _get_omnilabel_metadata(categories):
#     if len(categories) == 0:
#         return {}
#     id_to_name = {x["id"]: x["name"] for x in categories}
#     thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
#     thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
#     return {
#         "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
#         "thing_classes": thing_classes,
#     }

_PREDEFINED_SPLITS_OMNILABEL = {
    "omnilabel_test_v0.1.3_all":{
        "omnilabel_test_v0.1.3_all": (
            "omnilabel/images",
            "dataset_all_test_v0.1.3.json",
            "omnilabel/omnilabel_test_v0.1.3"
        )
    },
    "omnilabel_val_v0.1.3_all":{
        "omnilabel_val_v0.1.3_all": (
            "omnilabel/images",
            "dataset_all_val_v0.1.3.json",
            "omnilabel/omnilabel_val_v0.1.3/",
        )
    },
    "omnilabel_val_v0.1.4_all": {
        "omnilabel_val_v0.1.3_all": (
            "omnilabel/images",
            "dataset_all_val_v0.1.4.json",
            "omnilabel",
        )
    },
    "omnilabel_val_v0.1.3_coco":{
        "omnilabel_val_v0.1.3_coco": (
            "omnilabel/images",
            "dataset_all_val_v0.1.3_coco.json",
            "omnilabel/omnilabel_val_v0.1.3/"
        ),
    },
    "omnilabel_val_v0.1.3_object365":{
        "omnilabel_val_v0.1.3_object365": (
            "omnilabel/images",
            "dataset_all_val_v0.1.3_object365.json",
            "omnilabel/omnilabel_val_v0.1.3/",
        )
    },
    "omnilabel_val_v0.1.3_openimagesv5":{
        "omnilabel_val_v0.1.3_openimagesv5": (
            "omnilabel/images",
            "dataset_all_val_v0.1.3_openimagesv5.json",
            "omnilabel/omnilabel_val_v0.1.3/"
        )
    }
}

def _get_omnilabel_metadata(json_file):
    with open(json_file, 'r') as f:
        data_json = json.load(f)

    descriptions = data_json['descriptions']
    id2desc = {d['id']: d['text'] for d in descriptions}

    thing_classes = [v for k, v in id2desc.items()]

    print(len(thing_classes))
    thing_dataset_id_to_contiguous_id = {k: i for i,k in enumerate(id2desc.keys())}
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


def register_all_omnilabel(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_OMNILABEL.items():
        print(splits_per_dataset.items())
        for key, (image_root, json_file, anno_root) in splits_per_dataset.items():
            register_omnilabel_instances(
                key,
                {},
                # _get_omnilabel_metadata(os.path.join(root, anno_root, json_file)),
                # os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, anno_root, json_file),
                os.path.join(root, image_root),
                os.path.join(root, anno_root),
            )


if __name__.endswith(".omnilabel"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_omnilabel(_root)
