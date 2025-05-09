import sys
import os
from pathlib import Path
import torch
import numpy as np

from yolo import (
    AugmentationComposer,
    NMSConfig,
    PostProcess,
    create_converter,
    create_model,
    draw_bboxes,
)
from omegaconf import OmegaConf

def get_current_package_path():
    """
    Returns the absolute path to the current package directory.
    
    Returns:
        Path: The absolute path to the current package directory.
    """
    return Path(__file__).parent.absolute()


def load_dataset_config(dataset_name="coco"):
    """
    Loads the dataset configuration from the specified dataset YAML file.
    
    Args:
        dataset_name (str): Name of the dataset configuration file (without .yaml extension).
            Defaults to "coco".
    
    Returns:
        dict: The loaded dataset configuration.
    """
    config_path = get_current_package_path() / "config" / "dataset" / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")
    
    return OmegaConf.load(config_path)

def load_model_config(model_name):
    """
    Loads the model configuration from the specified model YAML file.
    
    Args:
        model_name (str): Name of the model configuration file (without .yaml extension).
    
    Returns:
        dict: The loaded model configuration.
    """
    config_path = get_current_package_path() / "config" / "model" / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def load_image(image_source):
    """
    Loads an image from various sources: URL, local file path, or base64 string.
    
    Args:
        image_source (str): Can be a URL, local file path, or base64 encoded string.
    
    Returns:
        PIL.Image.Image: The loaded image in RGB format.
    """
    import numpy as np
    import base64
    import requests
    from urllib.parse import urlparse
    from PIL import Image
    from io import BytesIO
    
    # Check if image_source is a URL
    try:
        parsed_url = urlparse(image_source)
        if parsed_url.scheme in ['http', 'https']:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image.convert('RGB')
    except (requests.exceptions.RequestException, ValueError):
        pass
    
    # Check if image_source is a base64 string
    if image_source.startswith(('data:image', 'base64:')):
        try:
            # Extract the base64 part if it's a data URL
            if image_source.startswith('data:image'):
                image_source = image_source.split(',')[1]
            elif image_source.startswith('base64:'):
                image_source = image_source[7:]
                
            image_data = base64.b64decode(image_source)
            image = Image.open(BytesIO(image_data))
            return image.convert('RGB')
        except Exception:
            pass
    
    # Try to load as a local file path
    try:
        image = Image.open(image_source)
        return image.convert('RGB')
    except Exception:
        pass
    
    raise ValueError(f"Could not load image from source: {image_source}")



class CocoDetector:

    def __init__(self):
        self.model = None
        self.converter = None
        self.transform = None

    def load_model(self, model_name, image_size=(640, 640), device=torch.device("cpu")):
        dataset_cfg = load_dataset_config()
        model_cfg = load_model_config(model_name)
        model_cfg.model.auxiliary = {}
        model = create_model(model_cfg, weight_path=True, class_num=80)
        converter = create_converter(model_cfg.name, model, model_cfg.anchor, image_size, device)
        model = model.to(device).eval()
        transform = AugmentationComposer([])

        self.model = model
        self.converter = converter
        self.transform = transform
        self.device = device
        self.class_list = dataset_cfg.class_list

    def predict_draw_one_image(self, image, nms_confidence, nms_iou, max_bbox):
        pred_bbox = self.predict_and_return_bboxes(image, nms_confidence, nms_iou, max_bbox)

        result_image = draw_bboxes(image, pred_bbox, idx2label=self.class_list)
        return result_image

    def predict_and_return_bboxes(self, image, nms_confidence, nms_iou, max_bbox):
        image_tensor, _, rev_tensor = self.transform(image)

        image_tensor = image_tensor.to(self.device)[None]
        rev_tensor = rev_tensor.to(self.device)[None]

        nms_config = NMSConfig(nms_confidence, nms_iou, max_bbox)
        post_proccess = PostProcess(self.converter, nms_config)

        with torch.no_grad():
            predict = self.model(image_tensor)
            pred_bbox = post_proccess(predict, rev_tensor)
        
        # pred_bbox is a list of lists
        # each item is a list of float. 
        # class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        return pred_bbox