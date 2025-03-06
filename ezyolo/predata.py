import os
import json
import requests
import base64
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import uuid

class PrepareYoloData:
    
    def _extract_unique_categories(self, df):
        """
        Extract unique category labels from the DataFrame.
        
        Args:
            df: DataFrame with a "annotations" column containing lists of label dictionaries
            
        Returns:
            Dictionary mapping category names to category IDs
        """
        unique_labels = set()
        for labels in df["annotations"]:
            if isinstance(labels, list):
                for label in labels:
                    if isinstance(label, dict) and "category" in label:
                        unique_labels.add(label["category"])
                    else: 
                        raise ValueError(f"Unrecognized label format: {label}, expecting dict with 'category' and 'bbox'")
            else:
                raise ValueError(f"Label is not a list of dicts: {labels}")
        
        return {cat: i+1 for i, cat in enumerate(sorted(unique_labels))}
    
    def process_df(self, df, output_dir="yolo_dataset", category_mapping=None):
        """
        Convert a DataFrame with image and label information to COCO format.
        
        Args:
            df: DataFrame with columns ["image", "annotations", "image_type", "label_type", "split"]
            output_dir: Directory to save the dataset
            annotations is a list of dicts, each dict contains "category" and "bbox"
            category_mapping: Optional dictionary mapping label names to category IDs
            
        Returns:
            Dictionary containing paths to the created dataset files
        """
        # Collect unique values in split
        unique_splits = df["split"].unique()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for split in unique_splits:
            os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        
        # Initialize COCO format dictionaries for train and val
        coco_data = {
            split: {"images": [], "annotations": [], "categories": []} for split in unique_splits
        }
        
        # Create category list if not provided
        if category_mapping is None:
            category_mapping = self._extract_unique_categories(df)
        
        # Populate categories in COCO format
        for category_name, category_id in category_mapping.items():
            category_info = {
                "id": category_id,
                "name": category_name,
                "supercategory": "none"
            }
            for split in unique_splits:
                coco_data[split]["categories"].append(category_info)
        
        # Process each row in the DataFrame
        annotation_id = 1
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            split = row["split"]
            if split not in unique_splits:
                continue
                
            # Get image
            image_data = row["image"]
            image_type = row["image_type"]
            
            # Load image based on type
            if image_type == "url":
                try:
                    response = requests.get(image_data, timeout=10)
                    img = Image.open(BytesIO(response.content))
                except Exception as e:
                    print(f"Error downloading image from URL {image_data}: {e}")
                    continue
            elif image_type == "image_path":
                try:
                    img = Image.open(image_data)
                except Exception as e:
                    print(f"Error opening image from path {image_data}: {e}")
                    continue
            elif image_type == "base64":
                try:
                    img_bytes = base64.b64decode(image_data)
                    img = Image.open(BytesIO(img_bytes))
                except Exception as e:
                    print(f"Error decoding base64 image: {e}")
                    continue
            else:
                print(f"Unsupported image type: {image_type}")
                continue
            
            # Generate a unique filename
            image_id = idx
            file_name = f"{image_id:012d}.jpg"
            img_path = os.path.join(output_dir, "images", split, file_name)
            
            # Save image
            img = img.convert("RGB")
            img.save(img_path)
            
            # Get image dimensions
            width, height = img.size
            
            # Add image info to COCO format
            image_info = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "date_captured": "",
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            }
            coco_data[split]["images"].append(image_info)
            
            # Process labels
            annotations = row["annotations"]
            if not isinstance(annotations, list):
                continue
                
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                    
                if "category" not in annotation or "bbox" not in annotation:
                    continue
                    
                category_name = annotation["category"]
                if category_name not in category_mapping:
                    continue
                    
                category_id = category_mapping[category_name]
                bbox = annotation["bbox"]
                
                # Convert bbox based on label_type
                label_type = row["label_type"]
                if label_type == "xyxy":
                    # [x1, y1, x2, y2] to COCO [x, y, width, height]
                    x1, y1, x2, y2 = bbox
                    x, y = x1, y1
                    w, h = x2 - x1, y2 - y1
                elif label_type == "xywh":
                    # [x, y, w, h] to COCO [x, y, width, height]
                    x, y, w, h = bbox
                else:
                    print(f"Unsupported label type: {label_type}")
                    continue
                
                # Create segmentation (simple polygon from bbox)
                segmentation = [[
                    x, y,
                    x + w, y,
                    x + w, y + h,
                    x, y + h
                ]]
                
                # Calculate area
                area = w * h
                
                # Add annotation to COCO format
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(area),
                    "segmentation": segmentation,
                    "iscrowd": 0
                }
                coco_data[split]["annotations"].append(annotation_info)
                annotation_id += 1
        
        # Save COCO format annotations
        train_json_path = os.path.join(output_dir, "annotations", "instances_train.json")
        val_json_path = os.path.join(output_dir, "annotations", "instances_val.json")
        
        with open(train_json_path, 'w') as f:
            json.dump(coco_data["train"], f)
            
        with open(val_json_path, 'w') as f:
            json.dump(coco_data["val"], f)
            
        return {
            "train_images": os.path.join(output_dir, "images", "train"),
            "val_images": os.path.join(output_dir, "images", "val"),
            "train_annotations": train_json_path,
            "val_annotations": val_json_path
        }

         
