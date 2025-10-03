"""
PLT_NET_Mini dataset for DASSL framework
"""
import os
import pickle
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
import pandas as pd
import openml


@DATASET_REGISTRY.register()
class PLTNetMini(DatasetBase):
    """PLT_NET_Mini dataset from OpenML
    
    Reference:
        OpenML dataset ID: 44293
        Plant classification dataset with 25 categories
    """
    
    dataset_dir = "plt_net_mini"
    
    def __init__(self, cfg):
        self.dataset_dir = cfg.DATASET.ROOT
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_PLTNetMini.json")
        
        # Download dataset if not exists
        if not os.path.exists(self.dataset_dir):
            self._download_dataset()
        
        # Load data
        train_x, val, test = self._read_data()
        
        super().__init__(train_x=train_x, val=val, test=test)
    
    def _download_dataset(self):
        """Download PLT_NET_Mini dataset from OpenML"""
        print("Downloading PLT_NET_Mini dataset from OpenML...")
        
        mkdir_if_missing(self.dataset_dir)
        
        # Download dataset
        dataset = openml.datasets.get_dataset(44293, download_data=True, download_all_files=True)
        
        # Get cache directory
        cache_dir = openml.config.get_cache_directory()
        source_images_dir = os.path.join(cache_dir, "datasets", "44293", "PLT_NET_Mini", "images")
        source_labels_path = os.path.join(cache_dir, "datasets", "44293", "PLT_NET_Mini", "labels.csv")
        
        # Copy images to dataset directory
        import shutil
        if os.path.exists(source_images_dir):
            if os.path.exists(self.image_dir):
                shutil.rmtree(self.image_dir)
            shutil.copytree(source_images_dir, self.image_dir)
            
        # Copy labels file
        if os.path.exists(source_labels_path):
            shutil.copy2(source_labels_path, os.path.join(self.dataset_dir, "labels.csv"))
        
        print(f"Dataset downloaded to {self.dataset_dir}")
    
    def _read_data(self):
        """Read and split dataset"""
        # Use hardcoded paths
        labels_csv_path = r"E:\Others\DATASETS\plant_net_mini\labels.csv"
        images_dir = r"E:\Others\DATASETS\plant_net_mini\images"
        
        if not os.path.exists(labels_csv_path):
            raise FileNotFoundError(f"Labels file not found at {labels_csv_path}")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found at {images_dir}")

        # Read labels CSV
        df = pd.read_csv(labels_csv_path)
        print(f"Loaded {len(df)} samples from labels.csv")

        # Get unique classes and create class mapping
        classes = sorted(df['CATEGORY'].unique().tolist())
        self._class_names = classes
        print(f"Found {len(classes)} classes: {classes}")
        
        # Group images by class
        class_to_images = defaultdict(list)
        for _, row in df.iterrows():
            image_path = os.path.join(images_dir, row['FILE_NAME'])
            if os.path.exists(image_path):
                class_to_images[row['CATEGORY']].append(image_path)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        # Create splits following DASSL protocol
        train_x, val, test = [], [], []
        
        for class_name in classes:
            images = class_to_images[class_name]
            if len(images) == 0:
                print(f"Warning: No images found for class {class_name}")
                continue
                
            print(f"Class {class_name}: {len(images)} images")
            
            # Shuffle images
            random.shuffle(images)
            
            # Split: 70% train, 15% val, 15% test
            n_total = len(images)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Create Datum objects
            for img_path in train_images:
                item = Datum(
                    impath=img_path,
                    label=classes.index(class_name),
                    classname=class_name
                )
                train_x.append(item)
            
            for img_path in val_images:
                item = Datum(
                    impath=img_path,
                    label=classes.index(class_name),
                    classname=class_name
                )
                val.append(item)
            
            for img_path in test_images:
                item = Datum(
                    impath=img_path,
                    label=classes.index(class_name),
                    classname=class_name
                )
                test.append(item)
        
        print(f"Dataset split: {len(train_x)} train, {len(val)} val, {len(test)} test")
        return train_x, val, test
    
    @property
    def classnames(self):
        return self._class_names


def get_plt_net_mini_classes():
    """Get all class names from PLT_NET_Mini dataset"""
    return [
        "Cirsium vulgare", "Fragaria vesca", "Cirsium arvense", "Aegopodium podagraria",
        "Daucus carota", "Alliaria petiolata", "Punica granatum", "Hypericum perforatum",
        "Centranthus ruber", "Lavandula angustifolia", "Pyracantha coccinea", "Calendula officinalis",
        "Lapsana communis", "Tagetes erecta", "Lamium galeobdolon", "Trifolium pratense",
        "Lamium purpureum", "Papaver somniferum", "Trifolium repens", "Papaver rhoeas",
        "Anemone nemorosa", "Cymbalaria muralis", "Lactuca serriola", "Alcea rosea",
        "Sedum album"
    ]


def select_random_forget_classes(n_forget=3):
    """Randomly select classes to forget"""
    all_classes = get_plt_net_mini_classes()
    forget_classes = random.sample(all_classes, n_forget)
    return forget_classes