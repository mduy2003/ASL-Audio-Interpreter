"""
ASL Dataset Loader

This module handles loading and preprocessing of the ASL dataset for training.
It includes functions for:
- Loading images and labels from directory structure
- Splitting data into train/validation/test sets
- Preprocessing images (resize, normalize)
- Creating TensorFlow/Keras data generators with augmentation

Dataset Info (from preprocessing analysis):
- Total images: 2515
- Classes: 36 (0-9, a-z)
- Original size: 400x400 RGB
- All images are RGB with consistent dimensions
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
except ImportError:
    from keras.preprocessing.image import ImageDataGenerator  # type: ignore
    from keras.utils import to_categorical  # type: ignore
import tensorflow as tf


class ASLDataLoader:
    """
    Data loader for ASL image dataset
    """
    
    def __init__(self, data_dir='data/asl_dataset', img_size=(64, 64), seed=42):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Path to the dataset directory
            img_size (tuple): Target image size (height, width)
            seed (int): Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.seed = seed
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Will be set after loading data
        self.classes = None
        self.num_classes = None
        self.class_to_idx = None
        self.idx_to_class = None
        
    def load_data(self):
        """
        Load all images and labels from the dataset directory
        
        Returns:
            tuple: (images, labels) as numpy arrays
        """
        print(f"Loading data from {self.data_dir}...")
        
        # Get sorted list of class directories
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.num_classes = len(self.classes)
        
        # Create mappings between class names and indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Found {self.num_classes} classes: {self.classes}")
        
        images = []
        labels = []
        
        # Load images from each class directory
        for class_name in self.classes:
            class_path = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in this class
            image_files = [f for f in class_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in self.valid_extensions]
            
            print(f"Loading {len(image_files)} images from class '{class_name}'...")
            
            for img_path in image_files:
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Warning: Could not read {img_path}")
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    img = cv2.resize(img, self.img_size)
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"\nLoaded {len(images)} images successfully!")
        print(f"Image shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return images, labels
    
    def preprocess_images(self, images):
        """
        Normalize images to [0, 1] range
        
        Args:
            images (np.array): Array of images
            
        Returns:
            np.array: Normalized images
        """
        return images / 255.0
    
    def split_data(self, images, labels, train_size=0.7, val_size=0.15, test_size=0.15):
        """
        Split data into train, validation, and test sets
        
        Args:
            images (np.array): Array of images
            labels (np.array): Array of labels
            train_size (float): Proportion of training data
            val_size (float): Proportion of validation data
            test_size (float): Proportion of test data
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "train_size + val_size + test_size must equal 1.0"
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=test_size, 
            random_state=self.seed,
            stratify=labels  # Maintain class distribution
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.seed,
            stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Training set: {len(X_train)} images ({len(X_train)/len(images)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} images ({len(X_val)/len(images)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} images ({len(X_test)/len(images)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        Create data generators with augmentation for training
        
        Args:
            X_train (np.array): Training images
            y_train (np.array): Training labels
            X_val (np.array): Validation images
            y_val (np.array): Validation labels
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Convert labels to categorical (one-hot encoding)
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=15,          # Rotate images by up to 15 degrees
            width_shift_range=0.1,      # Shift images horizontally by 10%
            height_shift_range=0.1,     # Shift images vertically by 10%
            zoom_range=0.1,             # Zoom in/out by 10%
            horizontal_flip=True,       # Flip images horizontally
            fill_mode='nearest'         # Fill empty pixels after transformations
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train_cat,
            batch_size=batch_size,
            seed=self.seed
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val_cat,
            batch_size=batch_size,
            seed=self.seed
        )
        
        return train_generator, val_generator
    
    def prepare_data(self, train_size=0.7, val_size=0.15, test_size=0.15, batch_size=32):
        """
        Complete pipeline: load, preprocess, split, and create generators
        
        Args:
            train_size (float): Proportion of training data
            val_size (float): Proportion of validation data
            test_size (float): Proportion of test data
            batch_size (int): Batch size for training
            
        Returns:
            dict: Dictionary containing all data splits and generators
        """
        # Load data
        images, labels = self.load_data()
        
        # Preprocess
        images = self.preprocess_images(images)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            images, labels, train_size, val_size, test_size
        )
        
        # Create generators for training and validation
        train_gen, val_gen = self.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Convert test labels to categorical
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        return {
            'train_generator': train_gen,
            'val_generator': val_gen,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_categorical': y_test_cat,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }


def load_single_image(image_path, img_size=(64, 64)):
    """
    Load and preprocess a single image for prediction
    
    Args:
        image_path (str): Path to the image file
        img_size (tuple): Target image size (height, width)
        
    Returns:
        np.array: Preprocessed image ready for prediction
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, img_size)
    
    # Normalize
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = ASLDataLoader(
        data_dir='../data/asl_dataset',
        img_size=(64, 64),  # Resize to 64x64 for faster training
        seed=42
    )
    
    # Prepare all data
    data = loader.prepare_data(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        batch_size=32
    )
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Number of classes: {data['num_classes']}")
    print(f"Classes: {data['classes']}")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Test samples: {len(data['X_test'])}")
    print(f"Image shape: {data['X_train'][0].shape}")
    print("="*60)
