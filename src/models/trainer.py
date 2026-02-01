"""Model training pipeline for the Fire Detection System."""

import os
from pathlib import Path
from typing import Optional
import yaml
from ultralytics import YOLO
import torch


class ModelTrainer:
    """Handles fine-tuning of pre-trained models for fire/smoke detection."""
    
    def __init__(self, base_model: str, dataset_path: str, config_path: str = "config.yaml"):
        """
        Initialize model trainer.
        
        Args:
            base_model: Path to base pre-trained model or model name (e.g., 'yolov8n-seg.pt')
            dataset_path: Path to dataset directory containing fire images and negatives
            config_path: Path to configuration file
        """
        self.base_model = base_model
        self.dataset_path = Path(dataset_path)
        self.config_path = config_path
        self.model: Optional[YOLO] = None
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Dataset paths
        self.fire_images_path = self.dataset_path / "positive-images"
        self.negative_images_path = self.dataset_path / "negatives"
        
        if not self.fire_images_path.exists():
            raise FileNotFoundError(f"Fire images directory not found: {self.fire_images_path}")
        if not self.negative_images_path.exists():
            raise FileNotFoundError(f"Negative images directory not found: {self.negative_images_path}")
    
    def prepare_dataset(self) -> None:
        """
        Prepare dataset for training by creating YOLO format dataset structure.
        
        This creates a dataset.yaml file and organizes images into train/val/test splits.
        """
        # Create dataset directory structure
        dataset_root = Path("dataset")
        dataset_root.mkdir(exist_ok=True)
        
        # Create train/val/test directories
        for split in ['train', 'val', 'test']:
            (dataset_root / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_root / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all fire and negative images
        fire_images = list(self.fire_images_path.glob('*.png')) + list(self.fire_images_path.glob('*.jpg'))
        negative_images = list(self.negative_images_path.glob('*.png')) + list(self.negative_images_path.glob('*.jpg'))
        
        print(f"Found {len(fire_images)} fire images and {len(negative_images)} negative images")
        
        # Split ratios from config
        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']
        
        # Split fire images
        n_fire = len(fire_images)
        n_train_fire = int(n_fire * train_split)
        n_val_fire = int(n_fire * val_split)
        
        train_fire = fire_images[:n_train_fire]
        val_fire = fire_images[n_train_fire:n_train_fire + n_val_fire]
        test_fire = fire_images[n_train_fire + n_val_fire:]
        
        # Split negative images
        n_neg = len(negative_images)
        n_train_neg = int(n_neg * train_split)
        n_val_neg = int(n_neg * val_split)
        
        train_neg = negative_images[:n_train_neg]
        val_neg = negative_images[n_train_neg:n_train_neg + n_val_neg]
        test_neg = negative_images[n_train_neg + n_val_neg:]
        
        # Copy images and create dummy labels (for now, we'll need proper annotations)
        # Note: In a real implementation, you would need proper bounding box/segmentation annotations
        import shutil
        
        def copy_images(image_list, split_name, is_fire=True):
            for img_path in image_list:
                # Copy image
                dest_img = dataset_root / split_name / 'images' / img_path.name
                shutil.copy(img_path, dest_img)
                
                # Create empty label file (placeholder - needs real annotations)
                label_path = dataset_root / split_name / 'labels' / f"{img_path.stem}.txt"
                label_path.touch()
        
        copy_images(train_fire, 'train', is_fire=True)
        copy_images(val_fire, 'val', is_fire=True)
        copy_images(test_fire, 'test', is_fire=True)
        copy_images(train_neg, 'train', is_fire=False)
        copy_images(val_neg, 'val', is_fire=False)
        copy_images(test_neg, 'test', is_fire=False)
        
        # Create dataset.yaml
        dataset_yaml = {
            'path': str(dataset_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {
                0: 'fire',
                1: 'smoke'
            },
            'nc': 2  # number of classes
        }
        
        with open(dataset_root / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        print(f"Dataset prepared at {dataset_root}")
        print(f"Train: {len(train_fire) + len(train_neg)} images")
        print(f"Val: {len(val_fire) + len(val_neg)} images")
        print(f"Test: {len(test_fire) + len(test_neg)} images")
    
    def train(self, epochs: Optional[int] = None, batch_size: Optional[int] = None):
        """
        Train the model with metrics tracking.
        
        Args:
            epochs: Number of training epochs (uses config if not specified)
            batch_size: Batch size for training (uses config if not specified)
            
        Returns:
            TrainingHistory object with all metrics
        """
        from src.data.models import TrainingMetrics
        import time
        
        # Use config values if not specified
        if epochs is None:
            epochs = self.config['training']['epochs']
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        
        # Load base model
        print(f"Loading base model: {self.base_model}")
        self.model = YOLO(self.base_model)
        
        # Training parameters
        img_size = self.config['training']['img_size']
        learning_rate = self.config['training']['learning_rate']
        patience = self.config['training']['patience']
        
        # Train the model
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print(f"Learning rate: {learning_rate}")
        
        results = self.model.train(
            data='dataset/dataset.yaml',
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            patience=patience,
            save=True,
            verbose=True,
            plots=True
        )
        
        # Extract training history
        training_history = []
        
        # YOLOv8 stores metrics in results object
        # Access the training metrics from the results
        if hasattr(results, 'results_dict'):
            metrics_dict = results.results_dict
        else:
            # Fallback: read from CSV file that YOLO creates
            csv_path = Path(self.model.trainer.save_dir) / 'results.csv'
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()  # Remove whitespace from column names
                
                for idx, row in df.iterrows():
                    epoch_metrics = TrainingMetrics(
                        epoch=int(idx),
                        train_loss=float(row.get('train/box_loss', 0.0) + row.get('train/cls_loss', 0.0) + row.get('train/dfl_loss', 0.0)),
                        train_accuracy=float(row.get('metrics/precision(B)', 0.0)),
                        val_loss=float(row.get('val/box_loss', 0.0) + row.get('val/cls_loss', 0.0) + row.get('val/dfl_loss', 0.0)),
                        val_accuracy=float(row.get('metrics/recall(B)', 0.0)),
                        learning_rate=learning_rate,
                        timestamp=time.time()
                    )
                    training_history.append(epoch_metrics)
                    
                    # Print epoch metrics
                    print(f"\nEpoch {epoch_metrics.epoch + 1}/{epochs}")
                    print(f"  Train Loss: {epoch_metrics.train_loss:.4f}")
                    print(f"  Train Accuracy: {epoch_metrics.train_accuracy:.4f}")
                    print(f"  Val Loss: {epoch_metrics.val_loss:.4f}")
                    print(f"  Val Accuracy: {epoch_metrics.val_accuracy:.4f}")
        
        print("\nTraining complete!")
        
        # Display final summary
        if training_history:
            final_metrics = training_history[-1]
            print("\n" + "="*50)
            print("TRAINING SUMMARY")
            print("="*50)
            print(f"Total Epochs: {len(training_history)}")
            print(f"Final Train Loss: {final_metrics.train_loss:.4f}")
            print(f"Final Train Accuracy: {final_metrics.train_accuracy:.4f}")
            print(f"Final Val Loss: {final_metrics.val_loss:.4f}")
            print(f"Final Val Accuracy: {final_metrics.val_accuracy:.4f}")
            print("="*50)
        
        return training_history
    
    def validate(self):
        """
        Run validation on the trained model.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        print("\nRunning validation...")
        results = self.model.val()
        
        # Extract validation metrics
        val_metrics = {
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            'map50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            'map50_95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        }
        
        print("\nValidation Results:")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  mAP@0.5: {val_metrics['map50']:.4f}")
        print(f"  mAP@0.5:0.95: {val_metrics['map50_95']:.4f}")
        
        return val_metrics
    
    def save_model(self, output_path: str) -> None:
        """
        Save the fine-tuned model weights.
        
        Args:
            output_path: Path where to save the model
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export the model
        # YOLOv8 automatically saves the best model during training
        # We'll copy it to the specified location
        best_model_path = Path(self.model.trainer.save_dir) / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, output_path)
            print(f"\nModel saved to: {output_path}")
        else:
            print(f"\nWarning: Best model not found at {best_model_path}")
            print("Saving current model state...")
            self.model.save(output_path)
    
    def save_training_history(self, training_history, output_path: str) -> None:
        """
        Save training history to a JSON file.
        
        Args:
            training_history: List of TrainingMetrics objects
            output_path: Path where to save the training history
        """
        import json
        from dataclasses import asdict
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert training history to dict format
        history_dict = {
            'epochs': len(training_history),
            'metrics': [asdict(m) for m in training_history]
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to: {output_path}")
