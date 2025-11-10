"""
ASL Model Training Script

This script handles the complete training pipeline:
- Load and prepare data
- Create and compile model
- Train with callbacks
- Evaluate on test set
- Save trained model
- Generate training visualizations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import ASLDataLoader
from src.model import (
    create_simple_cnn, 
    create_deeper_cnn, 
    compile_model, 
    get_callbacks,
    print_model_summary
)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='output/confusion_matrix.png'):
    """
    Plot confusion matrix to visualize classification errors
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - ASL Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_training_history(history, save_path='output/training_history.png'):
    """
    Plot training and validation accuracy/loss curves
    
    Args:
        history: Keras History object from model.fit()
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test_cat, class_names):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test_cat: Test labels (categorical)
        class_names: List of class names
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test_cat, axis=1)
    
    # Calculate per-class accuracy
    print("\n" + "-"*70)
    print("Per-Class Accuracy:")
    print("-"*70)
    
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            print(f"Class '{class_name}': {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    print("="*70)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }


def train_model(
    model_type='simple',
    img_size=(64, 64),
    batch_size=32,
    epochs=50,
    learning_rate=0.001,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
):
    """
    Main training function
    
    Args:
        model_type (str): 'simple' or 'deeper'
        img_size (tuple): Image size for training
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        train_size (float): Proportion of training data
        val_size (float): Proportion of validation data
        test_size (float): Proportion of test data
        
    Returns:
        tuple: (trained_model, history, evaluation_results, data_info)
    """
    print("="*70)
    print("ASL MODEL TRAINING")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {model_type}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    loader = ASLDataLoader(
        data_dir='data/asl_dataset',
        img_size=img_size,
        seed=42
    )
    
    data = loader.prepare_data(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        batch_size=batch_size
    )
    
    print(f"\nData loaded successfully!")
    print(f"  - Training samples: {len(data['X_train'])}")
    print(f"  - Validation samples: {len(data['X_val'])}")
    print(f"  - Test samples: {len(data['X_test'])}")
    print(f"  - Number of classes: {data['num_classes']}")
    print(f"  - Classes: {data['classes']}")
    
    # Step 2: Create model
    print("\n[2/5] Creating model...")
    if model_type.lower() == 'simple':
        model = create_simple_cnn(
            input_shape=(*img_size, 3),
            num_classes=data['num_classes']
        )
    elif model_type.lower() == 'deeper':
        model = create_deeper_cnn(
            input_shape=(*img_size, 3),
            num_classes=data['num_classes']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'simple' or 'deeper'")
    
    model = compile_model(model, learning_rate=learning_rate)
    print_model_summary(model)
    
    # Step 3: Set up callbacks
    print("\n[3/5] Setting up training callbacks...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    model_save_path = f'models/asl_model_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
    callbacks = get_callbacks(model_save_path=model_save_path)
    
    print(f"Model will be saved to: {model_save_path}")
    
    # Calculate class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(data['y_train']),
        y=data['y_train']
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"Using class weights to handle imbalance")
    print(f"Weight range: {min(class_weights):.3f} to {max(class_weights):.3f}")
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    print("-"*70)
    
    history = model.fit(
        data['train_generator'],
        validation_data=data['val_generator'],
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # ADDED: Handle class imbalance
        verbose=1
    )
    
    print("-"*70)
    print("Training completed!")
    
    # Step 5: Evaluate on test set
    print("\n[5/5] Evaluating model on test set...")
    evaluation = evaluate_model(
        model,
        data['X_test'],
        data['y_test_categorical'],
        data['classes']
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        evaluation['true_classes'],
        evaluation['predicted_classes'],
        data['classes']
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model saved to: {model_save_path}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test accuracy: {evaluation['test_accuracy']:.4f}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return model, history, evaluation, data


if __name__ == "__main__":
    """
    Run training with default parameters
    
    You can modify these parameters to experiment with different configurations:
    - model_type: 'simple' (faster) or 'deeper' (more capacity)
    - img_size: (64, 64) for faster training, (128, 128) for better quality
    - batch_size: 32 is a good default, increase if you have more RAM
    - epochs: Start with 50, can go higher if not overfitting
    - learning_rate: 0.001 is a good starting point
    """
    
    # Train the model - IMPROVED PARAMETERS
    model, history, evaluation, data = train_model(
        model_type='simple',      # Start with simple model
        img_size=(64, 64),        # 64x64 for faster training
        batch_size=32,            # Standard batch size
        epochs=100,               # INCREASED: 100 epochs (was 50) - model needs more time to learn
        learning_rate=0.0005,     # REDUCED: 0.0005 (was 0.001) - slower, more stable learning
        train_size=0.7,           # 70% training
        val_size=0.15,            # 15% validation
        test_size=0.15            # 15% testing
    )
    
    print("\n✓ Training complete! Model ready for predictions.")
