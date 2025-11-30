"""
ASL CNN Model

This module defines the Convolutional Neural Network architecture for ASL classification.
The model is designed to classify ASL hand signs into 36 classes (0-9, a-z).

Model Architecture:
- Simple CNN with 3 convolutional blocks
- Each block: Conv2D -> ReLU -> MaxPooling -> Dropout
- Dense layers for classification
- Output: 36 classes with softmax activation
"""

try:
    from tensorflow.keras.models import Sequential, load_model  # type: ignore
    from tensorflow.keras.layers import (  # type: ignore
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import (  # type: ignore
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )
except ImportError:
    from keras.models import Sequential, load_model  # type: ignore
    from keras.layers import (  # type: ignore
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    )
    from keras.optimizers import Adam  # type: ignore
    from keras.callbacks import (  # type: ignore
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )


def create_simple_cnn(input_shape=(64, 64, 3), num_classes=36):
    """
    Create an improved CNN model for ASL classification
    
    Enhanced architecture with increased capacity for better accuracy.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        # First Convolutional Block - INCREASED filters to 64
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block - INCREASED filters to 128
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block - INCREASED filters to 256
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Fourth Convolutional Block - NEW for better feature extraction
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Flatten and Dense Layers - INCREASED capacity
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ], name='improved_asl_cnn')
    
    return model


def create_deeper_cnn(input_shape=(64, 64, 3), num_classes=36):
    """
    Create a deeper CNN model with more capacity
    
    Use this if the simple model isn't achieving good accuracy.
    More parameters = more learning capacity but slower training.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ], name='deeper_asl_cnn')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate (float): Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_save_path='models/asl_model.h5'):
    """
    Get training callbacks for model improvement
    
    Callbacks:
    - ModelCheckpoint: Save best model based on validation accuracy
    - EarlyStopping: Stop training if no improvement
    - ReduceLROnPlateau: Reduce learning rate when stuck
    
    Args:
        model_save_path (str): Path to save the best model
        
    Returns:
        list: List of Keras callbacks
    """
    callbacks = [
        # Save the best model based on validation accuracy
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Stop training if validation accuracy doesn't improve for 20 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate if validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def load_trained_model(model_path='models/asl_model.h5'):
    """
    Load a pre-trained model from disk
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        Loaded Keras model
    """
    return load_model(model_path)


def print_model_summary(model):
    """
    Print a detailed summary of the model architecture
    
    Args:
        model: Keras model
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    model.summary()
    print("="*70)
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Model Name: {model.name}")
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Create a simple model
    print("Creating Simple CNN Model...")
    model = create_simple_cnn(input_shape=(64, 64, 3), num_classes=36)
    model = compile_model(model, learning_rate=0.001)
    print_model_summary(model)
    
    print("\n" + "="*70)
    print("Creating Deeper CNN Model...")
    deeper_model = create_deeper_cnn(input_shape=(64, 64, 3), num_classes=36)
    deeper_model = compile_model(deeper_model, learning_rate=0.001)
    print_model_summary(deeper_model)
    
    print("\nModels created successfully!")
    print("Recommendation: Start with the simple model for faster training.")
