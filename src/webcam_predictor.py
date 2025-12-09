"""
Real-time ASL Webcam Predictor

This module captures webcam feed and performs real-time ASL sign prediction.
Features:
- Live webcam capture
- Frame preprocessing to match model input
- Real-time predictions with confidence scores
- Visual feedback with bounding boxes and text overlays
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import ASLDataLoader


class WebcamPredictor:
    """
    Real-time ASL sign predictor using webcam feed
    """
    
    def __init__(self, model_path='models/asl_model_v2_production.h5', img_size=(128, 128), 
                 use_background_removal=True):
        """
        Initialize the webcam predictor
        
        Args:
            model_path (str): Path to the trained model file
            img_size (tuple): Image size expected by the model (height, width)
            use_background_removal (bool): Apply background removal for better accuracy
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.classes = None
        self.cap = None
        self.use_background_removal = use_background_removal
        
        # Background subtraction for hand isolation
        self.bg_subtractor = None
        self.bg_frames = []
        self.bg_calibrated = False
        
        # Prediction smoothing
        self.prediction_buffer = []
        self.buffer_size = 5  # Average predictions over last 5 frames
        
        # Display settings
        self.roi_size = 300  # Size of the region of interest box
        self.confidence_threshold = 0.5  # Minimum confidence to display prediction (lowered)
        self.debug_mode = False  # Show preprocessed image
        
        # Load model and setup
        self._load_model()
        self._initialize_classes()
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def _initialize_classes(self):
        """Initialize class names (0-9, a-z)"""
        # Numbers 0-9 then letters a-z
        self.classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
        print(f"✓ Initialized {len(self.classes)} classes: {', '.join(self.classes)}")
    
    def remove_background(self, frame):
        """
        Remove background using multiple skin color detection methods
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            Frame with background removed (BLACK background to match training data)
        """
        # Method 1: HSV-based skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Expanded range for better skin tone coverage
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        
        # Second range to catch different skin tones
        lower_skin2 = np.array([0, 40, 60], dtype=np.uint8)
        upper_skin2 = np.array([25, 150, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Method 2: YCrCb color space (often better for skin detection)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine all masks
        mask = cv2.bitwise_or(mask, mask_ycrcb)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find largest contour (assumed to be the hand)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Create new mask with only the largest contour
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Apply Gaussian blur to smooth edges
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        # Create BLACK background (to match training data)
        black_bg = np.zeros_like(frame)
        
        # Combine hand with black background using the mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (frame * mask_3channel + black_bg * (1 - mask_3channel)).astype(np.uint8)
        
        return result
    
    def normalize_lighting(self, frame):
        """
        Normalize lighting using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            frame: BGR frame
            
        Returns:
            Frame with normalized lighting
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back to BGR
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def preprocess_frame(self, frame, apply_bg_removal=True):
        """
        Preprocess a frame for model prediction
        
        Args:
            frame: Raw BGR frame from webcam
            apply_bg_removal: Whether to apply background removal
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Normalize lighting first
        frame = self.normalize_lighting(frame)
        
        # Apply background removal if enabled
        if apply_bg_removal and self.use_background_removal:
            frame = self.remove_background(frame)
        
        # Resize to model input size
        resized = cv2.resize(frame, self.img_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype('float32') / 255.0
        
        # Add batch dimension
        batch = np.expand_dims(normalized, axis=0)
        
        return batch
    
    def predict(self, frame):
        """
        Make a prediction on a single frame
        
        Args:
            frame: Preprocessed frame
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        predictions = self.model.predict(frame, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = self.classes[predicted_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def smooth_prediction(self, prediction, confidence):
        """
        Smooth predictions over multiple frames to reduce jitter
        
        Args:
            prediction: Current frame prediction
            confidence: Confidence score
            
        Returns:
            tuple: (smoothed_prediction, smoothed_confidence)
        """
        # Add to buffer
        self.prediction_buffer.append((prediction, confidence))
        
        # Keep only last N predictions
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        
        # Get most common prediction with high confidence
        if len(self.prediction_buffer) >= 3:  # Need at least 3 frames
            high_conf_predictions = [p for p, c in self.prediction_buffer if c > self.confidence_threshold]
            if high_conf_predictions:
                # Return most common high-confidence prediction
                most_common = max(set(high_conf_predictions), key=high_conf_predictions.count)
                avg_confidence = np.mean([c for p, c in self.prediction_buffer if p == most_common])
                return most_common, avg_confidence
        
        # Not enough data or confidence, return current
        return prediction, confidence
    
    def draw_roi_box(self, frame):
        """
        Draw the region of interest box where user should place hand
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            tuple: (frame with box, roi_coords (x1, y1, x2, y2))
        """
        h, w = frame.shape[:2]
        
        # Center the ROI box
        x1 = (w - self.roi_size) // 2
        y1 = (h - self.roi_size) // 2
        x2 = x1 + self.roi_size
        y2 = y1 + self.roi_size
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add instruction text
        cv2.putText(frame, "Place hand in green box", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, (x1, y1, x2, y2)
    
    def draw_prediction(self, frame, prediction, confidence, all_probs=None, top_n=3):
        """
        Draw prediction results on frame
        
        Args:
            frame: Video frame to draw on
            prediction: Predicted class
            confidence: Confidence score
            all_probs: All class probabilities
            top_n: Number of top predictions to show
            
        Returns:
            Frame with prediction overlay
        """
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Main prediction
        if confidence > self.confidence_threshold:
            color = (0, 255, 0)  # Green for confident
            text = f"Prediction: {prediction.upper()}"
        else:
            color = (0, 165, 255)  # Orange for uncertain
            text = f"Prediction: {prediction.upper()} (?)"
        
        cv2.putText(frame, text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Confidence
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, conf_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show top 3 predictions
        if all_probs is not None:
            top_indices = np.argsort(all_probs)[-top_n:][::-1]
            y_offset = 110
            for i, idx in enumerate(top_indices):
                prob = all_probs[idx]
                class_name = self.classes[idx]
                text = f"{i+1}. {class_name.upper()}: {prob:.1%}"
                cv2.putText(frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
        
        # Controls and status
        bg_status = "BG Removal: ON" if self.use_background_removal else "BG Removal: OFF"
        cv2.putText(frame, bg_status, (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(frame, "q: quit | c: capture | d: debug | b: toggle BG", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def extract_roi(self, frame, roi_coords):
        """
        Extract region of interest from frame
        
        Args:
            frame: Full video frame
            roi_coords: (x1, y1, x2, y2) coordinates
            
        Returns:
            ROI frame
        """
        x1, y1, x2, y2 = roi_coords
        return frame[y1:y2, x1:x2]
    
    def start(self):
        """
        Start the webcam prediction loop
        """
        print("\n" + "="*70)
        print("STARTING REAL-TIME ASL PREDICTOR")
        print("="*70)
        print("Instructions:")
        print("  1. Place your hand in the green box")
        print("  2. Make ASL sign (letters a-z or numbers 0-9)")
        print("  3. Press 'q' to quit")
        print("  4. Press 'c' to capture/save current frame")
        print("  5. Press 'd' to toggle debug view")
        print("  6. Press 'b' to toggle background removal")
        print("="*70 + "\n")
        print("Tips for better accuracy:")
        print("  - Use good lighting")
        print("  - Keep hand centered in box")
        print("  - Try enabling background removal (press 'b')")
        print("  - Keep background simple/plain")
        print("="*70 + "\n")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("✗ Error: Could not open webcam")
            return
        
        print("✓ Webcam opened successfully")
        print("✓ Starting prediction loop...\n")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("✗ Error: Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Draw ROI box
                frame, roi_coords = self.draw_roi_box(frame)
                
                # Extract ROI for prediction
                roi = self.extract_roi(frame, roi_coords)
                
                # Preprocess and predict every few frames to reduce lag
                if frame_count % 3 == 0:  # Predict every 3 frames
                    processed = self.preprocess_frame(roi)
                    prediction, confidence, all_probs = self.predict(processed)
                    
                    # Smooth prediction
                    prediction, confidence = self.smooth_prediction(prediction, confidence)
                    
                    # Store probabilities for display
                    self.all_probs = all_probs
                
                # Draw prediction on frame
                if hasattr(self, 'prediction'):
                    probs = self.all_probs if hasattr(self, 'all_probs') else None
                    frame = self.draw_prediction(frame, prediction, confidence, probs)
                
                # Store for next iteration
                self.prediction = prediction
                
                # Display frame
                cv2.imshow('ASL Real-time Predictor', frame)
                
                # Show debug window if enabled
                if self.debug_mode:
                    # Show what the model sees
                    debug_img = self.preprocess_frame(roi)[0]
                    debug_display = (debug_img * 255).astype(np.uint8)
                    debug_display = cv2.cvtColor(debug_display, cv2.COLOR_RGB2BGR)
                    debug_display = cv2.resize(debug_display, (300, 300))
                    cv2.imshow('Model Input (128x128)', debug_display)
                    
                    # Also show the ROI with background removed
                    if self.use_background_removal:
                        roi_processed = self.remove_background(roi)
                        roi_processed = cv2.resize(roi_processed, (300, 300))
                        cv2.imshow('Background Removed', roi_processed)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✓ Quitting...")
                    break
                elif key == ord('c'):
                    # Save current frame
                    filename = f"captured_sign_{prediction}_{frame_count}.jpg"
                    cv2.imwrite(filename, roi)
                    print(f"✓ Saved frame: {filename}")
                elif key == ord('d'):
                    # Toggle debug mode
                    self.debug_mode = not self.debug_mode
                    print(f"✓ Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('b'):
                    # Toggle background removal
                    self.use_background_removal = not self.use_background_removal
                    print(f"✓ Background removal: {'ON' if self.use_background_removal else 'OFF'}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Webcam released and windows closed")
            print("="*70)
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.cap is not None:
            self.cap.release()


def main():
    """
    Main function to run the webcam predictor
    """
    # Create predictor instance
    predictor = WebcamPredictor(
        model_path='models/asl_model_v2_production.h5',
        img_size=(128, 128),
        use_background_removal=True  # Enable background removal by default
    )
    
    # Start prediction loop
    predictor.start()


if __name__ == "__main__":
    main()
