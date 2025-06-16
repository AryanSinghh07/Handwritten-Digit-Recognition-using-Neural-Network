import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

class DigitRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.configure(bg='#f0f0f0')
        
        # Style configuration
        style = ttk.Style()
        style.configure('Custom.TButton', padding=10, font=('Arial', 12))
        style.configure('Custom.TLabel', font=('Arial', 14), background='#f0f0f0')
        
        # Main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for drawing
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(self.main_frame, 
                              width=self.canvas_width, 
                              height=self.canvas_height, 
                              bg='white',  # Changed to white background
                              cursor="crosshair")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        
        # Drawing variables
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), color='white')  # Changed to white background
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.line_width = 30  # Increased line width
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Button frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Buttons
        self.predict_btn = ttk.Button(button_frame, 
                                    text="Predict", 
                                    command=self.predict_digit,
                                    style='Custom.TButton')
        self.predict_btn.grid(row=0, column=0, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, 
                                   text="Clear", 
                                   command=self.clear_canvas,
                                   style='Custom.TButton')
        self.clear_btn.grid(row=0, column=1, padx=5)
        
        # Debug button
        self.debug_btn = ttk.Button(button_frame,
                                  text="Debug View",
                                  command=self.show_debug_view,
                                  style='Custom.TButton')
        self.debug_btn.grid(row=0, column=2, padx=5)
        
        # Prediction label with larger font and better formatting
        self.prediction_label = ttk.Label(self.main_frame, 
                                        text="Draw a digit and click Predict",
                                        style='Custom.TLabel')
        self.prediction_label.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Load the model
        try:
            self.model = load_model('digit_model.h5')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Error", "Could not load the model. Please ensure 'digit_model.h5' exists.")
            self.model = None
    
    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_line(self, event):
        if self.last_x and self.last_y:
            x = event.x
            y = event.y
            # Draw on canvas with smoother lines
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                  fill='black',  # Changed to black drawing
                                  width=self.line_width,
                                  capstyle=tk.ROUND,
                                  joinstyle=tk.ROUND,
                                  smooth=True)
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, x, y],
                          fill='black',  # Changed to black drawing
                          width=self.line_width,
                          joint='curve')
            self.last_x = x
            self.last_y = y
    
    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), color='white')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit and click Predict")
    
    def preprocess_image(self, img):
        # Convert to numpy array
        img_array = np.array(img)
        
        # Invert the image (black background, white digit)
        img_array = 255 - img_array
        
        # Find bounding box of digit
        coords = cv2.findNonZero(img_array)
        if coords is None:
            return None
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding to make it square
        size = max(w, h)
        x_pad = (size - w) // 2
        y_pad = (size - h) // 2
        
        # Ensure we have enough space for padding
        if (y-y_pad < 0 or x-x_pad < 0 or 
            y+h+y_pad > img_array.shape[0] or 
            x+w+x_pad > img_array.shape[1]):
            # If not enough space, just crop to bounding box
            cropped = img_array[y:y+h, x:x+w]
        else:
            # If enough space, add padding to make it square
            cropped = img_array[y-y_pad:y+h+y_pad, x-x_pad:x+w+x_pad]
        
        # Resize to slightly smaller than 28x28 to create padding
        target_size = 20  # This will create some padding in the 28x28 image
        resized = cv2.resize(cropped, (target_size, target_size), 
                           interpolation=cv2.INTER_AREA)
        
        # Create 28x28 image with padding
        final_img = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - target_size) // 2
        y_offset = (28 - target_size) // 2
        final_img[y_offset:y_offset+target_size, 
                 x_offset:x_offset+target_size] = resized
        
        # Normalize
        normalized = final_img.astype('float32') / 255.0
        
        # Apply slight Gaussian blur to smooth edges
        normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
        
        return normalized
    
    def show_debug_view(self):
        """Show the preprocessing steps for debugging"""
        # Get the current image
        img = self.image.convert('L')
        img_array = np.array(img)
        
        # Create a figure with all preprocessing steps
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        
        # Original
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('Original')
        
        # After inversion
        inverted = 255 - img_array
        axes[1].imshow(inverted, cmap='gray')
        axes[1].set_title('Inverted')
        
        # After preprocessing
        processed = self.preprocess_image(img)
        if processed is not None:
            axes[2].imshow(processed, cmap='gray')
            axes[2].set_title('Preprocessed')
            
            # Show prediction probabilities
            if self.model is not None:
                prediction = self.model.predict(processed.reshape(1, 28, 28, 1), verbose=0)
                axes[3].bar(range(10), prediction[0])
                axes[3].set_title('Prediction Probabilities')
                axes[3].set_xlabel('Digit')
                axes[3].set_ylabel('Probability')
        
        # Remove axis ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()
    
    def predict_digit(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        # Get image and convert to grayscale
        img = self.image.convert('L')
        
        # Preprocess image
        processed_img = self.preprocess_image(img)
        if processed_img is None:
            self.prediction_label.config(text="Please draw a digit first!")
            return
            
        # Reshape for model input
        img_array = processed_img.reshape(1, 28, 28, 1)
        
        # Get prediction
        prediction = self.model.predict(img_array, verbose=0)
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit] * 100
        
        # Only show high confidence predictions
        if confidence < 40:  # Lowered threshold
            self.prediction_label.config(
                text="Not confident enough. Please draw more clearly."
            )
        else:
            self.prediction_label.config(
                text=f"Prediction: {digit} (Confidence: {confidence:.1f}%)"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognitionGUI(root)
    root.mainloop() 