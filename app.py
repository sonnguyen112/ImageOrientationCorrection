import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from utils import angle_error, RotNetDataGenerator, rotate
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
bound_angle = 15

def upload_image():
    global img_path, img
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((300, 300))  # Resize for display purposes
        img_tk = ImageTk.PhotoImage(img)
        original_label.config(image=img_tk)
        original_label.image = img_tk


def combined_loss(y_true, y_pred):
    categorical_loss = K.categorical_crossentropy(y_true, y_pred)
    angle_loss = angle_error(y_true, y_pred)  # Assuming angle_error is defined appropriately
    total_loss = categorical_loss + angle_loss
    return total_loss


def process_image():
    if not img_path:
        return
    
    # Read the image and process it with OpenCV
    predictions = model.predict_generator(
        RotNetDataGenerator(
            img_path,
            input_shape=(224, 224, 3),
            batch_size=1,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        ),
    )

    predicted_angle = np.argmax(predictions, axis=1) - bound_angle
    image = cv2.imread(img_path)
    rotated_image = rotate(image, -predicted_angle)
    
    # Convert processed image to Pillow format
    processed_img = Image.fromarray(rotated_image)
    processed_img.thumbnail((300, 300))  # Resize for display
    processed_tk = ImageTk.PhotoImage(processed_img)
    
    processed_label.config(image=processed_tk)
    processed_label.image = processed_tk

# Initialize the main window
root = tk.Tk()
root.title("Image Processing UI")
root.geometry("650x400")

# Initialize global variables
img_path = None
img = None

# Set up UI components
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)

process_btn = tk.Button(root, text="Process Image", command=process_image)
process_btn.pack(pady=10)

# Labels to display images
original_label = tk.Label(root)
original_label.pack(side="left", padx=10)

processed_label = tk.Label(root)
processed_label.pack(side="right", padx=10)

if __name__ == "__main__":
    model_path = r'models\rotnet_indoor_resnet50.keras'
    model = load_model(model_path, custom_objects={'angle_error': angle_error, 'combined_loss': combined_loss})
    root.mainloop()
