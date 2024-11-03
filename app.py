import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
from methods import correct_image_by_line, correct_img_by_rotnet
import numpy as np
import cv2


def upload_image():
    """
    Upload an image from the user's file system and display it in the GUI.
    """
    global img_path, img, img_tk
    img_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((300, 300))  # Resize ảnh để hiển thị
        img_tk = ImageTk.PhotoImage(img)
        original_label.config(image=img_tk)
        original_label.image = img_tk


def process_image():
    """
    Process the uploaded image using the selected algorithm and display the result in the GUI.
    """
    if not img_path:
        return

    loading_label.config(text="Loading...")
    process_btn.config(state="disabled")

    thread = threading.Thread(target=process_image_thread)
    thread.start()


def process_image_thread():
    """
    Thread that processes the image using the selected algorithm.
    """
    algorithm = algorithm_var.get()

    if algorithm == "Line-Based Correction":
        processed_img = correct_image_by_line(img, crop=False)
        if processed_img is None:
            processed_img = img
    elif algorithm == "DeepLearning-Based Correction":
        processed_img = correct_img_by_rotnet(img_path, crop=False)

    processed_img.thumbnail((300, 300))
    processed_tk = ImageTk.PhotoImage(processed_img)
    processed_label.config(image=processed_tk)
    processed_label.image = processed_tk

    loading_label.config(text="Done!")
    process_btn.config(state="normal")


root = tk.Tk()
root.title("Image Orientation Correction")
root.geometry("650x400")

img_path = None
img = None

upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)

process_btn = tk.Button(root, text="Process Image", command=process_image)
process_btn.pack(pady=10)

algorithm_var = tk.StringVar(value="Line-Based Correction")
algorithm_menu = ttk.Combobox(root, textvariable=algorithm_var, values=[
                              "Line-Based Correction", "DeepLearning-Based Correction"], state="readonly", width=30)
algorithm_menu.pack(pady=10)

original_label = tk.Label(root)
original_label.pack(side="left", padx=10)

processed_label = tk.Label(root)
processed_label.pack(side="right", padx=10)

loading_label = tk.Label(root, text="")
loading_label.pack(pady=10)

root.mainloop()

