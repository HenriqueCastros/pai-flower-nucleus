import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from utils import scale_proportional
from segmentations import binary_segmentation, otsu_segmentation

class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("800x400")

        self.image_label = tk.Label(root)
        self.image_label.pack(padx=10, pady=10)

        open_button = tk.Button(root, text="Open Image", command=self.open_image)
        open_button.pack(pady=10)

        self.photo = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png .jpg")])

        if file_path:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = otsu_segmentation(image)
            shape = image.shape
            shape = (shape[1], shape[0])

            image = Image.fromarray(image)
            image = image.resize(scale_proportional(shape, (800,400)))

            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
