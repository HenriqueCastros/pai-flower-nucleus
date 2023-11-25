import math
import numpy as np
import tkinter as tk
from PIL import Image
from tkinter import ttk
from tkinter import filedialog

from .Zoom import Zoom_Advanced
from .utils import crop_image_around_point, getOriginFromCropped
from .segmentations import *
from .description import *

class ImageViewerApp:
    def __init__(self, root):
        self.centroid = (0, 0)
        self.N = 100
        self.segmentation_mode = tk.StringVar(root)
        self.segmentation_mode.set("binary")

        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("800x500")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        self.image_label = ttk.Frame(root)
        self.image_label.grid(row=0, column=0, columnspan=4)

        open_btn = tk.Button(root, text="Open Image", command=self.open_image)
        open_btn.grid(row=2, column=0, pady=5)

        params_dialog_btn = tk.Button(root, text="Set parameters", command=self.params_dialog)
        params_dialog_btn.grid(row=2, column=1, pady=5)

        segmentation_btn = tk.Button(root, text="Segmentate", command=self.segmentatio_dialog)
        segmentation_btn.grid(row=2, column=2, pady=5)

        gimme_data_btn = tk.Button(root, text="Get descriptors", command=self.descriptor_dialog)
        gimme_data_btn.grid(row=2, column=3, pady=5)

        self.photo = None

    def open_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png .jpg")])

        if self.file_path:
            self.image_label = Zoom_Advanced(self.root, self.file_path)
            self.image_label.grid(row=0, column=0, columnspan=4)
    
    def params_dialog(self):
        dialog = tk.Toplevel()

        entry1 = tk.Entry(dialog, width=10)
        entry1.grid(row=0, column=1, padx=5, pady=5)
        entry2 = tk.Entry(dialog, width=10)
        entry2.grid(row=1, column=1, padx=5, pady=5)
        entry3 = tk.Entry(dialog, width=10)
        entry3.grid(row=2, column=1, padx=5, pady=5)
        
        entry3.insert(0, "100")

        tk.Label(dialog, text="Centroid X:").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(dialog, text="Centroid Y:").grid(row=1, column=0, padx=5, pady=5)
        tk.Label(dialog, text="N:").grid(row=2, column=0, padx=5, pady=5)

        def close():
            if (entry1.get().isdigit() == False or int(entry1.get()) < 0) or \
                (entry2.get().isdigit() == False or int(entry2.get()) < 0) or \
                    (entry3.get().isdigit() == False  or int(entry3.get()) < 0):
                tk.Label(dialog, text="Warning: All data must be positive integer!").grid(row=4, column=0, padx=5, pady=5, columnspan=2)
                return

            self.centroid = (int(entry1.get()), int(entry2.get()))
            self.N = int(entry3.get())
            dialog.destroy()

        tk.Button(dialog, text="Close", command=close).grid(row=3, column=0, columnspan=2, pady=10)

    def segmentatio_dialog(self):
        dialog = tk.Toplevel()

        tk.Label(dialog, text="Segmentation mode:").grid(row=0, column=0, padx=5, pady=5)
        segmentation_drop = tk.OptionMenu(dialog, self.segmentation_mode, "binary", "otsu", "watershed")
        segmentation_drop.grid(row=0, column=1, padx=5, pady=5)
        segmentation_drop.config(width=10)

        def close():
            img = Image.open(self.file_path)
            img = crop_image_around_point(np.array(img), self.centroid[0], self.centroid[1], self.N)

            if self.segmentation_mode.get() == "binary":
                img = binary(img, invert=True)
            elif self.segmentation_mode.get() == "otsu":
                img = otsu(img, invert=True)
            elif self.segmentation_mode.get() == "watershed":
                img = watershed(img)

            self.image_label = Zoom_Advanced(self.root, img)
            self.image_label.grid(row=0, column=0, columnspan=4)
            dialog.destroy()

        action_btn = tk.Button(dialog, text="Close", command=close)
        action_btn.grid(row=1, column=0, columnspan=2, pady=10)

    def descriptor_dialog(self):
        dialog = tk.Toplevel()

        tk.Label(dialog, text="Centroid:").grid(row=0, column=0, padx=5, pady=5)
        new_centroid = calculate_centroid(self.image_label.image)
        tk.Label(dialog, text=f'({new_centroid[0]}, {new_centroid[1]})').grid(row=0, column=1, padx=5, pady=5)

        origin = getOriginFromCropped(self.centroid[0], self.centroid[1], self.N)
        norm_centroid = ((self.centroid[0] - origin[0]), self.centroid[1] - origin[1])
        tk.Label(dialog, text="Distance from original centroid:").grid(row=1, column=0, padx=5, pady=5)
        tk.Label(dialog, text=math.dist(new_centroid, norm_centroid)).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Area:").grid(row=2, column=0, padx=5, pady=5)
        tk.Label(dialog, text=calculate_area(self.image_label.image)).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Perimeter:").grid(row=3, column=0, padx=5, pady=5)
        tk.Label(dialog, text=calculate_perimeter(self.image_label.image)).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Compactnes:").grid(row=4, column=0, padx=5, pady=5)
        tk.Label(dialog, text=calculate_compactness(self.image_label.image)).grid(row=4, column=1, padx=5, pady=5)
        
        tk.Label(dialog, text="N:").grid(row=5, column=0, padx=5, pady=5)
        tk.Label(dialog, text=self.N).grid(row=5, column=1, padx=5, pady=5)

        def close():
            dialog.destroy()

        action_btn = tk.Button(dialog, text="Close", command=close)
        action_btn.grid(row=6, column=0, columnspan=2, pady=10)