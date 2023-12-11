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
from .classify import *


class ImageViewerApp:
    def __init__(self, root, N=10):
        self.centroid = (None, None)
        self.N = N
        self.segmentation_mode = tk.StringVar(root)
        self.segmentation_mode.set("binary")

        self.classification_type = tk.StringVar(root)
        self.classification_type.set("Multiclass")

        self.classification_mode = tk.StringVar(root)
        self.classification_mode.set("Mahalanobis")

        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("800x500")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_columnconfigure(4, weight=1)
        self.root.grid_columnconfigure(5, weight=1)
        self.root.grid_columnconfigure(6, weight=1)

        self.image_label = ttk.Frame(root)
        self.image_label.grid(row=0, column=0, columnspan=6)

        open_btn = tk.Button(root, text="Open Image", command=self.open_image)
        open_btn.grid(row=2, column=0, pady=5)

        params_dialog_btn = tk.Button(
            root, text="Set parameters", command=self.params_dialog
        )
        params_dialog_btn.grid(row=2, column=1, pady=5)

        segmentation_btn = tk.Button(
            root, text="Segmentate", command=self.segmentatio_dialog
        )
        segmentation_btn.grid(row=2, column=2, pady=5)

        gimme_data_btn = tk.Button(
            root, text="Get descriptors", command=self.descriptor_dialog
        )
        gimme_data_btn.grid(row=2, column=3, pady=5)

        gimme_data_btn = tk.Button(root, text="Classify", command=self.classify_dialog)
        gimme_data_btn.grid(row=2, column=4, pady=5)

        gimme_data_btn = tk.Button(root, text="Plot data", command=self.plot_dialog)
        gimme_data_btn.grid(row=2, column=5, pady=5)

        self.photo = None

    def open_image(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Image files", ".png .jpg")]
        )

        if self.file_path:
            self.image_label = Zoom_Advanced(self.root, self.file_path)
            self.image_label.grid(row=0, column=0, columnspan=4)

    def params_dialog(self):
        dialog = tk.Toplevel()

        if not hasattr(self, 'file_path'):
            tk.Label(dialog, text="Warning: No Image selected!").grid(
                row=0, column=0, padx=5, pady=5, columnspan=1
            )
            return

        entry1 = tk.Entry(dialog, width=10)
        entry1.grid(row=0, column=1, padx=5, pady=5)
        entry2 = tk.Entry(dialog, width=10)
        entry2.grid(row=1, column=1, padx=5, pady=5)
        entry3 = tk.Entry(dialog, width=10)
        entry3.grid(row=2, column=1, padx=5, pady=5)

        if self.centroid[0] is not None and self.centroid[1] is not None:
            entry1.insert(0, str(self.centroid[0]))
            entry2.insert(0, str(self.centroid[1]))
        entry3.insert(0, "100")

        tk.Label(dialog, text="Centroid X:").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(dialog, text="Centroid Y:").grid(row=1, column=0, padx=5, pady=5)
        tk.Label(dialog, text="N:").grid(row=2, column=0, padx=5, pady=5)

        def close():
            if (
                (entry1.get().isdigit() == False or int(entry1.get()) < 0)
                or (entry2.get().isdigit() == False or int(entry2.get()) < 0)
                or (entry3.get().isdigit() == False or int(entry3.get()) < 0)
            ):
                tk.Label(
                    dialog, text="Warning: All data must be positive integer!"
                ).grid(row=4, column=0, padx=5, pady=5, columnspan=2)
                return

            self.centroid = (int(entry1.get()), int(entry2.get()))
            self.N = int(entry3.get())

            img = Image.open(self.file_path)
            img = crop_image_around_point(
                np.array(img), self.centroid[0], self.centroid[1], self.N
            )
            self.image_label = Zoom_Advanced(self.root, img)

            dialog.destroy()

        tk.Button(dialog, text="Close", command=close).grid(
            row=3, column=0, columnspan=2, pady=10
        )

    def segmentatio_dialog(self):
        dialog = tk.Toplevel()

        if self.centroid[0] is None and self.centroid[1] is None:
            tk.Label(dialog, text="Warning: Cell not selected!").grid(
                row=0, column=0, padx=5, pady=5, columnspan=1
            )
            return

        tk.Label(dialog, text="Segmentation mode:").grid(
            row=0, column=0, padx=5, pady=5
        )
        segmentation_drop = tk.OptionMenu(
            dialog,
            self.segmentation_mode,
            "binary",
            "otsu",
            "watershed",
            "region growing",
        )
        segmentation_drop.grid(row=0, column=1, padx=5, pady=5)
        segmentation_drop.config(width=10)

        def close():
            img = Image.open(self.file_path)
            img = crop_image_around_point(
                np.array(img), self.centroid[0], self.centroid[1], self.N
            )

            if self.segmentation_mode.get() == "binary":
                img = binary(img, invert=True)
            elif self.segmentation_mode.get() == "otsu":
                img = otsu(img, invert=True)
            elif self.segmentation_mode.get() == "watershed":
                img = watershed(img)
            elif self.segmentation_mode.get() == "region growing":
                img = region_growing(img)

            self.image_label = Zoom_Advanced(self.root, img)
            self.image_label.grid(row=0, column=0, columnspan=4)
            dialog.destroy()

        action_btn = tk.Button(dialog, text="Close", command=close)
        action_btn.grid(row=1, column=0, columnspan=2, pady=10)

    def descriptor_dialog(self):
        dialog = tk.Toplevel()

        tk.Label(dialog, text="Centroid:").grid(row=0, column=0, padx=5, pady=5)
        new_centroid = calculate_centroid(self.image_label.image)
        tk.Label(dialog, text=f'({new_centroid[0]}, {new_centroid[1]})').grid(
            row=0, column=1, padx=5, pady=5
        )

        origin = getOriginFromCropped(self.centroid[0], self.centroid[1], self.N)
        norm_centroid = ((self.centroid[0] - origin[0]), self.centroid[1] - origin[1])
        tk.Label(dialog, text="Distance from original centroid:").grid(
            row=1, column=0, padx=5, pady=5
        )
        tk.Label(
            dialog, text="{:.2f}".format(math.dist(new_centroid, norm_centroid))
        ).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Area:").grid(row=2, column=0, padx=5, pady=5)
        tk.Label(
            dialog, text="{:.2f}".format(calculate_area(self.image_label.image))
        ).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Perimeter:").grid(row=3, column=0, padx=5, pady=5)
        tk.Label(
            dialog, text="{:.2f}".format(calculate_perimeter(self.image_label.image))
        ).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Compactnes:").grid(row=4, column=0, padx=5, pady=5)
        tk.Label(
            dialog, text="{:.2f}".format(calculate_compactness(self.image_label.image))
        ).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(dialog, text="N:").grid(row=5, column=0, padx=5, pady=5)
        tk.Label(dialog, text="{:.2f}".format(self.N)).grid(
            row=5, column=1, padx=5, pady=5
        )

        def close():
            dialog.destroy()

        action_btn = tk.Button(dialog, text="Close", command=close)
        action_btn.grid(row=6, column=0, columnspan=2, pady=10)

    def classify_dialog(self):
        dialog = tk.Toplevel()

        tk.Label(dialog, text="Classification Type:").grid(
            row=0, column=0, padx=5, pady=5
        )
        type_drop = tk.OptionMenu(
            dialog, self.classification_type, "Binary", "Multiclass"
        )
        type_drop.grid(row=0, column=1, padx=5, pady=5)
        type_drop.config(width=10)

        tk.Label(dialog, text="Classifier:").grid(row=1, column=0, padx=5, pady=5)
        segmentation_drop = tk.OptionMenu(
            dialog,
            self.classification_mode,
            "Mahalanobis",
            "Neural Network (Baseline)",
            "Neural Network (Resnet)",
        )
        segmentation_drop.grid(row=1, column=1, padx=5, pady=5)
        segmentation_drop.config(width=10)

        labels = []
        descriptors = []

        for i in range(6):
            labels.append(tk.Label(dialog, text=""))
            labels[-1].grid(row=i + 2, column=0, padx=5, pady=5)
            descriptors.append(tk.Label(dialog, text=""))
            descriptors[-1].grid(row=i + 2, column=1, padx=5, pady=5)

        tk.Label(dialog, text="Selected class:").grid(row=8, column=0, padx=5, pady=5)
        selected_class = tk.Label(dialog, text="")
        selected_class.grid(row=8, column=1, padx=5, pady=5)

        def close():
            dialog.destroy()

        def classify():
            subpath = (
                'multiclass'
                if self.classification_type.get() == 'Multiclass'
                else 'binary'
            )

            if self.classification_mode.get() == 'Mahalanobis':
                data, selected = classify_mahalanobis(
                    self.image_label.image, subpath=subpath
                )
            else:
                model_type = (
                    'baseline'
                    if self.classification_mode.get() == 'Neural Network (Baseline)'
                    else 'resnet'
                )
                data, selected = classify_nn(
                    self.image_label.image, model_type=model_type, subpath=subpath
                )

            for i, (label, value) in enumerate(data.items()):
                descriptor = '{:.2f}'.format(value)
                labels[i].config(text=label)
                descriptors[i].config(text=descriptor)

            if len(data) < 6:
                for i in range(6 - len(data)):
                    labels[-(i + 1)].config(text='')
                    descriptors[-(i + 1)].config(text='')

            selected_class.config(text=selected)

        action_btn = tk.Button(dialog, text="Close", command=close)
        action_btn.grid(row=9, column=0, columnspan=1, pady=10)

        action_btn = tk.Button(dialog, text="Classify", command=classify)
        action_btn.grid(row=9, column=1, columnspan=1, pady=10)

    def plot_dialog(self):
        dialog = tk.Toplevel()

        image_label = Zoom_Advanced(dialog, './model_data/scatter_all.png')
        image_label.grid(row=0, column=0)

        def close():
            dialog.destroy()

        action_btn = tk.Button(dialog, text="Close", command=close)
        action_btn.grid(row=1, column=0, columnspan=1, pady=10)
