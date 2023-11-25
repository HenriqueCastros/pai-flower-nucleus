import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from .utils import scale_proportional
from .segmentations import binary, otsu, watershed

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

        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=0, columnspan=4)

        open_btn = tk.Button(root, text="Open Image", command=self.open_image)
        open_btn.grid(row=1, column=0, pady=5)

        open_dialog_btn = tk.Button(root, text="Set parameters", command=self.open_dialog)
        open_dialog_btn.grid(row=1, column=1, pady=5)

        segmentation_drop = tk.OptionMenu(root, self.segmentation_mode, "binary", "otsu", "watershed")
        segmentation_drop.grid(row=1, column=2, pady=5)

        gimme_data_btn = tk.Button(root, text="Get descriptors", command=lambda: print(self.centroid, self.N))
        gimme_data_btn.grid(row=1, column=3, pady=5)

        self.photo = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png .jpg")])

        if file_path:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape = image.shape
            shape = (shape[1], shape[0])

            image = Image.fromarray(image)
            image = image.resize(scale_proportional(shape, (800,400)))

            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
    
    def open_dialog(self):
        dialog = tk.Tk()

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

