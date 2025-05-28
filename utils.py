import numpy as np

import tensorflow as tf
import math

from tensorflow_examples.models.pix2pix import pix2pix # type: ignore

from IPython.display import clear_output
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image

def display(display_list):
    """Wyświetla listę obrazów (Input Image, True Mask, Predicted Mask) obok siebie."""
    plt.figure(figsize=(15, 15))
    
    num_images = len(display_list)
    title = ['Input Image', 'True Mask', 'Predicted Mask'][:num_images]
    
    for i, image in enumerate(display_list):
        plt.subplot(1, num_images, i + 1)
        plt.title(title[i])
        
        # Konwersja tensora do numpy, jeśli to tensor
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        
        # Jeśli obraz ma 1 kanał, wyświetl jako grayscale
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        elif len(image.shape) == 3 and image.shape[2] == 1:
            plt.imshow(image.squeeze(), cmap='gray')
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # RGB - upewnij się, że wartości są w zakresie [0,1] lub [0,255]
            if image.dtype != np.uint8:
                # Normalizuj obraz, jeśli wartości są poza zakresem [0,1]
                if image.max() > 1.0:
                    image = image / 255.0
            plt.imshow(image)
        else:
            # Inny kształt - spróbuj wypakować kanały
            plt.imshow(image)
        
        plt.axis('off')
    plt.show()

def grid_display(display_list, rows=5):
    cols = math.ceil(len(display_list) / rows)
    plt.figure(figsize=(cols * 2, rows * 2))  # Możesz dostosować rozmiar

    for i in range(len(display_list)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def split_image(image_path, tile_size=(256, 256), fill_color=(0, 0, 0)):
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Konwersja na RGB, jeśli np. obraz ma kanał alfa
        img_width, img_height = img.size
        tile_width, tile_height = tile_size

        # Oblicz wymiary po dopełnieniu
        padded_width = ((img_width + tile_width - 1) // tile_width) * tile_width
        padded_height = ((img_height + tile_height - 1) // tile_height) * tile_height

        # Stwórz nowy obraz z dopełnieniem
        padded_img = Image.new("RGB", (padded_width, padded_height), fill_color)
        padded_img.paste(img, (0, 0))

        # Podziel obraz na kafelki
        tiles = []
        for top in range(0, padded_height, int(0.8 * tile_height)):
            for left in range(tile_width, padded_width - tile_width, int(0.8 * tile_width)):
                box = (left, top, left + tile_width, top + tile_height)
                tile = padded_img.crop(box)
                tiles.append(np.array(tile))

        return tiles




class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels