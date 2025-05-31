import numpy as np
import cv2

import tensorflow as tf
import math

from tensorflow_examples.models.pix2pix import pix2pix # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
from tabulate import tabulate

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

def read_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")  
        image_array = np.array(img)
  
    return image_array

def split_numpy_image(img, result, tile_size=(256, 256), filter_black = True, fill_color=(0, 0, 0)):
    img_height, img_width = img.shape[:2]
    tile_width, tile_height = tile_size


    pil_img = Image.fromarray(img)
    pil_result = Image.fromarray(result)

    padded_width = ((img_width + tile_width - 1) // tile_width) * tile_width
    padded_height = ((img_height + tile_height - 1) // tile_height) * tile_height

    padded_img = Image.new("RGB", (padded_width, padded_height), fill_color)
    padded_img.paste(pil_img, (0, 0))

    padded_result = Image.new("L", (padded_width, padded_height), 0)
    padded_result.paste(pil_result, (0, 0))

    center_y = tile_size[0] // 2
    center_x = tile_size[1] // 2

    images = []
    results = []
    counter = 0
    for top in range(0, padded_height, int(tile_height)):
        for left in range(tile_width, padded_width - tile_width, int(tile_width)):
            box = (left, top, left + tile_width, top + tile_height)
            
            result_tile = padded_result.crop(box)
            result_tile_array = (np.array(result_tile) / 255)

            if (filter_black and (np.all(result_tile_array == 0) or np.all(result_tile_array == 1))):
                continue
            
            tile = padded_img.crop(box)
            tile_array = np.array(tile)

            center_pixel = result_tile_array[center_y, center_x]

            result_value = center_pixel 

            if (counter % 5000 == 0):
                display([tile_array, result_tile_array])
                print(result_tile_array)
                print(result_value)
            counter += 1

            images.append(tile_array / 255)  
            results.append(result_value)

    return images, results

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

def add_mask_to_RGB(file_path, mask_path,):
    color_img = cv2.imread(file_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])

    masked_color = cv2.bitwise_and(color_img, mask_3ch)
    masked_color = cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB)

    return masked_color

def add_mask_to_grey(file_path, mask_path):
    gray_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if gray_img is None or mask_img is None:
        raise ValueError("Nie udało się wczytać jednego z obrazów.")

    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    masked_img = cv2.bitwise_and(gray_img, binary_mask)

    return masked_img 

def evaluate_segmentation(img_pred, img_true):

    # Sprawdzenie maksymalnych wartości
    print("Maksymalna wartość w predykcji:", img_pred.max())
    print("Maksymalna wartość w masce eksperckiej:", img_true.max())

    # Wizualizacja: Segmentacja vs Maska ekspercka
    fig_result = plt.figure(figsize=(12, 6))
    fig_result.suptitle('Porównanie przetwarzania vs maska ekspercka')

    # Segmentacja
    ax1 = fig_result.add_subplot(1, 2, 1)
    ax1.imshow(img_pred, cmap='gray')
    ax1.set_title("Obraz po przetwarzaniu")
    ax1.axis('off')

    # Maska ekspercka
    ax2 = fig_result.add_subplot(1, 2, 2)
    ax2.imshow(img_true, cmap='gray')
    ax2.set_title("Maska ekspercka")
    ax2.axis('off')

    fig_result.tight_layout()
    plt.show()

    # Normalizacja do zakresu [0, 1]
    y_true = img_true.ravel() / img_true.max()
    y_pred = img_pred.ravel() / img_pred.max()

    # Binaryzacja: naczynia = 1, tło = 0
    y_true_binary = (y_true > 0.5).astype(np.uint8)
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)

    # Obliczenie metryk i macierzy pomyłek
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # Accuracy (dokładność): ogólny odsetek poprawnych klasyfikacji
    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    # Sensitivity (czułość / recall): jak dobrze wykrywane są naczynia (True Positive Rate)
    sensitivity = recall_score(y_true_binary, y_pred_binary)

    # Specificity (swoistość): jak dobrze wykrywane jest tło (True Negative Rate)
    specificity = tn / (tn + fp)

    # Balanced Accuracy: średnia czułości i swoistości (bardziej odporna na niezbalansowane dane)
    balanced_accuracy = (sensitivity + specificity) / 2

    # G-Mean: geometryczna średnia czułości i swoistości (karze nierównowagę)
    g_mean = math.sqrt(sensitivity * specificity)

    # Wyświetlenie metryk w tabeli
    headers = ["Method", "Accuracy", "Sensitivity", "Specificity", "Balanced Accuracy", "G-Mean"]
    data = [["Normalized", f"{accuracy:.4f}", f"{sensitivity:.4f}", f"{specificity:.4f}",
            f"{balanced_accuracy:.4f}", f"{g_mean:.4f}"]]
    print(tabulate(data, headers=headers, tablefmt="pipe"))

    # Wizualizacja wyników
    fig_result = plt.figure(figsize=(12, 6))
    fig_result.suptitle('Ewaluacja wyników: Normalized')

    # Obraz wynikowy
    ax_img = fig_result.add_subplot(1, 2, 1)
    ax_img.imshow(img_pred, cmap='gray')
    ax_img.set_title("Prediction Image")

    # Macierz pomyłek
    ax_cm = fig_result.add_subplot(1, 2, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, cmap='Reds', colorbar=True)
    ax_cm.set_title("Confusion Matrix")

    # Tekst z metrykami (opisowymi)
    metrics_text = (
        f"Accuracy (Dokładność): {accuracy:.4f}\n"
        f"Sensitivity (Czułość): {sensitivity:.4f}\n"
        f"Specificity (Swoistość): {specificity:.4f}\n"
        f"Balanced Accuracy: {balanced_accuracy:.4f}\n"
        f"G-Mean: {g_mean:.4f}"
    )
    fig_result.text(0.02, 0.02, metrics_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


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
  


  