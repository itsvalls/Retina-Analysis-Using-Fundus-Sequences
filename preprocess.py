import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


augmenter = ImageDataGenerator(
    rotation_range=20,        
    width_shift_range=0.1,    
    height_shift_range=0.1,   
    zoom_range=0.15,         
    shear_range=0.1,         
    horizontal_flip=True,     # mirror image (fundus symmetry)
    brightness_range=[0.8, 1.2], 
    fill_mode='nearest'      
)

def preprocess_image(image_path, target_size=(224, 224), augment=False):
    """Reads, resizes, normalizes and optionally augments a single image."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0  # Normalize to [0, 1]

    if augment:
        image = augmenter.random_transform(image)  

    return image

def load_and_preprocess_images(folder_path, augment=False):
    """Loads and preprocesses all images in a folder."""
    image_list = []
    file_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            processed = preprocess_image(img_path, augment=augment)
            image_list.append(processed)
            file_names.append(filename)

    images_array = np.array(image_list)
    return images_array, file_names


if _name_ == "_main_":
    images, names = load_and_preprocess_images("data/frames", augment=True)
    print(f" Loaded and augmented {len(images)} images.")