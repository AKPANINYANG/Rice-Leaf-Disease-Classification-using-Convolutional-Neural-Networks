import argparse
import sys
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images_and_labels(image_dict, labels_dict):
    X, y = [], []
    for name, images in image_dict.items():
        for image in images:
            if isinstance(image, Image.Image):
                img = np.array(image)
            else:
                img = cv2.imread(str(image))
        
            if img is not None:
                resized_img = cv2.resize(img,(180,180))
                X.append(resized_img)
                y.append(labels_dict[name])
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255
    return X_train_scaled, y_train

def build_and_train_model(X_train_scaled, y_train):
    num_classes = 3
    model = tf.keras.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=30)
    return model

def main(args):
    # Load images
    bacterial_blight = load_images_from_folder(args.bacterial_blight_folder)
    brown_spot = load_images_from_folder(args.brown_spot_folder)
    leaf_smut = load_images_from_folder(args.leaf_smut_folder)

    # Preprocess images and labels
    image_dict = {
        'Bacterial_leaf_blight': bacterial_blight,
        'Brown_spot': brown_spot,
        'Leaf_smut': leaf_smut
    }
    labels_dict = {
        'Bacterial_leaf_blight': 0,
        'Brown_spot': 1,
        'Leaf_smut': 2
    }
    X_train_scaled, y_train = preprocess_images_and_labels(image_dict, labels_dict)

    # Build and train model
    model = build_and_train_model(X_train_scaled, y_train)

    # Save the model
    model.save(args.model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN for rice leaf disease classification.')
    parser.add_argument('--bacterial_blight_folder', type=str, help='Path to the folder containing bacterial blight images.')
    parser.add_argument('--brown_spot_folder', type=str, help='Path to the folder containing brown spot images.')
    parser.add_argument('--leaf_smut_folder', type=str, help='Path to the folder containing leaf smut images.')
    parser.add_argument('--model_output', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    main(args)
