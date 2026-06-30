import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the input shape
img_height, img_width = 150, 150

# Function to recreate the model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
    ])
    return model

# Load the trained model
model_path = r"/Users/praveen/Desktop/annotated image project /model/pixelation_detector_model.h5"
model = create_model()
model.load_weights(model_path)

# Function to predict whether an image is pixelated or not
def predict_image(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add a new dimension for batch size
        img = img / 255.0  # Normalize the image

        # Make prediction on the preprocessed image
        prediction = model.predict(img)

        # Interpret the prediction (round the sigmoid output to 0 or 1)
        predicted_class = np.round(prediction[0][0])
        return int(predicted_class)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage (replace with your image paths and true labels)
test_images = [
    (r"/Users/praveen/Downloads/WhatsApp Image 2024-07-15 at 21.46.34.jpeg", 1),  # (image_path, true_label: 0 for non-pixelated, 1 for pixelated)
    (r"/Users/praveen/Downloads/WhatsApp Image 2024-07-15 at 21.46.34.jpeg", 0),
    (r"/Users/praveen/Downloads/WhatsApp Image 2024-07-15 at 21.47.18.jpeg",1),
    (r"/Users/praveen/Downloads/WhatsApp Image 2024-07-15 at 21.48.22.jpeg",1),
    (r"/Users/praveen/Downloads/WhatsApp Image 2024-07-15 at 21.50.36.jpeg",0),
    # Add more test images and their true labels here
]

true_labels = []
predicted_labels = []

for image_path, true_label in test_images:
    true_labels.append(true_label)
    predicted_label = predict_image(image_path)
    if predicted_label is not None:
        predicted_labels.append(predicted_label)

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
