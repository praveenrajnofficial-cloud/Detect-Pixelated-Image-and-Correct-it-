import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image

# Define the input shape
img_height, img_width = 150, 150

# Data Augmentation (Optional)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values (0-1)
    shear_range=0.2,  # Optional: Random shear images for data augmentation
    zoom_range=0.2,  # Optional: Randomly zoom images for data augmentation
    horizontal_flip=True  # Optional: Randomly flip images horizontally
)

# Define paths to your training and test data directories (replace with your actual paths)
train_data_dir = r"/Users/praveen/Desktop/annotated images/train_data"
test_data_dir = r"/Users/praveen/Desktop/annotated images/test data"

# Load training and test data using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,  # Adjust batch size based on your hardware limitations
    class_mode='binary'  # Assuming binary classification (pixelated vs non-pixelated)
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Inherit normalization from train_datagen
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,  # Adjust batch size based on your hardware limitations
    class_mode='binary'  # Assuming binary classification
)

# Define CNN model
# Create the Sequential model
model = Sequential([
    Input(shape=(img_height, img_width, 3)),  # Specify input shape here
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),  # Add another convolutional layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)

# Train the model
model.fit(train_generator,
          epochs=10,  # Adjust number of epochs based on your data and validation
          validation_data=test_generator,
          callbacks=[early_stopping, reduce_lr])

# Save the model for future use
model.save('/Users/praveen/Desktop/annotated_images/model/pixelation_detector_model.h5')

# Use the trained model to predict on a new image
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add a new dimension for batch size
    img = img / 255.0  # Normalize the image

    # Make prediction on the preprocessed image
    prediction = model.predict(img)

    # Interpret the prediction (round the sigmoid output to 0 or 1)
    predicted_class = np.round(prediction[0][0])
    if predicted_class == 0:
        print("Image is classified as non-pixelated.")
    else:
        print("Image is classified as pixelated.")

# Example usage (replace with your image path)
image_path = "/Users/praveen/Downloads/WhatsApp Image 2024-07-05 at 19.44.28.jpeg"
predict_image(image_path)
