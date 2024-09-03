import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping

# Check TensorFlow version
print(tf.__version__)  # This should print the version of TensorFlow installed

# Define paths to training and validation datasets
train_dir = r'C:\Users\rathn\OneDrive\Documents\Tensorflow_new_recycle\dataset\train'
validation_dir = r'C:\Users\rathn\OneDrive\Documents\Tensorflow_new_recycle\dataset\validation'

# Check the number of classes by counting subdirectories
classes = os.listdir(train_dir)
print(f"Classes: {classes}")

# Image data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Dynamically determine the number of classes
num_classes_train = len(train_generator.class_indices)
num_classes_val = len(validation_generator.class_indices)

# Print class indices to debug
print("Training class indices:", train_generator.class_indices)
print("Validation class indices:", validation_generator.class_indices)

# Ensure the number of classes is consistent
if num_classes_train != num_classes_val:
    raise ValueError(f"Mismatch in number of classes. Training classes: {num_classes_train}, Validation classes: {num_classes_val}")

num_classes = num_classes_train  # Use the training number of classes

# Define the custom CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Use dynamically determined number of classes
])

# Compile the custom CNN model
cnn_model.compile(optimizer=Adam(learning_rate=0.0012456), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the custom CNN model architecture
cnn_model.summary()

# Train the custom CNN model
cnn_history = cnn_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained custom CNN model
cnn_model.save('custom_cnn_waste_segregate.keras')

# Evaluate the custom CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(validation_generator)
print(f"Custom CNN Validation Accuracy: {cnn_accuracy * 100:.2f}%")

# Define and train the VGG16-based model for transfer learning

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model
base_model.trainable = False

# Create new model on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# This is the model we will train using VGG16 as the base
vgg16_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the VGG16-based model
vgg16_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the VGG16-based model architecture
vgg16_model.summary()

# Train the VGG16-based model with early stopping
vgg16_history = vgg16_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Save the trained VGG16-based model
vgg16_model.save('vgg16_waste_segregate.keras')

# Evaluate the VGG16-based model
vgg16_loss, vgg16_accuracy = vgg16_model.evaluate(validation_generator)
print(f"VGG16 Model Validation Accuracy: {vgg16_accuracy * 100:.2f}%")
