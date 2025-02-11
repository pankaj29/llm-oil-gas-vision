#%% Import Libraries
import tensorflow as tf
from tensorflow.keras.applications import (
    DenseNet121, ResNet50, VGG16, InceptionV3, MobileNetV2, EfficientNetB0,
    NASNetMobile, Xception, InceptionResNetV2, ResNet152V2
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
import time
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import json
import numpy as np

# Enable mixed precision training for efficiency
tf.keras.mixed_precision.set_global_policy("mixed_float16")

#%% Paths to your datasets
train_csv = 'OGNetDevelopmentData/train.csv'
val_csv = 'OGNetDevelopmentData/val.csv'
test_csv = 'OGNetDevelopmentData/test.csv'
image_dir = 'OGNetDevelopmentData'

output_dir = 'results'  # Directory to save results
os.makedirs(output_dir, exist_ok=True)

#%% Load datasets
def load_datasets(train_csv, val_csv, test_csv, image_dir):
    train_df = pd.read_csv(train_csv).head(50)
    val_df = pd.read_csv(val_csv).head(10)
    test_df = pd.read_csv(test_csv).head(5)

    for df in [train_df, val_df, test_df]:
        df['Target'] = df['Target'].astype(str)
        df['Image_Path'] = df['Image_Path'].apply(lambda x: os.path.join(image_dir, x))
    return train_df, val_df, test_df

train_df, val_df, test_df = load_datasets(train_csv, val_csv, test_csv, image_dir)

#%% ImageDataGenerator for Data Augmentation
def create_generators(train_df, val_df, test_df, img_size=(500, 500), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.2
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='Image_Path',
        y_col='Target',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col='Image_Path',
        y_col='Target',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col='Image_Path',
        y_col='Target',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Keep order for predictions
    )
    return train_generator, val_generator, test_generator

train_generator, val_generator, test_generator = create_generators(train_df, val_df, test_df)

#%% List of Models to Compare
models_to_compare = [
    # ("DenseNet121", DenseNet121),
    # ("ResNet50", ResNet50),
    ("VGG16", VGG16),
    ("InceptionV3", InceptionV3),


    
    ("MobileNetV2", MobileNetV2),
    ("EfficientNetB0", EfficientNetB0),
    ("NASNetMobile", NASNetMobile),
    ("Xception", Xception),
    ("InceptionResNetV2", InceptionResNetV2),
    ("ResNet152V2", ResNet152V2)
]

#%% Helper Function for Model Building, Training, and Evaluation
def train_and_evaluate_model(base_model_class, model_name, input_shape, train_gen, val_gen, test_gen, output_dir):
    tracker = EmissionsTracker()
    tracker.start()
    start_time = time.time()
    
    # Load base model and add custom layers
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers and compile
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Fine-tune the model
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    fine_tuning_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)

    # Track runtime and emissions
    end_time = time.time()
    emissions = tracker.stop()

    # Return metrics
    return {
        'Model': model_name,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'Runtime (seconds)': end_time - start_time,
        'Carbon Emissions (kg CO2)': emissions
    }

#%% Train and Evaluate All Models
results = []
for model_name, model_class in models_to_compare:
    print(f"Training and evaluating {model_name}...")
    metrics = train_and_evaluate_model(
        model_class, model_name, (500, 500, 3),
        train_generator, val_generator, test_generator,
        output_dir
    )
    results.append(metrics)

#%% Save and Display Results
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(output_dir, 'model_comparison_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Comparison results saved to {results_csv_path}")
print(results_df)
#%%
