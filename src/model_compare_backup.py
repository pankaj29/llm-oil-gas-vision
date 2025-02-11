#%% Import Libraries#%% Import Libraries
#%% Import Libraries
#%% Import Libraries
import tensorflow as tf
from tensorflow.keras.applications import (
    DenseNet121, ResNet50, VGG16, InceptionV3, MobileNetV2, EfficientNetB0,
    NASNetMobile, Xception, InceptionResNetV2, ResNet152V2
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os
import time
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from kerastuner.tuners import RandomSearch
import logging

# Enable mixed precision training for efficiency
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Configure logging
logging.basicConfig(filename='training_log.log', level=logging.INFO)

# Paths to datasets
train_csv = 'Data/train.csv'
val_csv = 'Data/val.csv'
test_csv = 'Data/test.csv'
image_dir = 'Data'
img_size_x = 128
img_size_y = 128

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)
hyperparams_file = os.path.join(output_dir, 'best_hyperparams.csv')

# Load datasets without strict balancing to maximize data usage
def load_datasets(train_csv='Data/train.csv', val_csv='Data/val.csv', test_csv='Data/test.csv', image_dir='Data'):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    for df in [train_df, val_df, test_df]:
        df['Target'] = df['Target'].astype(str)
        df['Image_Path'] = df['Image_Path'].apply(lambda x: os.path.join(image_dir, x))

    print("Length of train_df= ", len(train_df))
    print("Length of val_df= ", len(val_df))
    print("Length of test_df= ", len(test_df))

    return train_df, val_df, test_df

train_df, val_df, test_df = load_datasets()

# Compute class weights for imbalanced data
unique_classes = np.unique(train_df['Target'])
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_df['Target'])
class_weights_dict = dict(zip(unique_classes, class_weights))

# ImageDataGenerator with extensive augmentation for training data
def create_generators(train_df, val_df, test_df, img_size=(img_size_x, img_size_y), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='Image_Path', y_col='Target', target_size=img_size,
        batch_size=batch_size, class_mode='binary', shuffle=True
    )
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df, x_col='Image_Path', y_col='Target', target_size=img_size,
        batch_size=batch_size, class_mode='binary'
    )
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df, x_col='Image_Path', y_col='Target', target_size=img_size,
        batch_size=batch_size, class_mode='binary', shuffle=False
    )
    return train_generator, val_generator, test_generator

train_generator, val_generator, test_generator = create_generators(train_df, val_df, test_df)

# List of Models to Compare
models_to_compare = [
    ("DenseNet121", DenseNet121),
    ("ResNet50", ResNet50),
    ("VGG16", VGG16),
    ("InceptionV3", InceptionV3),
    ("MobileNetV2", MobileNetV2),
    ("EfficientNetB0", EfficientNetB0),
    ("NASNetMobile", NASNetMobile),
    ("Xception", Xception),
    ("InceptionResNetV2", InceptionResNetV2),
    ("ResNet152V2", ResNet152V2)
]

# Helper Functions for Plotting
def plot_and_save(history, model_name, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_accuracy_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

# Build and Compile Model
def build_model(hp, base_model_class, input_shape):
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(hp.Float('dropout_rate', 0.3, 0.7))(x)
    x = Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = hp.Boolean('train_base', default=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train and Evaluate Models
def train_and_evaluate_model(base_model_class, model_name, input_shape, train_gen, val_gen, test_gen, output_dir):
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    if os.path.exists(hyperparams_file):
        hyperparams_df = pd.read_csv(hyperparams_file)
    else:
        hyperparams_df = pd.DataFrame(columns=['Model', 'dropout_rate', 'dense_units', 'train_base'])

    if not hyperparams_df.empty and model_name in hyperparams_df['Model'].values:
        best_params = hyperparams_df[hyperparams_df['Model'] == model_name].iloc[0]
        dropout_rate = best_params['dropout_rate']
        dense_units = best_params['dense_units']
        train_base = best_params['train_base']

        def build_best_model():
            base_model = base_model_class(weights='imagenet', include_top=False, input_shape=input_shape)
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(dropout_rate)(x)
            x = Dense(dense_units, activation='relu')(x)
            predictions = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            for layer in base_model.layers:
                layer.trainable = train_base

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model

        best_model = build_best_model()
    else:
        tuner = RandomSearch(
            lambda hp: build_model(hp, base_model_class, input_shape),
            objective='val_accuracy',
            max_trials=5, executions_per_trial=1,
            directory=model_output_dir, project_name=f'{model_name}_tuning'
        )

        tuner.search(train_gen, epochs=10, validation_data=val_gen)
        best_model = tuner.get_best_models(num_models=1)[0]

        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_params = {
            'Model': model_name,
            'dropout_rate': best_hp.get('dropout_rate'),
            'dense_units': best_hp.get('dense_units'),
            'train_base': best_hp.get('train_base')
        }

        hyperparams_df = pd.concat([hyperparams_df, pd.DataFrame([best_params])])
        hyperparams_df.to_csv(hyperparams_file, index=False)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=os.path.join(model_output_dir, f'{model_name}_checkpoint.h5'),
                                 save_best_only=True, monitor='val_loss', mode='min')
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    history = best_model.fit(
        train_gen, validation_data=val_gen,
        epochs=10, callbacks=[early_stopping, checkpoint, lr_scheduler],
        class_weight=class_weights_dict, verbose=1
    )

    test_loss, test_accuracy = best_model.evaluate(test_gen, verbose=1)
    predictions = best_model.predict(test_gen)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = np.array(test_gen.classes).astype(int)

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    plot_and_save(history, model_name, model_output_dir)
    plot_confusion_matrix(y_true, y_pred, model_name, model_output_dir)

    results = {
        'Model': model_name, 'Test Loss': test_loss,
        'Test Accuracy': test_accuracy, 'F1 Score': f1,
        'Precision': precision, 'Recall': recall
    }
    pd.DataFrame([results]).to_csv(os.path.join(model_output_dir, f'{model_name}_results.csv'), index=False)
    logging.info(f"Completed training and evaluation for {model_name}")

    tf.keras.backend.clear_session()

    return results

# Run Training for All Models
results = []
for model_name, model_class in models_to_compare:
    print(f"Training and evaluating {model_name}...")
    metrics = train_and_evaluate_model(
        model_class, model_name, (img_size_x, img_size_y, 3),
        train_generator, val_generator, test_generator, output_dir
    )
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'overall_model_comparison_results.csv'), index=False)
print(results_df)

#%%


#%%
