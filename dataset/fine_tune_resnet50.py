import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Directories
dataset_dir = "/Users/aurorekouakou/image_recognition/organized_dataset"
model_save_path = "models/fine_tuned_resnet50.keras"

# Hyperparameters
img_size = 224
batch_size = 16
epochs = 10  # Nombre d'époques d'entraînement

# Callbacks pour surveiller et sauvegarder le modèle
checkpoint = ModelCheckpoint(
    model_save_path, monitor='val_loss', save_best_only=True, verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
)

# Data augmentation et génération
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% pour training, 20% pour validation
)

train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Charger le modèle ResNet50 pré-entraîné
base_model = ResNet50(weights="imagenet", include_top=False)

# Geler les couches du modèle de base
for layer in base_model.layers:
    layer.trainable = False

# Ajouter des couches personnalisées
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(train_data.num_classes, activation="softmax")(x)

# Créer le modèle final
model = Model(inputs=base_model.input, outputs=predictions)

# Compiler le modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Entraînement du modèle
print("Training the custom model...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Débloquer certaines couches du modèle et fine-tune
print("Fine-tuning the model...")
for layer in base_model.layers[-10:]:  # Débloquer les 10 dernières couches
    layer.trainable = True

# Recompiler le modèle avec un taux d'apprentissage plus faible
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Reprendre l'entraînement pour fine-tuning
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

model.save(model_save_path)
print(f"Model saved to {model_save_path}")
