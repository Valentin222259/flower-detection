import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import pathlib

# Setari
DATASET_PATH = r"C:\Users\Victus\Downloads\archive\flowers"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Incarca dataset-ul
data_dir = pathlib.Path(DATASET_PATH)

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Numele claselor
class_names = train_ds.class_names
print("Clase detectate:", class_names)

# Optimizare performanta
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Model MobileNetV2 pre-antrenat
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Construieste modelul
model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

# Compileaza
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Antreneaza
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Salveaza modelul
model.save('flower_model.keras')
print("Model salvat!")

# Grafic acuratete
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Acuratete model')
plt.savefig('training_results.png')
plt.show()
print("Grafic salvat!")