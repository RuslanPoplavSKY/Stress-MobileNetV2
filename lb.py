import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# Шляхи до папок
train_path = r"C:\Users\\Administrator\\Desktop\\Diplomna\\facesData\\train"
valid_path = r"C:\Users\\Administrator\\Desktop\\Diplomna\\facesData\\test"
expression = 'nostress'

# Налаштування
picture_size = 224
no_of_classes = 2
batch_size = 64
epochs = 30

# Візуалізація зображень
plt.style.use('dark_background')
plt.figure(figsize=(12, 12))
for i in range(1, 10):
    plt.subplot(3, 3, i)
    img = load_img(os.path.join(valid_path, expression, os.listdir(os.path.join(valid_path, expression))[i]),
                   target_size=(picture_size, picture_size))
    plt.imshow(img)
plt.show()

# Аугментація
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(picture_size, picture_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(picture_size, picture_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Базова модель
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(picture_size, picture_size, 3), alpha=1.0)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Побудова моделі
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(no_of_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Компіляція
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

model.summary()

# Колбеки
checkpoint = ModelCheckpoint("model_mobilenet.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# Навчання без ваг класів
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // batch_size,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Візуалізація графіків
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()

# --- Додатково: Обчислення матриці плутанини, precision, recall, f1-score ---

# Оновлюємо генератор для початку з початку
valid_generator.reset()

# Отримуємо передбачення ймовірностей на всіх валідаційних даних
y_pred_prob = model.predict(valid_generator, steps=valid_generator.n // batch_size + 1)

# Визначаємо передбачені класи
y_pred = np.argmax(y_pred_prob, axis=1)

# Реальні мітки
y_true = valid_generator.classes

# Матриця плутанини
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Класіфікаційний звіт із precision, recall, f1-score
report = classification_report(y_true, y_pred, target_names=valid_generator.class_indices.keys())
print("Classification Report:")
print(report)
