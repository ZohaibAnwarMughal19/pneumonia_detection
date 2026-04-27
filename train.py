# ============================================
# Pneumonia Detection - train.py (IMPROVED)
# Better Accuracy Version
# ============================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, 
    Dropout, BatchNormalization
)
# BatchNormalization — training stable karta hai, accuracy badhata hai

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator — images load aur augment karne ke liye

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# EarlyStopping — jab improvement ruk jaye to training band kar do
# ReduceLROnPlateau — jab accuracy stuck ho to learning rate kam karo

import matplotlib.pyplot as plt
import os

# ============================================
# STEP 1: PATHS
# ============================================
BASE_DIR  = r"E:\python\ML\pnumonia\chest_xray\chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")
# os.path.join — sahi tarike se folder path banata hai

# ============================================
# STEP 2: SETTINGS
# ============================================
IMG_SIZE   = (150, 150)
# har image 150x150 pixels mein resize hogi

BATCH_SIZE = 32
# ek baar mein 32 images process hongi

EPOCHS     = 25
# pehle 10 the — ab 25 kar diye — zyada seekhega
# EarlyStopping rakha hai — agar improvement ruk gayi to khud band ho jayega

# ============================================
# STEP 3: DATA AUGMENTATION (Improved)
# ============================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # pixel values 0-255 se 0-1 range mein lao

    rotation_range=15,
    # pehle 10 tha — ab 15 degree tak rotate karega — zyada variety

    zoom_range=0.15,
    # pehle 0.1 tha — ab 15% zoom — aur variety

    horizontal_flip=True,
    # image ko mirror flip karo

    width_shift_range=0.1,
    # image ko thoda left/right shift karo — naya addition

    height_shift_range=0.1,
    # image ko thoda upar/neeche shift karo — naya addition

    shear_range=0.1,
    # image ko thoda teda karo — naya addition

    fill_mode='nearest'
    # shift ke baad khali jagah nearest pixel se bharo
)

val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# validation aur test ke liye sirf rescale — koi augmentation nahi

# ============================================
# STEP 4: DATA LOAD
# ============================================
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
    # binary — sirf 2 classes: NORMAL(0) aur PNEUMONIA(1)
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ============================================
# STEP 5: IMPROVED CNN MODEL
# ============================================
model = Sequential([

    # === BLOCK 1 ===
    Conv2D(32, (3,3), activation='relu', padding='same', 
           input_shape=(150, 150, 3)),
    # 32 filters se basic features dhundho (edges, lines)
    # padding='same' — image size same rahegi

    BatchNormalization(),
    # values normalize karo — training faster aur stable hoti hai

    Conv2D(32, (3,3), activation='relu', padding='same'),
    # same block mein ek aur conv layer — zyada features seekhe

    BatchNormalization(),
    MaxPooling2D(2, 2),
    # image size half karo — 150x150 → 75x75

    Dropout(0.25),
    # 25% neurons off karo — overfitting rokne ke liye

    # === BLOCK 2 ===
    Conv2D(64, (3,3), activation='relu', padding='same'),
    # 64 filters — complex features dhundho

    BatchNormalization(),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    # 75x75 → 37x37

    Dropout(0.25),

    # === BLOCK 3 ===
    Conv2D(128, (3,3), activation='relu', padding='same'),
    # 128 filters — high level features (lung patterns)

    BatchNormalization(),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    # 37x37 → 18x18

    Dropout(0.25),

    # === BLOCK 4 — NAYA ===
    Conv2D(256, (3,3), activation='relu', padding='same'),
    # 256 filters — bahut deep features — pneumonia specific patterns

    BatchNormalization(),
    MaxPooling2D(2, 2),
    # 18x18 → 9x9

    Dropout(0.25),

    # === DENSE LAYERS ===
    Flatten(),
    # 2D features ko 1D mein convert karo

    Dense(512, activation='relu'),
    # 512 neurons — final decision making

    BatchNormalization(),
    Dropout(0.5),
    # 50% neurons off — strong overfitting protection

    Dense(256, activation='relu'),
    # extra dense layer — pehle nahi tha

    Dropout(0.3),

    Dense(1, activation='sigmoid')
    # output layer — 0=NORMAL, 1=PNEUMONIA
])

# ============================================
# STEP 6: COMPILE
# ============================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # Adam optimizer — 0.001 se shuru karega
    # ReduceLROnPlateau baad mein automatically kam karega

    loss='binary_crossentropy',
    # binary classification ke liye standard loss function

    metrics=['accuracy']
)

model.summary()
# model ki layers aur parameters print karo

# ============================================
# STEP 7: CALLBACKS (Smart Training)
# ============================================
early_stop = EarlyStopping(
    monitor='val_loss',
    # val_loss dekho — agar improve nahi hua to band karo

    patience=5,
    # 5 epochs tak wait karo improvement ke liye

    restore_best_weights=True,
    # jab band ho to best wali weights wapas lao

    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    # val_loss stuck ho to learning rate kam karo

    factor=0.5,
    # learning rate ko half kar do

    patience=3,
    # 3 epochs baad reduce karo

    min_lr=0.00001,
    # minimum learning rate — is se kam nahi jayega

    verbose=1
)

# ============================================
# STEP 8: TRAIN
# ============================================
print("\n Improved Training shuru ho rahi hai... Sabar karo!\n")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    # maximum 25 epochs — EarlyStopping pehle band kar sakta hai

    validation_data=val_data,
    callbacks=[early_stop, reduce_lr]
    # dono callbacks use karo
)

# ============================================
# STEP 9: SAVE MODEL
# ============================================
model.save(r"E:\python\ML\pnumonia\model\pneumonia_model.h5")
print("\n Improved Model save ho gaya!\n")

# ============================================
# STEP 10: GRAPHS
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Improved Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Improved Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(r"E:\python\ML\pnumonia\training_results_improved.png")
plt.show()
print("\n Graph save ho gaya!\n")

# ============================================
# STEP 11: EVALUATE
# ============================================
print("\n Test data par evaluate kar raha hai...\n")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"\n Test Accuracy: {test_accuracy * 100:.2f}%")
print(f" Test Loss:     {test_loss:.4f}") 