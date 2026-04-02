"""
hybrid_model.py — Improved for small datasets (140 images)
Colour feature branch is stronger — more reliable with small datasets.
CNN acts as a spatial supplement with L2 regularisation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_hybrid_model(color_feature_dim: int = 7,
                       image_shape: tuple = (128, 128, 3)) -> keras.Model:

    # ── Branch A: Lightweight CNN ─────────────────────────────────────────────
    img_input = keras.Input(shape=image_shape, name="image_input")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    cnn_features = layers.Dense(32, activation="relu")(x)

    # ── Branch B: Stronger Colour Feature pathway ─────────────────────────────
    color_input = keras.Input(shape=(color_feature_dim,), name="color_input")

    c = layers.Dense(64, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(color_input)
    c = layers.BatchNormalization()(c)
    c = layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(c)
    c = layers.BatchNormalization()(c)
    c = layers.Dense(64, activation="relu")(c)
    color_features = layers.BatchNormalization()(c)

    # ── Feature Fusion ────────────────────────────────────────────────────────
    fused = layers.Concatenate(name="feature_fusion")([cnn_features, color_features])

    # ── Regression head ───────────────────────────────────────────────────────
    y = layers.Dense(64, activation="relu")(fused)
    y = layers.Dropout(0.25)(y)
    y = layers.Dense(32, activation="relu")(y)
    y = layers.Dropout(0.1)(y)
    output = layers.Dense(1, name="spad_output")(y)

    model = keras.Model(
        inputs=[img_input, color_input],
        outputs=output,
        name="HybridCNN_SPAD_v2"
    )
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = 0.001) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae"]
    )
    return model