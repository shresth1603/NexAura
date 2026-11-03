"""
NexAura - MobileNetV2 Fine-Tuning Pipeline
==========================================
Enables Stage 2 Transfer Learning by unfreezing upper convolution blocks
and training with a fine-tuning learning rate (1e-5) with data augmentation.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def fine_tune_model(
    data_dir: str,
    base_model_path: str = "model/krishi_model.h5",
    output_model_path: str = "model/krishi_model_finetuned.h5",
    output_tflite_path: str = "model/krishi_model_finetuned.tflite",
    unfreeze_layers: int = 40,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    val_split: float = 0.2,
):
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks

    print("=" * 70)
    print("           NEXAURA: MOBILENETV2 FINE-TUNING PIPELINE")
    print("=" * 70)
    print(f"Dataset Directory  : {data_dir}")
    print(f"Base Model         : {base_model_path}")
    print(f"Output H5 Model    : {output_model_path}")
    print(f"Output TFLite Model: {output_tflite_path}")
    print(f"Unfrozen Layers    : Top {unfreeze_layers} layers")
    print(f"Learning Rate      : {learning_rate}")
    print(f"Epochs             : {epochs}")
    print("-" * 70)

    # 1. Load Data
    print("Loading image dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Discovered {num_classes} classes in dataset.")

    # Rescaling and Performance optimizations
    AUTOTUNE = tf.data.AUTOTUNE
    rescale_layer = layers.Rescaling(1.0 / 255.0)

    # Data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    train_ds = train_ds.map(lambda x, y: (data_augmentation(rescale_layer(x), training=True), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 2. Load Base Model
    print(f"\nLoading base model from {base_model_path}...")
    model = tf.keras.models.load_model(base_model_path)
    total_layers = len(model.layers)
    print(f"Base model loaded. Total layers: {total_layers}")

    # 3. Unfreeze Upper Layers for Fine-Tuning
    unfreeze_from = max(0, total_layers - unfreeze_layers)
    print(f"Unfreezing layers from index {unfreeze_from} to {total_layers}...")
    
    for i, layer in enumerate(model.layers):
        if i >= unfreeze_from:
            layer.trainable = True
        else:
            layer.trainable = False

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    print(f"Trainable parameters: {trainable_count:,} | Non-trainable: {non_trainable_count:,}")

    # 4. Compile with Fine-Tuning Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # 5. Callbacks
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 6. Train
    print("\nStarting fine-tuning...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
    )

    # 7. Export to TFLite
    print(f"\nExporting fine-tuned model to TFLite ({output_tflite_path})...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    tflite_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite export complete: {output_tflite_path} ({tflite_size_mb:.2f} MB)")
    print("=" * 70)
    print("Fine-tuning completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NexAura MobileNetV2 Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing class folders of images")
    parser.add_argument("--base_model", type=str, default="model/krishi_model.h5", help="Path to base .h5 model")
    parser.add_argument("--output_model", type=str, default="model/krishi_model_finetuned.h5", help="Path to output .h5 model")
    parser.add_argument("--output_tflite", type=str, default="model/krishi_model_finetuned.tflite", help="Path to output .tflite model")
    parser.add_argument("--unfreeze", type=int, default=40, help="Number of upper layers to unfreeze (default: 40)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")

    args = parser.parse_args()

    fine_tune_model(
        data_dir=args.data_dir,
        base_model_path=args.base_model,
        output_model_path=args.output_model,
        output_tflite_path=args.output_tflite,
        unfreeze_layers=args.unfreeze,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
