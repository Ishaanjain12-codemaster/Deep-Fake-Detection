# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
# import keras_cv

# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt

# device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
# print("Using device:", device)

# def extract_frames_from_videos(videos_dir, output_dir, label, max_videos=50):
#     if not os.path.exists(videos_dir):
#         print(f"Directory not found: {videos_dir}")
#         return

#     video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
#     video_files = video_files[:max_videos]

#     for video_file in video_files:
#         video_path = os.path.join(videos_dir, video_file)
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             continue
            
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         if fps <= 0: fps = 25 # Fallback
        
#         frame_count = 0
#         success, image = cap.read()

#         while success:
#             if frame_count % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
#                 frame_filename = f"{label}_{video_file}_frame{frame_count // int(cap.get(cv2.CAP_PROP_FPS))}.jpg"
#                 frame_path = os.path.join(output_dir, frame_filename)
#                 cv2.imwrite(frame_path, image)
#             success, image = cap.read()
#             frame_count += 1

#         cap.release()

# real_videos_dir = "dataset/Real"
# manipulated_videos_dir = "dataset/Fake"
# output_real_dir = "frames_dataset/real"
# output_manipulated_dir = "frames_dataset/manipulated"

# os.makedirs(output_real_dir, exist_ok=True)
# os.makedirs(output_manipulated_dir, exist_ok=True)

# extract_frames_from_videos(real_videos_dir, output_real_dir, "real", max_videos=10)
# extract_frames_from_videos(manipulated_videos_dir, output_manipulated_dir, "manipulated", max_videos=10)
# print("Frame extraction completed.")

# IMG_SIZE = (224, 224)

# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomContrast(0.2),
# ])

# imagenet_mean = tf.constant([0.485, 0.456, 0.406])
# imagenet_std  = tf.constant([0.229, 0.224, 0.225])

# def preprocess_image(image, label, training=True):
#     image = tf.cast(image, tf.float32) / 255.0
#     if training:
#         image = data_augmentation(image)
#     image = (image - imagenet_mean) / imagenet_std
#     return image, label

# BATCH_SIZE = 8
# SEED = 42
# dataset_dir = "frames_dataset"
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     dataset_dir,
#     label_mode="int",
#     validation_split=0.2,
#     subset="training",
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     dataset_dir,
#     label_mode="int",
#     validation_split=0.2,
#     subset="validation",
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.map(lambda x, y: preprocess_image(x, y, True)).prefetch(AUTOTUNE)
# val_ds   = val_ds.map(lambda x, y: preprocess_image(x, y, False)).prefetch(AUTOTUNE)

# with tf.device(device):
#     model = keras_cv.models.ImageClassifier.from_preset(
#         "tf.keras.applications.vit.ViTSmall16", # Using a valid preset name, vit_l_16 might need specific naming like vit_l_16_imagenet1k
#         num_classes=2,
#     )

# model.summary()

# model.compile(
#     optimizer=tf.keras.optimizers.AdamW(
#         learning_rate=1e-5,
#         weight_decay=1e-4
#     ),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"]
# )

# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch, lr: lr * 0.7 if (epoch + 1) % 5 == 0 else lr
# )

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     patience=5,
#     restore_best_weights=True
# )

# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     "best_vit_model_tf",
#     monitor="val_accuracy",
#     save_best_only=True,
#     save_weights_only=True
# )

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=20,
#     callbacks=[early_stopping, checkpoint, lr_scheduler]
# )

# model.load_weights("best_vit_model_tf")

# from collections import defaultdict
# from sklearn.metrics import confusion_matrix, classification_report
# import cv2

# def get_video_id(filename):
#     return filename.split("_frame")[0]

# def preprocess_single_image(img):
#     img = tf.image.resize(img, IMG_SIZE)
#     img = tf.cast(img, tf.float32) / 255.0
#     img = (img - imagenet_mean) / imagenet_std
#     return img

# def predict_video_mean_pool(model, frames):
#     logits = model(frames, training=False)   # (N, 2)
#     video_logits = tf.reduce_mean(logits, axis=0)
#     return tf.argmax(video_logits).numpy()

# video_preds = []
# video_labels = []

# for class_name, class_label in [("real", 0), ("manipulated", 1)]:
#     frame_dir = f"Dataset/{class_name}" #Don't change this

#     video_groups = defaultdict(list)
#     for fname in os.listdir(frame_dir):
#         vid = get_video_id(fname)
#         video_groups[vid].append(fname)

#     for vid, frame_files in video_groups.items():
#         frames = []

#         for f in frame_files:
#             img = cv2.imread(os.path.join(frame_dir, f))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = preprocess_single_image(img)
#             frames.append(img)

#         if len(frames) == 0:
#             continue

#         frames = tf.stack(frames)   # (N, 224, 224, 3)
#         pred = predict_video_mean_pool(model, frames)

#         video_preds.append(pred)
#         video_labels.append(class_label)
# print("VIDEO-LEVEL Classification Report")
# print(classification_report(
#     video_labels,
#     video_preds,
#     target_names=["Real", "Manipulated"]
# ))

# cm = confusion_matrix(video_labels, video_preds)
# print("Confusion Matrix:\n", cm)

# all_labels = []
# all_preds = []

# for images, labels in val_ds:
#     logits = model(images, training=False)
#     preds = tf.argmax(logits, axis=1)

#     all_labels.extend(labels.numpy())
#     all_preds.extend(preds.numpy())

# cm = confusion_matrix(all_labels, all_preds)

# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Real", "Manipulated"],
#     yticklabels=["Real", "Manipulated"]
# )
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()

# print("Classification Report:")
# print(
#     classification_report(
#         all_labels,
#         all_preds,
#         target_names=["Real", "Manipulated"]
#     )
# )

# acc = accuracy_score(all_labels, all_preds)
# print(f"Accuracy: {acc * 100:.2f}%")

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
# import keras_cv
# from vit_keras import vit
# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)

# # =============================
# # DEVICE (APPLE METAL GPU)
# # =============================
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

# device = "/GPU:0" if gpus else "/CPU:0"
# print("Using device:", device)

# # =============================
# # CONFIG
# # =============================
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 8
# SEQ_LEN = 8
# EPOCHS = 10
# SEED = 42
# dataset_dir = "frames_dataset"

# # =============================
# # DATA AUGMENTATION
# # =============================
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomContrast(0.2),
# ])

# def preprocess_image(image, label, training=True):
#     image = tf.cast(image, tf.float32) / 255.0
#     if training:
#         image = data_augmentation(image)
#     return image, label

# # =============================
# # DATASET
# # =============================
# # train_ds = tf.keras.utils.image_dataset_from_directory(
# #     dataset_dir,
# #     label_mode="int",
# #     validation_split=0.2,
# #     subset="training",
# #     seed=SEED,
# #     image_size=IMG_SIZE,
# #     batch_size=BATCH_SIZE
# # )

# # val_ds = tf.keras.utils.image_dataset_from_directory(
# #     dataset_dir,
# #     label_mode="int",
# #     validation_split=0.2,
# #     subset="validation",
# #     seed=SEED,
# #     image_size=IMG_SIZE,
# #     batch_size=BATCH_SIZE
# # )

# AUTOTUNE = tf.data.AUTOTUNE
# # train_ds = train_ds.map(lambda x, y: preprocess_image(x, y, True)).prefetch(AUTOTUNE)
# # val_ds   = val_ds.map(lambda x, y: preprocess_image(x, y, False)).prefetch(AUTOTUNE)

# def build_video_sequences(dataset_dir, split="training", val_ratio=0.2):
#     videos = []
#     labels = []

#     class_map = {"real": 0, "manipulated": 1}

#     for class_name, class_label in class_map.items():
#         frame_dir = os.path.join(dataset_dir, class_name)
#         if not os.path.exists(frame_dir):
#             continue

#         video_groups = {}
#         for fname in os.listdir(frame_dir):
#             if "_frame" not in fname:
#                 continue
#             vid = fname.split("_frame")[0]
#             video_groups.setdefault(vid, []).append(fname)

#         video_ids = sorted(video_groups.keys())
#         split_idx = int(len(video_ids) * (1 - val_ratio))

#         if split == "training":
#             video_ids = video_ids[:split_idx]
#         else:
#             video_ids = video_ids[split_idx:]

#         for vid in video_ids:
#             frame_files = sorted(video_groups[vid])[:SEQ_LEN]
#             frames = []

#             for f in frame_files:
#                 img = cv2.imread(os.path.join(frame_dir, f))
#                 if img is None:
#                     continue
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 img = tf.image.resize(img, IMG_SIZE)
#                 img = tf.cast(img, tf.float32) / 255.0
#                 frames.append(img)

#             if len(frames) < SEQ_LEN:
#                 continue

#             videos.append(tf.stack(frames))
#             labels.append(class_label)

#     return tf.data.Dataset.from_tensor_slices((videos, labels))

# train_ds = build_video_sequences(dataset_dir, split="training")
# val_ds   = build_video_sequences(dataset_dir, split="validation")

# train_ds = train_ds.shuffle(128).batch(BATCH_SIZE).prefetch(AUTOTUNE)
# val_ds   = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# # =============================
# # MODEL (LIGHTWEIGHT ViT)
# # =============================
# # with tf.device(device):
# #     # backbone = keras_cv.models.VisionTransformer.from_preset("vit_b16_fe", load_weights=True) # Preset not found in installed KerasCV
# #     backbone = vit.vit_b16(
# #         image_size=IMG_SIZE[0],
# #         activation="linear",
# #         pretrained=True,
# #         include_top=False,
# #         pretrained_top=False
# #     )
# #     backbone.trainable = False

# #     inputs = layers.Input(shape=(*IMG_SIZE, 3))
# #     x = backbone(inputs)
# #     x = layers.Dense(256, activation="relu")(x)
# #     x = layers.Dropout(0.5)(x)
# #     outputs = layers.Dense(2)(x)  # logits

# #     model = tf.keras.Model(inputs, outputs)

# # model.summary()

# with tf.device(device):
#     vit_backbone = vit.vit_b16(
#         image_size=IMG_SIZE[0],
#         activation="linear",
#         pretrained=True,
#         include_top=False,
#         pretrained_top=False
#     )
#     vit_backbone.trainable = False

#     frame_input = layers.Input(shape=(*IMG_SIZE, 3))
#     frame_features = vit_backbone(frame_input)
#     frame_encoder = tf.keras.Model(frame_input, frame_features)

#     video_input = layers.Input(shape=(SEQ_LEN, *IMG_SIZE, 3))

#     x = layers.TimeDistributed(frame_encoder)(video_input)

#     mean_pool = tf.reduce_mean(x, axis=1)
#     max_pool  = tf.reduce_max(x, axis=1)

#     x = layers.Concatenate()([mean_pool, max_pool])
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(2)(x)

#     model = tf.keras.Model(video_input, outputs)

# model.summary()

# # =============================
# # COMPILE
# # =============================
# model.compile(
#     optimizer=tf.keras.optimizers.AdamW(
#         learning_rate=1e-4,
#         weight_decay=1e-4
#     ),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"]
# )

# # =============================
# # CALLBACKS
# # =============================
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor="val_accuracy",
#         patience=10,
#         restore_best_weights=True
#     ),
#     tf.keras.callbacks.ModelCheckpoint(
#         "best_vit_tf.keras",
#         monitor="val_accuracy",
#         save_best_only=True
#     )
# ]

# # =============================
# # TRAIN
# # =============================
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS,
#     callbacks=callbacks,
#     verbose=1
# )

# # =============================
# # EVALUATION & PREDICTION
# # =============================
# model.load_weights("best_vit_tf.keras")

# from collections import defaultdict
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import cv2
# import seaborn as sns
# import matplotlib.pyplot as plt

# def get_video_id(filename):
#     # filename format: label_videoname_frameX.jpg
#     # We want valid_videoname
#     # Adjust based on how you construct filenames in extract_frames
#     # e.g. "real_video.mp4_frame0.jpg" -> "video.mp4"
#     # Current code: f"{label}_{video_file}_frame..."
#     # So split by "_frame" gives "label_video.mp4" which is unique enough per video?
#     # Or just use the whole prefix.
#     return filename.split("_frame")[0]

# def preprocess_single_image(img):
#     img = tf.image.resize(img, IMG_SIZE)
#     img = tf.cast(img, tf.float32) / 255.0
#     return img

# def predict_video_pool(model, frames):
#     frames = frames[:SEQ_LEN]
#     frames = tf.expand_dims(frames, axis=0)  # (1, T, H, W, C)

#     logits = model(frames, training=False)
#     probs = tf.nn.softmax(logits, axis=1)

#     pred = tf.argmax(probs, axis=1).numpy()[0]
#     confidence = probs[0, pred].numpy()

#     return pred, confidence

# print("\nRunning Video-Level Evaluation...")
# video_preds = []
# video_labels = []
# video_confidences = []

# for class_name, class_label in [("real", 0), ("manipulated", 1)]:
#     frame_dir = f"frames_dataset/{class_name}" 
#     if not os.path.exists(frame_dir):
#         print(f"Skipping {frame_dir}, does not exist.")
#         continue

#     video_groups = defaultdict(list)
#     for fname in os.listdir(frame_dir):
#         if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
#             continue
#         vid = get_video_id(fname)
#         video_groups[vid].append(fname)

#     for vid, frame_files in video_groups.items():
#         frames = []

#         for f in frame_files:
#             img = cv2.imread(os.path.join(frame_dir, f))
#             if img is None: continue
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = preprocess_single_image(img)
#             frames.append(img)

#         if len(frames) == 0:
#             continue

#         frames = tf.stack(frames)   # (N, 224, 224, 3)
#         pred, conf = predict_video_pool(model, frames)

#         video_preds.append(pred)
#         video_labels.append(class_label)
#         video_confidences.append(conf)
        
#         label_str = "Real" if pred == 0 else "Manipulated"
#         print(f"Video: {vid} | Prediction: {label_str} | Confidence: {conf:.4f}")

# print("\nVIDEO-LEVEL Classification Report")
# print(classification_report(
#     video_labels,
#     video_preds,
#     target_names=["Real", "Manipulated"]
# ))

# acc = accuracy_score(video_labels, video_preds)
# print(f"Video-Level Accuracy: {acc * 100:.2f}%")

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from vit_keras import vit
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================
# DEVICE
# =============================
gpus = tf.config.list_physical_devices("GPU")
device = "/GPU:0" if gpus else "/CPU:0"
print("Using device:", device)

# =============================
# CONFIG
# =============================
IMG_SIZE = (224, 224)
SEQ_LEN = 8
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42
DATASET_DIR = "frames_dataset"

AUTOTUNE = tf.data.AUTOTUNE

# =============================
# FRAME SAMPLING (CRITICAL)
# =============================
def sample_frames(frame_list, seq_len):
    if len(frame_list) >= seq_len:
        idx = np.linspace(0, len(frame_list) - 1, seq_len).astype(int)
        return [frame_list[i] for i in idx]
    else:
        return frame_list + [frame_list[-1]] * (seq_len - len(frame_list))

# =============================
# DATASET BUILDER (VIDEO-WISE)
# =============================
def build_video_sequences(dataset_dir, split="training", val_ratio=0.2):
    videos, labels = [], []
    class_map = {"real": 0, "manipulated": 1}

    for class_name, class_label in class_map.items():
        frame_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(frame_dir):
            continue

        video_groups = defaultdict(list)
        for fname in os.listdir(frame_dir):
            if "_frame" in fname:
                vid = fname.split("_frame")[0]
                video_groups[vid].append(fname)

        video_ids = sorted(video_groups.keys())
        split_idx = int(len(video_ids) * (1 - val_ratio))

        if split == "training":
            video_ids = video_ids[:split_idx]
        else:
            video_ids = video_ids[split_idx:]

        for vid in video_ids:
            frame_files = sample_frames(sorted(video_groups[vid]), SEQ_LEN)
            frames = []

            for f in frame_files:
                img = cv2.imread(os.path.join(frame_dir, f))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = tf.image.resize(img, IMG_SIZE)
                img = tf.cast(img, tf.float32) / 255.0
                frames.append(img)

            if len(frames) == SEQ_LEN:
                videos.append(tf.stack(frames))
                labels.append(class_label)

    return tf.data.Dataset.from_tensor_slices((videos, labels))

train_ds = build_video_sequences(DATASET_DIR, "training")
val_ds   = build_video_sequences(DATASET_DIR, "validation")

train_ds = train_ds.shuffle(128).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds   = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# =============================
# MODEL (ViT + TEMPORAL POOLING)
# =============================
with tf.device(device):
    vit_backbone = vit.vit_b16(
        image_size=IMG_SIZE[0],
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        activation="linear",
    )
    vit_backbone.trainable = False

    frame_input = layers.Input(shape=(*IMG_SIZE, 3))
    frame_features = vit_backbone(frame_input)
    frame_encoder = tf.keras.Model(frame_input, frame_features)

    video_input = layers.Input(shape=(SEQ_LEN, *IMG_SIZE, 3))
    x = layers.TimeDistributed(frame_encoder)(video_input)

    mean_pool = tf.reduce_mean(x, axis=1)
    max_pool  = tf.reduce_max(x, axis=1)

    x = layers.Concatenate()([mean_pool, max_pool])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(video_input, outputs)

model.summary()

# =============================
# COMPILE
# =============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =============================
# CLASS WEIGHTS (IMPORTANT)
# =============================
num_real = 400
num_fake = 3000

class_weights = {
    0: num_fake / num_real,
    1: 1.0
}

# =============================
# TRAIN
# =============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=1
)

model.save("vit_temporal_pooling_model")

# =============================
# VIDEO-LEVEL EVALUATION
# =============================
def preprocess_frame(img):
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def predict_video(model, frames):
    frames = sample_frames(frames, SEQ_LEN)
    frames = tf.stack(frames)
    frames = tf.expand_dims(frames, axis=0)

    prob = model(frames, training=False)[0][0].numpy()
    label = 1 if prob > 0.5 else 0
    confidence = prob if label == 1 else 1 - prob
    return label, confidence

print("\nRunning Video-Level Evaluation...")

video_preds, video_labels = [], []

for class_name, class_label in [("real", 0), ("manipulated", 1)]:
    frame_dir = os.path.join(DATASET_DIR, class_name)
    video_groups = defaultdict(list)

    for fname in os.listdir(frame_dir):
        if "_frame" in fname:
            vid = fname.split("_frame")[0]
            video_groups[vid].append(fname)

    for vid, frame_files in video_groups.items():
        frames = []
        for f in frame_files:
            img = cv2.imread(os.path.join(frame_dir, f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_frame(img)
            frames.append(img)

        if len(frames) == 0:
            continue

        pred, conf = predict_video(model, frames)
        video_preds.append(pred)
        video_labels.append(class_label)

        label_str = "Real" if pred == 0 else "Manipulated"
        print(f"{vid} → {label_str} ({conf:.3f})")

print("\nVIDEO-LEVEL CLASSIFICATION REPORT")
print(classification_report(video_labels, video_preds, target_names=["Real", "Manipulated"]))

acc = accuracy_score(video_labels, video_preds)
print(f"\nVideo-Level Accuracy: {acc * 100:.2f}%")