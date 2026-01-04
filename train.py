import os
import re
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt   # 🔥 EKLENDİ
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

# =============================
# NLTK
# =============================
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =============================
# TEXT PREPROCESSING
# =============================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("data/enron_spam.csv")

df["clean_text"] = df["text"].apply(preprocess)

X = df["clean_text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =============================
# CLASS WEIGHTS (ÇOK KRİTİK)
# =============================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# =============================
# TEXT VECTORIZATION (MODELE GÖMÜLÜ)
# =============================
vectorizer = TextVectorization(
    max_tokens=5000,
    output_sequence_length=200
)
vectorizer.adapt(X_train)

# =============================
# DNN + BiLSTM MODEL
# =============================
model = Sequential([
    vectorizer,
    Embedding(input_dim=5000, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# CALLBACK (OVERFITTING ÖNLEME)
# =============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =============================
# TRAIN
# =============================
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[early_stop]
)
y_prob = model.predict(X_test).ravel()

thresholds = [0.3, 0.5, 0.7]

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    print(f"Threshold: {t}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("-"*30)
# =============================
# 📊 TEST SETİ ÜZERİNDE DEĞERLENDİRME
# =============================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns

# Test seti üzerinde tahmin
y_test_prob = model.predict(X_test, verbose=0)
y_test_pred = (y_test_prob >= 0.5).astype(int)

# =============================
# 📌 Performans Metrikleri
# =============================
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("📊 Test Performans Sonuçları")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Ham", "Spam"]))

# =============================
# 📊 CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Confusion Matrix – Test Verisi")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")
plt.show()


# =============================
# SAVE MODEL
# =============================
os.makedirs("model", exist_ok=True)
model.save("model/enron_dnn_bilstm.keras")

print("✅ Model başarıyla eğitildi, kaydedildi ve grafikler oluşturuldu.")
