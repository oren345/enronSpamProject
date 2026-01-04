import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import nltk
import sys
import os

# =============================
# PATH AYARI (text_utils için)
# =============================
sys.path.append(os.path.abspath("../new_backend"))
from text_utils import preprocess_text

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =============================
# VERİYİ YÜKLE
# =============================
df = pd.read_csv("enron_spam.csv")

raw_texts = df["text"].astype(str)
labels = df["label"]

# =============================
# 1️⃣ ÖNCE–SONRA KELİME SAYISI
# =============================
raw_word_counts = raw_texts.apply(lambda x: len(x.split()))
processed_texts = raw_texts.apply(preprocess_text)
processed_word_counts = processed_texts.apply(lambda x: len(x.split()))

avg_raw = raw_word_counts.mean()
avg_processed = processed_word_counts.mean()

plt.figure()
plt.bar(["Raw Text", "Preprocessed Text"], [avg_raw, avg_processed])
plt.ylabel("Average Word Count")
plt.title("Average Word Count Before and After Preprocessing")
plt.show()

# =============================
# 2️⃣ STOP-WORD ORANI
# =============================
def count_stopwords(text):
    words = text.split()
    return sum(1 for w in words if w.lower() in stop_words), len(words)

raw_sw = raw_texts.apply(count_stopwords)
proc_sw = processed_texts.apply(count_stopwords)

raw_sw_ratio = sum(x[0] for x in raw_sw) / sum(x[1] for x in raw_sw)
proc_sw_ratio = sum(x[0] for x in proc_sw) / sum(x[1] for x in proc_sw)

plt.figure()
plt.bar(["Before Preprocessing", "After Preprocessing"],
        [raw_sw_ratio * 100, proc_sw_ratio * 100])
plt.ylabel("Stop-word Ratio (%)")
plt.title("Stop-word Ratio Before and After Preprocessing")
plt.show()

# =============================
# 3️⃣ EN SIK 20 KELİME (SONRASI)
# =============================
spam_words = []
ham_words = []

for text, label in zip(processed_texts, labels):
    if label == 1:
        spam_words.extend(text.split())
    else:
        ham_words.extend(text.split())

spam_common = Counter(spam_words).most_common(20)
ham_common = Counter(ham_words).most_common(20)

# Spam
plt.figure(figsize=(10,4))
plt.bar([w for w, _ in spam_common], [c for _, c in spam_common])
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words (Spam Emails)")
plt.ylabel("Frequency")
plt.show()

# Ham
plt.figure(figsize=(10,4))
plt.bar([w for w, _ in ham_common], [c for _, c in ham_common])
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words (Ham Emails)")
plt.ylabel("Frequency")
plt.show()
