📧 Enron Spam E-Posta Tespiti ve Web Arayüzü Geliştirme Projesi

Bu proje, Enron e-posta veri seti kullanılarak derin öğrenme tabanlı bir spam e-posta tespit sistemi geliştirilmesini ve bu sistemin Flask tabanlı bir web arayüzü ile son kullanıcıya sunulmasını amaçlamaktadır.

Proje, Yazılımda Siber Güvenlik dersi kapsamında yüksek lisans vize projesi olarak hazırlanmıştır.

🎯 Projenin Amacı

Bu çalışmanın temel amaçları şunlardır:

Spam ve ham (normal) e-postaların otomatik olarak sınıflandırılması

DNN + Bidirectional LSTM (BiLSTM) mimarisi kullanarak metin tabanlı saldırıların tespiti

Eğitimde kullanılan ön işleme adımlarıyla tam uyumlu bir tahmin servisi geliştirilmesi

Model çıktılarının web arayüzü üzerinden kullanıcıya sunulması

Siber güvenlik kapsamında e-posta tabanlı tehditlere karşı bir savunma mekanizması oluşturulması

🧠 Kullanılan Yöntemler ve Teknolojiler
🔹 Makine Öğrenmesi & Derin Öğrenme

Derin Sinir Ağları (DNN)

Bidirectional LSTM (BiLSTM)

Sigmoid aktivasyonlu ikili sınıflandırma

Class Weight kullanımı (dengesiz veri problemi için)

Early Stopping ile overfitting önleme

🔹 Metin Ön İşleme

Küçük harfe dönüştürme

HTML etiket temizleme

Özel karakter ve sayıların kaldırılması

Stop-word çıkarımı

Lemmatization

🔹 Teknolojiler

Python

TensorFlow / Keras

Scikit-learn

NLTK

Flask

HTML / CSS (frontend)

📂 Proje Dosya Yapısı
enronSpamProject
│
├── train.py                # Model eğitim scripti
├── README.md               # Proje açıklaması
│
├── backend
│   ├── app.py              # Flask backend uygulaması
│   └── text_utils.py       # Metin ön işleme yardımcı fonksiyonları
│
├── frontend
│   └── index.html          # Web arayüzü
│
├── src
│   └── preprocess.py       # Eğitim ve tahmin için ortak ön işleme adımları


⚠️ Not:

Veri seti (.csv) ve eğitilmiş model dosyaları (.keras) bilinçli olarak GitHub’a eklenmemiştir.

Bunun nedeni dosya boyutu ve akademik kullanım kısıtlarıdır.

📊 Kullanılan Veri Seti

Enron Spam Dataset

Kamuya açık, akademik çalışmalarda yaygın olarak kullanılan bir veri setidir.

Spam filtreleme literatüründe referans niteliğindedir.

Kaynak:
Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). Spam filtering with the Enron email dataset.

🌐 Web Arayüzü Çalışma Mantığı

Kullanıcı, e-posta metnini web arayüzüne girer

Metin, Flask backend servisine JSON formatında gönderilir

Backend tarafında:

Eğitim sürecindekiyle aynı ön işleme adımları uygulanır

Eğitilmiş DNN–BiLSTM modele girdi verilir

Model:

Spam olasılığı üretir

Belirlenen eşiklere göre Spam / Ham / Belirsiz olarak sınıflandırır

Sonuç kullanıcıya arayüz üzerinden gösterilir

⚙️ Kurulum ve Çalıştırma (Özet)
pip install -r requirements.txt
python train.py
python backend/app.py
Frontend dosyası (index.html) tarayıcı üzerinden çalıştırılabilir.

📌 Akademik Not

Bu proje eğitsel ve akademik amaçlıdır.
Ticari kullanım hedeflenmemektedir.

👩‍💻 Hazırlayan

Ayşe Nur Ören
Yüksek Lisans Öğrencisi
Yazılım Mühendisliği Anabilim Dalı
Turgut Özal Üniversitesi

👩‍🏫 Ders Bilgisi

Ders: Yazılımda Siber Güvenlik
Öğretim Üyesi: Doç. Dr. Canan Batur Şahin
Yıl: 2026
