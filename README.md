# YtScraperS  
*Yorumları Anla, Markanı Yönet*

---

## Genel Bakış

**YtScraperS**, markaların YouTube platformundaki ürün ve hizmet yorumlarını derinlemesine analiz ederek müşteri algısını anlamalarına ve stratejilerini buna göre geliştirmelerine yardımcı olan Yapay Zeka destekli yenilikçi bir çözümdür. Gelişmiş doğal dil işleme teknikleri sayesinde milyonlarca yorumu tarar, olumlu ve olumsuz geri bildirimleri ayrıştırır ve anahtar çıkarımlar sunar.

Markalar, YtScraperS ile dijital itibarlarını, müşteri algılarını ve beklentilerini net bir şekilde görebilirler.

---

## Özellikler ve Fonksiyonlar

YtScraperS, markaların YouTube yorumlarından stratejik içgörüler elde etmesini sağlayan beş entegre modülden oluşmaktadır:

### 1. Veri Toplama Modülü  
- Anahtar kelime veya URL bazlı YouTube videolarını otomatik tespit eder.  
- İlgili video URL’lerini CSV formatında kaydeder.  
- Video başına yorum metinleri, tarihleri, kullanıcı adları ve meta veriler gelişmiş web scraping teknolojisiyle hızlıca toplanır.

### 2. Duygu Analizi Modülü  
- BERT tabanlı NLP modeli ile her yorum -1 (çok negatif) ile +1 (çok pozitif) arasında puanlanır.  
- "Pozitif" veya "Negatif" olarak etiketlenir.  
- Sonuçlar interaktif pasta grafikleri ve zaman çizelgesi grafiklerle görselleştirilir.

### 3. Akıllı Yorum Kategorizasyon Modülü  
- Gelişmiş Büyük Dil Modelleri (LLM) ile yorumları belirlenen kategorilere otomatik ve akıllı şekilde sınıflandırır.  
- Kategoriler bazında zaman grafikleri ve tablo gösterimleri sağlar.

### 4. Akıllı Raporlama Modülü  
- Tüm analizleri anlaşılır ve profesyonel raporlar halinde sunar.  
- Raporlar markaların stratejik karar almasını kolaylaştırır.

### 5. Kesintisiz Çalışma ve Veri Güvenliği  
- Veriler anlık olarak "state.json" dosyasına kaydedilir.  
- Elektrik kesintisi veya kapanma durumunda kaldığı yerden devam eder.  
- Harici veritabanı gerekmez; tüm veriler düzenli CSV dosyasında yerel olarak saklanır.

---

## Faydaları

- **Veri Odaklı Kararlar:** Ürün ve hizmet iyileştirmeleri için somut içgörüler sağlar.  
- **Dijital İtibar Yönetimi:** Marka algısını net biçimde görmenizi sağlar.  
- **Müşteri Sadakati ve Karlılık:** Sadakati artırarak rekabet avantajı sağlar.  
- **Gerçek Sesleri Duyma:** Geleneksel anketlerin ötesinde dijital ortamdan gerçek geri bildirimler sunar.

---

## İletişim
**Ürün Sahibi: Süleyman Kürşat DEMİR
**E-posta: demiresa38@gmail.com
**Telefon: 0506 036 5724
**LinkedIn: Süleyman Kürşat DEMİR

## Kurulum ve Kullanım

```bash
# 1. Depoyu klonlayın
git clone https://github.com/SuleymanKursatDemir/YtScraperS.git

# 2. Proje dizinine gidin
cd YtScraperS

# 3. Gerekli bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Uygulamayı çalıştırın
python main.py
