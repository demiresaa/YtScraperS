# bert_utils.py - DÜZENLENMİŞ HALİ (Tekrar kontrol amaçlı)
import pandas as pd
from transformers import pipeline
import re # Dosya adı temizleme için eklendi (bu dosyada kullanılmasa da main.py için gerekliydi)
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np # np.nan veya np.NaT için

class SentimentAnalysisError(Exception):
    """Duygu analizi sırasında oluşacak özel hata sınıfı."""
    pass

def convert_relative_time(relative_time_str):
    """
    'X [birim] önce' formatındaki stringi, bugünün tarihine göre mutlak bir tarihe dönüştürür.
    Birimler: 'gün', 'ay', 'yıl'.
    """
    if not isinstance(relative_time_str, str):
        # NaN veya diğer string olmayan girdileri atla
        return np.nan # Pandas'ta zaman için uygun boş değer NaT'dir, np.nan kullanmak da çalışır

    parts = relative_time_str.lower().split()

    # Beklenen formatı kontrol et: [sayı] [birim] önce
    if len(parts) < 3 or parts[-1] != 'önce':
        print(f"Uyarı: Beklenmeyen format: '{relative_time_str}'. Atlanıyor.")
        return np.nan # Geçersiz formatlar için NaT dön

    try:
        number = int(parts[0])
        unit = parts[1]
        today = datetime.date.today() # Bugünün tarihini al

        if unit == 'gün':
            # Günler için timedelta kullan
            calculated_date = today - datetime.timedelta(days=number)
        elif unit == 'ay':
            # Aylar için relativedelta kullan (ay sonlarını, yıl geçişlerini doğru yönetir)
            calculated_date = today - relativedelta(months=number)
        elif unit == 'yıl':
            # Yıllar için relativedelta kullan (artık yılları, yıl geçişlerini doğru yönetir)
            calculated_date = today - relativedelta(years=number)
        elif unit == 'hafta':
            # Haftalar için timedelta kullan
            calculated_date = today - datetime.timedelta(weeks=number)
        elif unit == 'saat':
            # Saatler için timedelta kullan
            calculated_date = today - datetime.timedelta(hours=number)
        elif unit == 'dakika':
            # Dakikalar için timedelta kullan
            calculated_date = today - datetime.timedelta(minutes=number)
        else:
            # Bilinmeyen birimler
            print(f"Uyarı: Bilinmeyen birim '{unit}' in '{relative_time_str}'. Atlanıyor.")
            return np.nan

        return calculated_date

    except ValueError:
        # Sayı dönüştürülemezse
        print(f"Uyarı: Sayı parse edilemedi: '{relative_time_str}'. Atlanıyor.")
        return np.nan
    except Exception as e:
        # Diğer beklenmeyen hatalar
        print(f"Hata oluştu: '{relative_time_str}' işlenirken: {e}. Atlanıyor.")
        return np.nan



def perform_sentiment_analysis(csv_input_path, text_column_name, csv_output_path, ):
    """
    Verilen CSV dosyasındaki metin sütununu kullanarak BERT ile duygu analizi yapar,
    sonuçları orijinal DataFrame'e ekler ve yeni bir CSV'ye kaydeder.

    Args:
        csv_input_path (str): Yorumları içeren giriş CSV dosyasının yolu.
        text_column_name (str): Yorum metinlerinin bulunduğu sütunun adı.
        csv_output_path (str): Analiz sonuçlarının kaydedileceği çıkış CSV dosyasının yolu.

    Returns:
        tuple: (pozitif_yorum_sayisi, negatif_yorum_sayisi)
               Hata durumunda exception fırlatır.
    """
    # --- Model Yükleme ---
    try:
        print("Duygu analizi modeli yükleniyor...")
        model_name = "savasy/bert-base-turkish-sentiment-cased"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            # device=0 # Eğer GPU varsa ve kullanmak isterseniz bu satırı açın
        )
        print(f"Model ('{model_name}') başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        raise SentimentAnalysisError(f"BERT modeli yüklenemedi: {e}. İnternet bağlantınızı ve gerekli kütüphaneleri kontrol edin.")


    # 1. Veriyi Yükleme
    try:
        # time ve time_parsed sütunlarının geleceğini varsayıyoruz (CommentDownloaderWorker kaydediyorsa)
        df = pd.read_csv(csv_input_path)
        print(f"'{csv_input_path}' başarıyla yüklendi. Satır sayısı: {len(df)}")
        if text_column_name not in df.columns:
             raise SentimentAnalysisError(f"Belirtilen '{text_column_name}' sütunu CSV dosyasında bulunamadı.\nMevcut sütunlar: {df.columns.tolist()}")
    except FileNotFoundError:
        raise SentimentAnalysisError(f"'{csv_input_path}' dosyası bulunamadı.")
    except Exception as e:
        raise SentimentAnalysisError(f"CSV dosyası okunurken bir hata oluştu: {e}")

    # NaN değerleri doldur ve string'e çevir (sadece analiz edilecek sütun için)
    df[text_column_name] = df[text_column_name].fillna('').astype(str)
    

    print("Veri ön işleme tamamlandı.")

    yorum_listesi = df[text_column_name].tolist()

    if not yorum_listesi:
        print("Yorum listesi boş. Analiz edilecek veri yok.")
        # Yorum yoksa bile çıktı dosyasını boş DataFrame ile oluşturabiliriz
        try:
            # Boş veya sadece başlık içeren bir CSV oluştur
            pd.DataFrame(columns=df.columns.tolist() + ['duygu_tahmini', 'duygu_skoru']).to_csv(csv_output_path, index=False, encoding='utf-8-sig')
            print(f"Boş analiz sonuçları '{csv_output_path}' dosyasına başarıyla kaydedildi.")
        except Exception as e:
             print(f"Uyarı: Boş sonuç dosyası kaydedilemedi: {e}")

        return 0, 0 # Yorum yoksa 0, 0 döndür


    # 4. Yorumlara Analizi Uygulama
    print(f"{len(yorum_listesi)} adet yorum için duygu analizi başlıyor...")
    sonuclar = []
    try:
        # Pipeline'ı kullan
        # batch_size ve truncation gibi parametreler performansı etkiler.
        results_generator = sentiment_pipeline(yorum_listesi, batch_size=16, truncation=True)

        # Generator'dan sonuçları çek
        sonuclar = list(results_generator)

        print("Duygu analizi tamamlandı.")

    except Exception as e:
        print(f"Analiz sırasında bir hata oluştu: {e}")
        raise SentimentAnalysisError(f"Analiz sırasında bir hata oluştu: {e}")


    # 5. Sonuçları DataFrame'e Ekleme
    if len(sonuclar) == len(df):
        # Orijinal DataFrame'e yeni sütunlar ekleyelim
        df['duygu_tahmini'] = [sonuc['label'].lower() for sonuc in sonuclar] # Etiketleri küçük harfe çevir
        df['duygu_skoru'] = [sonuc['score'] for sonuc in sonuclar]

        df["hesaplanan_tarih"] = df['time'].apply(convert_relative_time) # Tarih dönüşümünü uygula

        # 6. Sonuçları Kaydetme
        try:
            # Orijinal sütunları ve yeni eklenen sütunları içeren DataFrame'i kaydet
            df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
            print(f"Analiz sonuçları '{csv_output_path}' dosyasına başarıyla kaydedildi.")
        except Exception as e:
             raise SentimentAnalysisError(f"Sonuçlar CSV dosyasına kaydedilirken bir hata oluştu: {e}")

        # 7. Pozitif ve Negatif Sayılarını Hesaplama
        # 'positive' ve 'negative' olarak geldiğini varsayıyoruz (modelin çıktısına göre ayarlayın)
        counts = df['duygu_tahmini'].value_counts()
        positive_count = counts.get('positive', 0) # 'positive' yoksa 0 al
        negative_count = counts.get('negative', 0) # 'negative' yoksa 0 al

        # Diğer etiketler varsa (neutral gibi) bunları da sayabilirsiniz
        # neutral_count = counts.get('neutral', 0)

        print(f"Analiz Tamamlandı. Pozitif: {positive_count}, Negatif: {negative_count}")
        return positive_count, negative_count

    else:
        raise SentimentAnalysisError(f"Analiz sonucu sayısı ({len(sonuclar)}) ile DataFrame satır sayısı ({len(df)}) eşleşmiyor.")