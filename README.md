# YtScraperS: Yorumları Anla, Markanı Yönet

## Genel Bakış

[cite_start]YtScraperS, markaların YouTube platformundaki ürün ve hizmet yorumlarını derinlemesine analiz ederek müşteri algısını anlamalarına ve stratejilerini buna göre geliştirmelerine yardımcı olan, Yapay Zeka destekli yenilikçi bir çözümdür[cite: 2, 3]. [cite_start]Gelişmiş doğal dil işleme yeteneklerimiz sayesinde YtScraperS, milyonlarca kullanıcı yorumunu tarayarak markanıza özel olumlu ve olumsuz geri bildirimleri titizlikle ayrıştırmaktadır[cite: 2].

[cite_start]Uygulamamız, sadece yorumları toplamakla kalmaz; her bir yorumun duygu analizini gerçekleştirir ve Yapay Zeka destekli analizler yaparak anahtar çıkarımlar sunar[cite: 3]. [cite_start]Bu sayede markaların dijital itibarlarını, müşteri algılarını ve beklentilerini net bir tablo halinde görmelerini sağlar[cite: 4].

## Özellikler ve Fonksiyonlar

[cite_start]YtScraperS, markaların YouTube yorumlarından stratejik içgörüler elde etmesini sağlayan beş entegre modülden oluşmaktadır[cite: 19, 20]. [cite_start]Her bir modül, markanızın dijital itibarını güçlendirme ve müşteri memnuniyetini yükseltme yolculuğunda kritik bir rol oynar[cite: 21].

### 1. Veri Toplama Modülü

[cite_start]Kullanıcılar, belirledikleri ürün veya hizmetle ilgili anahtar kelimeleri içeren YouTube URL'lerini otomatik olarak tespit edebilirler[cite: 22]. [cite_start]Bu URL'ler anında CSV formatında kaydedilir[cite: 23]. [cite_start]Belirlenen bu video URL'leri üzerinden, her bir videoya ait yorum metinleri, yorum tarihleri, kullanıcı adları ve diğer ilgili meta veriler, gelişmiş web scraping teknolojimizle saniyeler içinde toplanır ve analiz için hazır hale getirilir[cite: 24].

![Uygulamanın arayüzü ve veri toplama adımı](images/image-2.png)


### 2. Duygu Analizi Modülü

[cite_start]BERT tabanlı gelişmiş NLP modeliyle her yorum -1 (çok negatif) ile +1 (çok pozitif) arasında puanlanır ve "Pozitif" veya "Negatif" olarak etiketlenir[cite: 27]. [cite_start]Sonuçlar, interaktif pasta ve zaman çizelgesi grafikleriyle duygu dağılımı ve değişimi şeklinde görselleştirilir[cite: 27].

![Yorumların duygu dağılımını gösteren pasta grafiği](image-3.png)

![Yorumların duygu skorlarının zaman grafiği](image-4.png)

### 3. Akıllı Yorum Kategorizasyon Modülü


![Kullanıcıların kategorilerini girdiği alan](image-5.png)

![Kategorileme sonuçlarının tabloda gösterimi](image-6.png)

![Kategori bazında zaman grafikleri](image-7.png)

### 4. Akıllı Raporlama Modülü

[cite_start]YtScraperS'in tüm bu güçlü analizleri, markanızın stratejik kararlar almasını sağlayacak anlaşılır ve profesyonel raporlar halinde sunulur[cite: 34].

![Analiz raporu dosyası](image-1.png)

## Uygulama Geliştirme Metodu ve Çalışma Prensibi

[cite_start]YtScraperS, verimlilik ve güvenilirlik hedeflenerek tamamen Python programlama dili ile geliştirilmiştir[cite: 35]. [cite_start]Uygulama, doğrudan yerel bir bilgisayarda (PC) çalıştırılmak üzere tasarlanmıştır, bu da kullanıcılara veri üzerinde tam kontrol ve yüksek güvenlik sağlamaktadır[cite: 35].

[cite_start]YtScraperS'in öne çıkan özelliği, çalışma sırasında verileri anlık olarak işlemesi ve kaydetmesidir[cite: 36]. [cite_start]Bu sayede elektrik kesintisi, internet bağlantısı sorunu veya beklenmedik kapanma gibi durumlarda, analiz sürecindeki tüm ilerleme ve toplanan veriler otomatik olarak "state.json" adlı bir durum dosyasına kaydedilir ve uygulama tekrar başlatıldığında kesintiye uğradığı yerden sorunsuz bir şekilde devam ederek veri kaybını önler[cite: 37].

[cite_start]Ayrıca, harici bir veritabanına ihtiyaç duymayan YtScraperS, toplanan tüm yorum verilerini ve analiz çıktılarını kullanıcının bilgisayarında tek bir düzenli CSV dosyası halinde kaydederek hem veri erişimini kolaylaştırır hem de kurulum karmaşıklığını minimize eder[cite: 38].

## Faydaları

* [cite_start]**Veri Odaklı Kararlar:** Ürün ve hizmet iyileştirmeleri, gelecekteki stratejilerin belirlenmesi ve müşteri memnuniyetini en üst düzeye çıkarma konularında veri odaklı kararlar almanıza olanak tanır[cite: 6].
* [cite_start]**Dijital İtibar Yönetimi:** Markaların dijital itibarlarını, müşteri algılarını ve beklentilerini net bir tablo halinde görmelerini sağlar[cite: 4].
* [cite_start]**Müşteri Sadakati ve Karlılık Artışı:** Müşteri sadakatini artırmasına, karlılıklarını yükseltmesine ve hızla değişen pazar koşullarında emin adımlarla büyüyerek lider konumlarını pekiştirmelerine yardımcı olan stratejik bir iş ortağıdır[cite: 8].
* [cite_start]**Gerçek Sesleri Duyma:** Geleneksel anket yöntemlerinin ötesine geçerek, markaların dijitaldeki gerçek seslerini duymalarını sağlar[cite: 7].

## Kurulum ve Kullanım 

```bash
# Bu kısım uygulamanızın nasıl kurulacağına dair adımları içermelidir.
# Örnek:
# 1. Depoyu klonlayın:
#    git clone [https://github.com/SuleymanKursatDemir/YtScraperS.git](https://github.com/SuleymanKursatDemir/YtScraperS.git)
# 2. Proje dizinine gidin:
#    cd YtScraperS
# 3. Gerekli bağımlılıkları yükleyin:
#    pip install -r requirements.txt
# 4. Uygulamayı çalıştırın:
#    python main.py
