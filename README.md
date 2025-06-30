YtScraperS: Yorumları Anla, Markanı Yönet
Genel Bakış
YtScraperS, markaların YouTube platformundaki ürün ve hizmet yorumlarını derinlemesine analiz ederek müşteri algısını anlamalarına ve stratejilerini buna göre geliştirmelerine yardımcı olan, Yapay Zeka destekli yenilikçi bir çözümdür. Gelişmiş doğal dil işleme yeteneklerimiz sayesinde YtScraperS, milyonlarca kullanıcı yorumunu tarayarak markanıza özel olumlu ve olumsuz geri bildirimleri titizlikle ayrıştırmaktadır.


Uygulamamız, sadece yorumları toplamakla kalmaz; her bir yorumun duygu analizini gerçekleştirir ve Yapay Zeka destekli analizler yaparak anahtar çıkarımlar sunar. Bu sayede markaların dijital itibarlarını, müşteri algılarını ve beklentilerini net bir tablo halinde görmelerini sağlar.


Özellikler ve Fonksiyonlar
YtScraperS, markaların YouTube yorumlarından stratejik içgörüler elde etmesini sağlayan beş entegre modülden oluşmaktadır. Her bir modül, markanızın dijital itibarını güçlendirme ve müşteri memnuniyetini yükseltme yolculuğunda kritik bir rol oynar.


1. Veri Toplama Modülü
Kullanıcılar, belirledikleri ürün veya hizmetle ilgili anahtar kelimeleri içeren YouTube URL'lerini otomatik olarak tespit edebilirler. Bu URL'ler anında CSV formatında kaydedilir. Belirlenen bu video URL'leri üzerinden, her bir videoya ait yorum metinleri, yorum tarihleri, kullanıcı adları ve diğer ilgili meta veriler, gelişmiş web scraping teknolojimizle saniyeler içinde toplanır ve analiz için hazır hale getirilir.



Resim 1: Uygulamanın arayüzü ve veri toplama adımı

2. Duygu Analizi Modülü
BERT tabanlı gelişmiş NLP modeliyle her yorum -1 (çok negatif) ile +1 (çok pozitif) arasında puanlanır ve "Pozitif" veya "Negatif" olarak etiketlenir. Sonuçlar, interaktif pasta ve zaman çizelgesi grafikleriyle duygu dağılımı ve değişimi şeklinde görselleştirilir.

Resim 2: Yorumların duygu dağılımını gösteren pasta grafiği

Resim 3: Yorumların duygu skorlarının zaman grafiği

3. Akıllı Yorum Kategorizasyon Modülü
Yorumlarınızı istediğiniz alanlardaki kategorilere, gelişmiş Büyük Dil Modelleri (LLM) ile otomatik ve akıllı şekilde sınıflandırıyoruz. Böylece müşteri geri bildirimlerine odaklanarak kaynaklarınızı daha verimli yönetebilirsiniz.


Resim 4: Kullanıcıların kategorilerini girdiği alan

Resim 5: Kategorileme sonuçlarının tabloda gösterimi

Resim 6: Kategori bazında zaman grafikleri

4. Akıllı Raporlama Modülü
YtScraperS'in tüm bu güçlü analizleri, markanızın stratejik kararlar almasını sağlayacak anlaşılır ve profesyonel raporlar halinde sunulur.

Resim 7: Analiz raporu dosyası

Uygulama Geliştirme Metodu ve Çalışma Prensibi
YtScraperS, verimlilik ve güvenilirlik hedeflenerek tamamen Python programlama dili ile geliştirilmiştir. Uygulama, doğrudan yerel bir bilgisayarda (PC) çalıştırılmak üzere tasarlanmıştır, bu da kullanıcılara veri üzerinde tam kontrol ve yüksek güvenlik sağlamaktadır.

YtScraperS'in öne çıkan özelliği, çalışma sırasında verileri anlık olarak işlemesi ve kaydetmesidir. Bu sayede elektrik kesintisi, internet bağlantısı sorunu veya beklenmedik kapanma gibi durumlarda, analiz sürecindeki tüm ilerleme ve toplanan veriler otomatik olarak "state.json" adlı bir durum dosyasına kaydedilir ve uygulama tekrar başlatıldığında kesintiye uğradığı yerden sorunsuz bir şekilde devam ederek veri kaybını önler.


Ayrıca, harici bir veritabanına ihtiyaç duymayan YtScraperS, toplanan tüm yorum verilerini ve analiz çıktılarını kullanıcının bilgisayarında tek bir düzenli CSV dosyası halinde kaydederek hem veri erişimini kolaylaştırır hem de kurulum karmaşıklığını minimize eder.

Faydaları

Veri Odaklı Kararlar: Ürün ve hizmet iyileştirmeleri, gelecekteki stratejilerin belirlenmesi ve müşteri memnuniyetini en üst düzeye çıkarma konularında veri odaklı kararlar almanıza olanak tanır.


Dijital İtibar Yönetimi: Markaların dijital itibarlarını, müşteri algılarını ve beklentilerini net bir tablo halinde görmelerini sağlar.


Müşteri Sadakati ve Karlılık Artışı: Müşteri sadakatini artırmasına, karlılıklarını yükseltmesine ve hızla değişen pazar koşullarında emin adımlarla büyüyerek lider konumlarını pekiştirmelerine yardımcı olan stratejik bir iş ortağıdır.


Gerçek Sesleri Duyma: Geleneksel anket yöntemlerinin ötesine geçerek, markaların dijitaldeki gerçek seslerini duymalarını sağlar.

Kurulum ve Kullanım
Yerel bilgisayarınızda YtScraperS'i kurmak ve çalıştırmak için aşağıdaki adımları takip ediniz:

Bash

# 1. Depoyu klonlayın:
#    git clone https://github.com/SuleymanKursatDemir/YtScraperS.git

# 2. Proje dizinine gidin:
#    cd YtScraperS

# 3. Gerekli bağımlılıkları yükleyin:
#    pip install -r requirements.txt

# 4. Uygulamayı çalıştırın:
#    python main.py
İletişim
Markanızın dijital varlığını YtScraperS'in gücüyle zirveye taşımaya hazır mısınız? Müşteri içgörülerinden faydalanarak fark yaratmak ve rekabet avantajı elde etmek için bizimle iletişime geçin ve bir sonraki adımı birlikte planlayalım.

Ürün Sahibi: Süleyman Kürşat DEMİR

E-posta: demiresa38@gmail.com

Telefon: 05060365724

LinkedIn: Süleyman Kürşat DEMİR
