# web_search.py
import time
import urllib.parse
import sys # Hata loglarını stderr'e yazdırmak için

# Selenium modülleri
# Service sınıfı WebDriver'ı başlatmak için gereklidir (webdriver_manager ile kullanacağız)
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException,
    SessionNotCreatedException, InvalidArgumentException
)

# webdriver_manager kütüphanesi
# Bu kütüphane ChromeDriver'ı otomatik olarak indirip yönetir
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
    print("webdriver_manager kütüphanesi bulundu.") # Konsol logu
except ImportError:
    print("UYARI: webdriver_manager kütüphanesi bulunamadı. Lütfen 'pip install webdriver-manager' komutunu çalıştırın.") # Konsol logu
    print("ChromeDriver.exe'nin sistem PATH'inizde veya uygulamanın yanında olması gerekecek.") # Konsol logu
    USE_WEBDRIVER_MANAGER = False


# PyQt6 modülleri (Worker için gerekli QObject ve Sinyaller)
from PyQt6.QtCore import QObject, pyqtSignal, QThread

# --- Selenium Arama İşçisi (Worker) Sınıfı ---
class SearchWorker(QObject):
    # Sinyaller
    # GUI tarafından manüel set edilecek, worker sadece ilk kez "başladı" bilgisini konsola yazacak.
    # search_progress = pyqtSignal(str) # Artık GUI'de direkt kullanılmıyor
    # Arama tamamlandığında yayılır (Bulunan URL listesi)
    search_finished = pyqtSignal(list)
    # Arama sırasında hata oluştuğunda yayılır (Hata mesajı)
    search_error = pyqtSignal(str)

    def __init__(self, query, limit, parent=None):
        super().__init__(parent)
        self.query = query
        self.limit = limit
        self._is_running = True # Durdurma bayrağı
        self.driver = None      # WebDriver instance'ı
        self.found_urls = []    # Bulunan URL'leri burada tutalım

    def stop(self):
        """Worker'ı durdurma isteği gönderir."""
        print("Arama durdurma isteği alındı...") # Konsol logu
        self._is_running = False
        # WebDriver'ı kapatma işlemi run metodunun finally bloğunda yönetilir.
        # Burada sadece bayrağı set etmek yeterlidir.

    def run(self):
        """Selenium ile YouTube araması yapar."""
        self._is_running = True # run her çağrıldığında bayrağı resetle
        self.found_urls = [] # Her çalıştırmada listeyi temizle

        print(f"Arama Başlatılıyor (Selenium): '{self.query}' için {self.limit} URL aranıyor.") # Konsol logu

        # --- Selenium WebDriver Ayarları ---
        options = webdriver.ChromeOptions()
        # Headless mod: Tarayıcı penceresini göstermez
        options.add_argument('--headless')
        # Güvenlik ve ortam ayarları (bazı Linux ortamları için gerekli olabilir)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # Log seviyesini azalt
        options.add_argument('--log-level=3')
        # GPU kullanımını kapat (headless modda bazen sorun çıkarır)
        options.add_argument('--disable-gpu')
        # User-Agent belirleme (bot gibi görünmemek için)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        # Otomasyon izlerini gizleme (YouTube'un bot tespiti için bakabileceği şeyler)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # options.add_argument("--disable-blink-features=AutomationControlled") # Daha agresif gizleme


        # WebDriver başlatma ve ana işlem bloğu
        try:
            # --- ChromeDriver Başlatılıyor (webdriver-manager kullanarak veya PATH'ten) ---
            print("ChromeDriver başlatılıyor...") # Konsol logu
            if USE_WEBDRIVER_MANAGER:
                # webdriver-manager kullanıyorsak, otomatik indirip yolunu al
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                print("ChromeDriver webdriver_manager ile başlatıldı.") # Konsol logu
            else:
                # webdriver-manager yoksa, PATH'ten veya manuel yoldan bulmaya çalış
                # Manuel yol göndermeyecekseniz, driver'ın PATH'te olması gerekir.
                # Daha önceki manuel yol belirten satırı kaldırdık: Service('C:\\Users\\Dell\\Desktop\\Youtube Projects\\chromedriver.exe')
                self.driver = webdriver.Chrome(options=options)
                print("ChromeDriver PATH'ten başlatıldı (webdriver_manager kullanılmadı).") # Konsol logu



            # İlk kontrol: Durdurma isteği geldi mi?
            if not self._is_running:
                 print("Başlamadan durdurma isteği alındı.") # Konsol logu
                 raise Exception("Arama başlatılmadan durdurma isteği alındı.")


            # --- Arama Sayfasına Git ---
            encoded_sorgu = urllib.parse.quote_plus(self.query)
            search_url = f"https://www.youtube.com/results?search_query={encoded_sorgu}"
            print(f"Arama URL'i yükleniyor: {search_url}") # Konsol logu
            self.driver.get(search_url)

            # Sayfanın yüklenmesini bekle (video elementlerinin görünmesini bekle)
            # YouTube sonuçlarında video elementleri genellikle ytd-video-renderer veya ytd-compact-video-renderer gibi custom elementlerdir.
            # a#video-title selector'ü ise video başlık linkidir.
            video_link_selector = "a#video-title"
            print(f"Arama sonuçları yükleniyor ve '{video_link_selector}' elementi bekleniyor...") # Konsol logu
            try:
                # Sayfanın temel yüklenmesini bekleyin
                 WebDriverWait(self.driver, 20).until( # Sayfanın yüklenmesi için bekleme süresini artırdık
                     lambda driver: driver.execute_script("return document.readyState") == "complete"
                 )
                 print("Sayfa yüklenmesi tamamlandı (readyState).") # Konsol logu

                 # Video linklerinin görünmesini bekleyin (bazı sonuçlar gelene kadar)
                 WebDriverWait(self.driver, 10).until( # İlk video linkinin gelmesi için ek bekleme
                     EC.presence_of_element_located((By.CSS_SELECTOR, video_link_selector))
                 )
                 print("İlk video link elementi bulundu.") # Konsol logu

            except TimeoutException:
                 # Eğer belirli bir süre içinde arama sonuçları yüklenmezse
                 error_msg = "Arama sonuçları zamanında yüklenemedi (Timeout)."
                 print(f"Hata: {error_msg}", file=sys.stderr) # Konsol logu
                 self.search_error.emit(error_msg)
                 # TimeoutException'ı tekrar fırlat ki finally bloğu çalışsın
                 raise TimeoutException(error_msg)
            except Exception as e:
                error_msg = f"Sayfa yüklenirken beklenmeyen hata: {e}"
                print(f"Hata: {error_msg}", file=sys.stderr) # Konsol logu
                self.search_error.emit(error_msg)
                raise # Hatayı yakalayıp tekrar fırlatıyoruz ki finally çalışsın

            # Durdurma isteği geldi mi?
            if not self._is_running:
                 print("Arama sonuçları yüklendikten sonra durdurma isteği alındı.") # Konsol logu
                 raise Exception("Arama durdurma isteği alındı.")

            # --- Video Linklerini Çekme ve Sayfayı Kaydırma ---
            processed_urls = set() # Daha önce işlenmiş URL'leri takip etmek için set
            self.found_urls = [] # Bulunan URL'leri tutan liste (limit kontrolü için)

            scroll_pause_time = 1.5 # Kaydırma sonrası bekleme süresi (yeni içeriğin yüklenmesi için)
            scroll_distance = 3000 # Her seferinde kaydırılacak piksel miktarı (daha büyük bir değer olabilir)

            last_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            scroll_attempts_without_new_content = 0 # Yüksekliğin veya yeni URL sayısının değişmediği deneme sayısı
            MAX_IDLE_SCROLL_ATTEMPTS = 7 # Kaç boş kaydırma denemesinden sonra durulacak (Timeout'tan sonra durabilir)

            print(f"İlk {self.limit} URL bulunana veya sayfa sonuna ulaşılana kadar kaydırılıyor...") # Konsol logu

            # Limite ulaşılana veya durdurulana veya sayfa sonuna gelinene kadar döngü
            while len(self.found_urls) < self.limit and self._is_running and scroll_attempts_without_new_content < MAX_IDLE_SCROLL_ATTEMPTS:

                initial_urls_count_in_loop = len(self.found_urls) # Bu iterasyona girerken bulunan URL sayısı

                # Şu anki görünümdeki tüm video link elementlerini bul
                # Bu liste her kaydırmada büyüyebilir
                # Hatalı elementleri try-except ile atla
                try:
                    # Görünür veya sayfada olan link elementlerini bulmaya çalış
                    # find_elements her zaman bir liste döndürür, bulunamazsa boş liste.
                    video_link_elements = self.driver.find_elements(By.CSS_SELECTOR, video_link_selector)
                    # print(f"Mevcut görünümde {len(video_link_elements)} olası link elementi bulundu.") # Debug log

                except Exception as e:
                    print(f"Uyarı: video link elementleri bulunurken hata: {e}. Devam ediliyor.") # Konsol logu
                    video_link_elements = [] # Hata olursa boş liste ile devam et

                # Bulunan link elementleri içinde dönerek yeni linkleri çek
                for link_element in video_link_elements:
                    # İç döngüde de limit veya durdurma kontrolü yap
                    if len(self.found_urls) >= self.limit or not self._is_running:
                        break # İç döngüyü kır

                    try:
                        href = link_element.get_attribute("href")

                        if href and "/watch?v=" in href:
                             # URL'in sadece video ID kısmını al (ek parametreleri temizle)
                             full_url = href.split('&')[0]

                             # Eğer bu URL daha önce eklenmemişse listeye ekle
                             if full_url not in processed_urls:
                                 self.found_urls.append(full_url)
                                 processed_urls.add(full_url) # İşlenmiş URL'ler setine ekle
                                 # print(f"Bulunan URL ({len(self.found_urls)}/{self.limit}): {full_url}") # Konsol logu


                    except Exception as e:
                        # Bir link elementini işlerken hata oluşursa (nadir olmalı)
                        print(f"Uyarı: Bir link elementini işlerken hata: {e}. Atlanıyor.") # Konsol logu
                        continue # Bu linki atla

                # İç döngü bittikten sonra, dış döngü koşullarını tekrar kontrol et
                if len(self.found_urls) >= self.limit or not self._is_running:
                     print(f"Döngü sonlandırma koşulu sağlandı: Limit ({self.limit}) ulaşıldı mı? {len(self.found_urls) >= self.limit}. Durdurma isteği var mı? {not self._is_running}") # Konsol logu
                     break # Limite ulaşıldı veya durdurma isteği geldi, ana döngüyü kır

                # --- Aşağı kaydırma ---
                print(f"Aşağı kaydırılıyor... Şu ana kadar bulunan: {len(self.found_urls)}/{self.limit}") # Konsol logu
                self.driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
                time.sleep(scroll_pause_time) # Yeni içeriğin yüklenmesi için bekle

                # Yeni sayfa yüksekliğini al ve karşılaştır
                new_height = self.driver.execute_script("return document.documentElement.scrollHeight")

                # Yüksekliğin değişmediği veya yeni URL bulunmadığı durumları kontrol et
                if new_height == last_height and len(self.found_urls) == initial_urls_count_in_loop:
                    # Hem yükseklik değişmediyse HEM DE yeni URL bulunamadıysa
                    scroll_attempts_without_new_content += 1
                    print(f"Sayfa yüksekliği değişmedi VEYA yeni URL bulunamadı. Boş deneme: {scroll_attempts_without_new_content}/{MAX_IDLE_SCROLL_ATTEMPTS}") # Konsol logu
                else:
                    # Yükseklik değiştiyse VEYA yeni URL bulunduysa sayacı sıfırla
                    scroll_attempts_without_new_content = 0
                    if new_height > last_height:
                         print("Sayfa yüksekliği arttı.") # Konsol logu
                    if len(self.found_urls) > initial_urls_count_in_loop:
                         print("Yeni URL'ler bulundu.") # Konsol logu
                    if new_height > last_height or len(self.found_urls) > initial_urls_count_in_loop:
                         print("Yeni içerik bulundu veya kaydırıldı, sayaç sıfırlandı.") # Konsol logu


                last_height = new_height # Son yüksekliği güncelle

            # --- Arama Tamamlandı (Döngü Sonlandı) ---
            print("Arama döngüsü tamamlandı.") # Konsol logu

            if not self.found_urls and self._is_running and self.limit > 0:
                 # Döngü bitti, URL bulunamadı, durdurulmadı ve limit > 0 ise uyarı ver
                 warning_msg = "Arama tamamlandı ancak beklenen sayıda veya hiç video linki bulunamadı."
                 print(f"Uyarı: {warning_msg}", file=sys.stderr) # Konsol logu
                 # search_error sinyali yaymak tüm süreci durdurur.
                 # Sadece bir uyarı vermek istiyorsak, GUI'de farklı bir sinyal veya mekanizma olmalı.
                 # Şimdilik loglayıp finished sinyali ile boş liste gönderelim.
                 # self.search_error.emit(warning_msg) # Tüm süreci durdurmak istemiyorsak bunu yapmayalım.


        # --- Hata Yakalama ---
        # chromedriver'ın bulunamadığı veya başlatılamadığı hata
        except SessionNotCreatedException as e:
             error_msg = f"ChromeDriver başlatılamadı: {e}.\nLütfen Chrome tarayıcınızın kurulu olduğundan ve sürümünün kullanılan ChromeDriver sürümüyle uyumlu olduğundan emin olun.\nEğer webdriver-manager kullanıyorsanız, 'pip install webdriver-manager' komutunu çalıştırın.\nKullanmıyorsanız, Chrome sürümünüze uygun ChromeDriver.exe dosyasını indirin ve sistem PATH'inize ekleyin."
             print(f"Hata: {error_msg}", file=sys.stderr) # Konsol logu
             self.search_error.emit(error_msg)
        except InvalidArgumentException as e:
             error_msg = f"Selenium argüman hatası: {e}.\nMuhtemelen ChromeOptions ayarlarında veya URL'de bir problem var."
             print(f"Hata: {error_msg}", file=sys.stderr) # Konsol logu
             self.search_error.emit(error_msg)
        except TimeoutException:
            # TimeoutException yukarıda zaten yakalanıp işlendi ve tekrar fırlatıldı
            pass # Burada tekrar işleme gerek yok
        except WebDriverException as e:
            error_msg = f"WebDriver hatası oluştu: {e}\nBu genellikle ChromeDriver ile tarayıcı arasındaki bir sorundan kaynaklanır.\nChrome tarayıcınızın ve ChromeDriver sürümünüzün uyumlu olduğundan emin olun."
            print(f"Hata: {error_msg}", file=sys.stderr) # Konsol logu
            self.search_error.emit(error_msg)
        except Exception as e:
            error_msg = f"Arama sırasında beklenmeyen bir hata oluştu: {e}"
            print(f"Hata: {error_msg}", file=sys.stderr) # Konsol logu
            self.search_error.emit(error_msg)

        # --- Temizlik Bloğu (Hata olsa da olmasa da çalışır) ---
        finally:
            # Tarayıcıyı kapatmayı garanti et
            if self.driver:
                print("Tarayıcı kapatılıyor.") # Konsol logu
                try:
                    self.driver.quit()
                except Exception as e:
                    # quit() sırasında hata olursa logla ama işlemi durdurma
                    print(f"Tarayıcı kapatılırken hata: {e}", file=sys.stderr) # Konsol logu
                self.driver = None # Temizlik

            # İşlem bitti sinyalini gönder
            # finished sinyali her zaman o ana kadar bulunan URL listesi ile yayılır.
            # main.py'deki handle_search_completion metodu liste boşsa bunu yorumlayacaktır.
            print(f"Arama worker finished sinyali yayıldı. {len(self.found_urls)} URL ile.")
            self.search_finished.emit(self.found_urls)


# Test için:
if __name__ == '__main__':
    from PyQt6.QtCore import QCoreApplication, QTimer # QTimer test durdurma için
    import os # Dosya var mı diye kontrol etmek için

    # QApplication olmadan QThread kullanmak için QCoreApplication kullanabiliriz
    # Eğer GUI test edecekseniz QApplication kullanın.
    # app = QApplication([]) # GUI testi için
    app = QCoreApplication([]) # Worker testi için

    def search_done(urls):
        print("\nArama Sonuçları:")
        if urls:
            for i, url in enumerate(urls):
                print(f"{i+1}: {url}")
        else:
            print("URL bulunamadı veya işlem durduruldu/hatalı tamamlandı.")
        app.quit()

    def search_err(msg):
        print(f"\nArama Hatası: {msg}")
        app.quit()
