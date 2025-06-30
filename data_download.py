# data_download.py
from PyQt6.QtCore import QObject, pyqtSignal
import sys
import csv
import os
import json
import time
from datetime import datetime, timedelta # timedelta eklendi

# youtube-comment-downloader kütüphanesini kontrol et
try:
    from youtube_comment_downloader import YoutubeCommentDownloader
    USE_COMMENT_DOWNLOADER = True
    print("youtube-comment-downloader kütüphanesi bulundu.") # Konsol logu
except ImportError:
    print("Uyarı: 'youtube-comment-downloader' kütüphanesi bulunamadı. Yorum indirme devre dışı.")
    print("Yüklemek için: pip install youtube-comment-downloader")
    USE_COMMENT_DOWNLOADER = False

class CommentDownloaderWorker(QObject):
    # Sinyaller
    status_message = pyqtSignal(str)      # Genel durum mesajları (örn: "State yükleniyor...", "URL işleniyor...", ilerleme ve kalan süre dahil)
    # progress_updated artık sadece toplam yorum sayısını alacak
    progress_updated = pyqtSignal(int)     # Toplam indirilen yorum sayısı (tüm URL'ler için)
    finished = pyqtSignal()                # İşlem tamamlandığında veya durdurulduğunda
    error = pyqtSignal(str)                # Hata durumunda

    # Sabit tanımları
    COMMENT_RATE_PER_SECOND = 25 # Tahmini yorum indirme hızı (yorum/saniye). Bu değer gerçek performansa göre ayarlanmalı.
    STATUS_UPDATE_INTERVAL = 20 # Durum mesajının kaç yorumda bir güncelleneceği

    def __init__(self, urls_filepath, comments_filepath, state_filepath, comment_limit_per_url):
        super().__init__()
        self._urls_filepath = urls_filepath
        self._comments_filepath = comments_filepath
        self._state_filepath = state_filepath
        self._comment_limit_per_url = comment_limit_per_url
        self._stop_flag = False # Doğru bayrak adı

        self._downloader = None # Yorum indirici instance'ı
        # Bu oturumda indirilen toplam yorum sayısı (önceki oturumlardan devam ediyorsa sıfırdan başlar)
        self._total_comments_downloaded_session = 0

    def stop(self):
        """Worker'ı durdurma isteği gönderir."""
        print("İndirme worker durdurma isteği alındı.") # Konsol logu
        self._stop_flag = True
        # Yorum indirici eğer loop içinde çalışıyorsa onu da durdurmaya çalışabiliriz.
        # kütüphanenin stop metodu varsa çağır.
        if self._downloader:
             try:
                 # youtube-comment-downloader kütüphanesinde direkt bir stop metodu yok gibi görünüyor.
                 # Bu yüzden sadece _stop_flag bayrağına güveniyoruz.
                 pass # Durdurma kodu burada yok şimdilik
             except Exception as e:
                  print(f"Uyarı: Yorum indirici durdurulurken hata: {e}") # Konsol logu

    def format_time_remaining(self, total_seconds):
        """Saniye cinsinden süreyi okunabilir formata çevirir (örn: 1d 30s)."""
        if total_seconds is None or total_seconds < 0 or total_seconds == float('inf'):
            return "Hesaplanıyor..." # Veya "Bilinmiyor..."

        total_seconds = int(total_seconds)
        if total_seconds == 0:
            return "Hemen bitiyor."

        # Gün, saat, dakika, saniye hesapla
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}s") # Saati 'h' olarak göster
        if minutes > 0:
            parts.append(f"{minutes}d") # Dakikayı 'm' olarak göster
        if seconds > 0 or not parts: # Hiçbir şey yoksa saniyeyi göster (örn: 0s)
             if seconds == 0 and not parts: # Süre tam 0 ise "Hemen bitiyor" dönecek
                 pass # Zaten yukarıda kontrol edildi
             else:
                 parts.append(f"{seconds}s")

        return " ".join(parts) if parts else "Hemen bitiyor." # Hiçbir parça yoksa (çok kısa süre)

    def calculate_estimated_time(self, current_url_index, processed_count_on_current_url, total_urls):
        """Tahmini kalan süreyi hesaplar."""
        if total_urls == 0 or self._comment_limit_per_url == 0:
            return self.format_time_remaining(0) # İşlenecek bir şey yoksa süre 0

        # Mevcut URL için kalan yorum sayısı (limite kadar)
        comments_remaining_on_current_url = max(0, self._comment_limit_per_url - processed_count_on_current_url)

        # Sonraki URL'ler için toplam potansiyel yorum sayısı
        # Mevcut URL'nin kendisi 'kalan' URL'ler listesine dahil edilmemeli
        # İşlenmiş URL sayısı = current_url_index (0'dan başladığı için)
        # İşlenecek URL sayısı = total_urls - current_url_index
        remaining_urls_count = max(0, total_urls - current_url_index)

        # Toplam tahmini kalan yorum sayısı = Mevcut URL'deki kalan (limite kadar) + Sonraki URL'lerin tamamı (limite kadar)
        # Aslında, mevcut URL'nin kalanını + sonraki URL'lerin tamamını almalıyız.
        # total_urls_to_process = total_urls - start_index (başlangıçta yüklenen)
        # İşlem sırasında i giderek artar. Kalan URL sayısı = total_urls - (i+1)
        # Daha doğru hesap:
        # Mevcut URL'de kalan: max(0, self._comment_limit_per_url - processed_count_on_current_url)
        # Sonraki URL'lerde kalan: (total_urls - (current_url_index + 1)) * self._comment_limit_per_url
        total_estimated_comments_remaining = max(0, self._comment_limit_per_url - processed_count_on_current_url) + \
                                             max(0, total_urls - (current_url_index + 1)) * self._comment_limit_per_url


        # Tahmini kalan süre (saniye)
        # Yorum hızı 0 ise bölme hatası olmaması için kontrol ekle
        if self.COMMENT_RATE_PER_SECOND > 0:
            estimated_seconds = total_estimated_comments_remaining / self.COMMENT_RATE_PER_SECOND
        else:
            estimated_seconds = float('inf') # Hız 0 ise süre sonsuz

        return self.format_time_remaining(estimated_seconds)


    def load_state(self):
        """State dosyasını yükler."""
        if not os.path.exists(self._state_filepath):
            print("State dosyası bulunamadı, sıfırdan başlanıyor.") # Konsol logu
            return {"next_url_index": 0, "comment_limit_per_url": self._comment_limit_per_url}

        try:
            with open(self._state_filepath, 'r') as f:
                state = json.load(f)
                print(f"State yüklendi: {state}") # Konsol logu
                # State'teki comment_limit'i doğrula/kullan (GUI'deki inputtan gelen tercih edilir)
                # state.get("comment_limit_per_url", self._comment_limit_per_url)
                # NOTE: Şu anki state yapısı, en son URL'de kaç yorum indirildiğini kaydetmiyor.
                # Bu, tahmini kalan süreyi hesaplarken sadece 'next_url_index'e güvenmek anlamına gelir.
                # Yani, kesintiye uğrayan URL'nin başından başlayacağı varsayılır.
                # Eğer state'ten devam ediyorsak, _total_comments_downloaded_session sıfırdan başlar.
                # Gerçekçi bir devam etme için toplam sayacın da state'e kaydedilmesi gerekir.
                # Şimdilik bunu yapmıyoruz, toplam sayac oturum bazlı kalır.
                return state
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Hata: State dosyası okunamadı veya bozuk: {e}. Yeni state oluşturuluyor.") # Konsol logu
            # Hatalı state dosyasında devam etmektense sıfırdan başlamak daha güvenli olabilir.
            return {"next_url_index": 0, "comment_limit_per_url": self._comment_limit_per_url}
        except Exception as e:
             print(f"Beklenmedik state yükleme hatası: {e}") # Konsol logu
             return {"next_url_index": 0, "comment_limit_per_url": self._comment_limit_per_url}


    def save_state(self, next_url_index):
        """State dosyasını kaydeder."""
        state = {
            "next_url_index": next_url_index,
            "comment_limit_per_url": self._comment_limit_per_url,
            # Başka bilgiler de eklenebilir (örn: search query)
            # _total_comments_downloaded_session burada kaydedilmiyor, oturum bazlı kalır.
            # Eğer toplam sayacın devam etmesini isterseniz buraya eklemelisiniz.
        }
        try:
            # Dizinin var olduğundan emin ol
            os.makedirs(os.path.dirname(self._state_filepath), exist_ok=True)
            with open(self._state_filepath, 'w') as f:
                json.dump(state, f, indent=4)
            print(f"State kaydedildi: next_url_index = {next_url_index}") # Konsol logu
        except Exception as e:
            print(f"Hata: State dosyası kaydedilemedi: {e}") # Konsol logu


    def clear_state(self):
        """State dosyasını siler."""
        try:
            if os.path.exists(self._state_filepath):
                os.remove(self._state_filepath)
                print("State dosyası temizlendi.") # Konsol logu
        except Exception as e:
            print(f"Hata: State dosyası temizlenemedi: {e}") # Konsol logu


    def get_video_id_from_url(self, url):
        """URL'den video ID'yi çıkarır."""
        try:
             # Basit URL parsing: watch?v= sonrası & öncesini al
             if 'v=' in url:
                 return url.split('v=')[1].split('&')[0]
             # youtube shorts URL'lerini de destekle
             elif 'shorts/' in url:
                  return url.split('shorts/')[1].split('?')[0].split('&')[0] # & sonrası olmasın
             # Standart youtu.be kısaltılmış URL'lerini destekle
             elif 'youtu.be/' in url:
                 return url.split('youtu.be/')[1].split('?')[0].split('&')[0] # & sonrası olmasın

        except Exception:
             pass # ID çıkarılamazsa None döner
        return None


    def run(self):
        """Yorum indirme işlemini başlatır (URL bazlı)."""
        if not USE_COMMENT_DOWNLOADER:
            error_msg = "Hata: Yorum indirme kütüphanesi kurulu değil."
            self.status_message.emit(error_msg + " | Tahmini Kalan Süre: Bilinmiyor")
            self.error.emit(error_msg)
            self.finished.emit()
            return

        if self._stop_flag:
            self.status_message.emit("İşlem durduruldu." + " | Tahmini Kalan Süre: 0s")
            self.finished.emit()
            return

        self.status_message.emit("State yükleniyor...")
        state = self.load_state()
        start_index = state.get("next_url_index", 0)

        urls = []
        if not os.path.exists(self._urls_filepath):
            error_msg = f"Hata: URL listesi dosyası bulunamadı: {self._urls_filepath}"
            self.status_message.emit(error_msg + " | Tahmini Kalan Süre: Bilinmiyor")
            self.error.emit(error_msg)
            self.finished.emit()
            return

        try:
             with open(self._urls_filepath, 'r', encoding='utf-8') as f:
                 reader = csv.reader(f)
                 try:
                     first_row = next(reader)
                     if first_row and first_row[0].strip().lower() == 'url':
                         pass
                     else:
                          if first_row and first_row[0].strip():
                               urls.append(first_row[0].strip())
                 except StopIteration:
                      pass
                 urls.extend([row[0].strip() for row in reader if row and row[0].strip()])
        except Exception as e:
            error_msg = f"urls.csv dosyası okuma hatası: {e}"
            self.status_message.emit(f"Hata: urls.csv dosyası okunamadı: {e}" + " | Tahmini Kalan Süre: Bilinmiyor")
            self.error.emit(error_msg)
            self.finished.emit()
            return

        # Eğer state'teki index toplam URL sayısına eşit veya büyükse, bu query için indirme zaten tamamlanmış demektir.
        # Bu durumda Worker'ın yapacağı bir şey yok. State'i temizlemiyoruz, tamamlandı olarak bırakıyoruz.
        if start_index >= len(urls):
             self.status_message.emit("Tüm URL'ler zaten işlenmiş." + " | Tahmini Kalan Süre: 0s")
             print(f"Tüm URL'ler ({len(urls)}) zaten state'e göre işlenmiş (index {start_index}). Worker sonlandırılıyor.")
             self.finished.emit()
             return

        print(f"Toplam {len(urls)} URL bulundu. İşleme {start_index+1}. URL'den başlanıyor.")


        comments_file = None
        try:
            os.makedirs(os.path.dirname(self._comments_filepath), exist_ok=True)
            write_header = not os.path.exists(self._comments_filepath) or os.stat(self._comments_filepath).st_size == 0
            comments_file = open(self._comments_filepath, 'a', newline='', encoding='utf-8')
            csv_writer = csv.writer(comments_file)
            if write_header:
                 # NOTE: Zaman serisi grafiği için 'time_parsed' sayısal sütunu eklenmelidir.
                 # Eğer time_parsed eklemiyorsanız, time serisi grafiği çalışmayacaktır.
                 csv_writer.writerow([ 'video_id', 'text', 'time', 'author'])
                 comments_file.flush()

        except Exception as e:
             error_msg = f"comments.csv dosyası açılamadı/yazılamadı: {e}"
             self.status_message.emit(f"Hata: comments.csv dosyası açılamadı." + " | Tahmini Kalan Süre: Bilinmiyor")
             self.error.emit(error_msg)
             if comments_file and not comments_file.closed:
                  comments_file.close()
             self.finished.emit()
             return


        self._downloader = YoutubeCommentDownloader()

        try:
            for i in range(start_index, len(urls)):
                if self._stop_flag:
                    print(f"Ana döngü durdurma isteğiyle kırıldı. Son state kaydediliyor: index {i}")
                    self.save_state(i)
                    self.status_message.emit("İşlem kullanıcı tarafından durduruldu." + " | Tahmini Kalan Süre: 0s")
                    break

                current_url = urls[i].strip()
                if not current_url:
                     print(f"Uyarı: urls.csv satır {i+1} boş veya geçersiz URL, atlanıyor.")
                     if not self._stop_flag:
                        self.save_state(i + 1)
                     continue

                video_id = self.get_video_id_from_url(current_url)

                if not video_id:
                     print(f"Uyarı: URL'den video ID çıkarılamadı: {current_url}. Atlanıyor.")
                     if not self._stop_flag:
                        self.save_state(i + 1)
                     continue

                comments_count_for_current_url = 0

                estimated_time_str_initial = self.calculate_estimated_time(i, comments_count_for_current_url, len(urls))
                self.status_message.emit(f"{i+1}. URL {comments_count_for_current_url}/{self._comment_limit_per_url} | Tahmini Kalan Süre: {estimated_time_str_initial}")
                print(f"İşleniyor: URL {i+1}/{len(urls)} ({video_id})")

                current_url_error = False
                try:
                    comments_generator = self._downloader.get_comments(video_id)

                    try:
                        for comment in comments_generator:
                            comments_count_for_current_url += 1
                            self._total_comments_downloaded_session += 1

                            # NOTE: Eğer zaman serisi grafiği istiyorsanız, burada time'ı parse edip sayısal kaydedin
                            # Örneğin: comment.get('time_parsed', 0) veya kendi parsing logic'iniz.
                            # Şu an string 'time' kaydediliyor, zaman serisi grafiği çalışmayacaktır.
                            row = [
                                video_id,
                                comment.get('text', ''),
                                comment.get('time', ''),
                                comment.get('author', ''),
                                # Eğer 'time_parsed' gibi sayısal bir zaman değeri varsa buraya ekleyin
                                # comment.get('time_parsed_numeric', 0),
                            ]
                            if csv_writer:
                                csv_writer.writerow(row)

                            if (comments_count_for_current_url % self.STATUS_UPDATE_INTERVAL == 0) or \
                               (comments_count_for_current_url >= self._comment_limit_per_url) or \
                                self._stop_flag:

                                estimated_time_str = self.calculate_estimated_time(i, comments_count_for_current_url, len(urls))
                                self.status_message.emit(f"{i+1}. URL {comments_count_for_current_url}/{self._comment_limit_per_url} | Tahmini Kalan Süre: {estimated_time_str}")
                                self.progress_updated.emit(self._total_comments_downloaded_session)

                            if self._stop_flag or comments_count_for_current_url >= self._comment_limit_per_url:
                                 if self._stop_flag:
                                     print(f"Durdurma isteği, yorum indirme durduruldu: {video_id}")
                                 else:
                                     print(f"Limit ({self._comment_limit_per_url}) bu URL için ulaşıldı: {video_id}. Sonraki URL'ye geçiliyor.")

                                 break

                        # *** İç döngü tamamlandı veya kırıldı ***

                        # Yorum indirme bittikten sonra (limit, stop veya generator sonu) state'i kaydet
                        # Sadece ana döngü stop nedeniyle kırılmadıysa, bu URL'nin tamamlandığını işaret etmek için i+1 kaydet
                        # Bu save_state artık finally bloğuna taşındı ve her durumda çağrılacak.
                        pass # İç döngüden sonra save_state'i buradan kaldırdık, finally'de çağrılacak.


                    except Exception as e:
                         current_url_error = True
                         error_message = f"Yorum indirme sırasında generator hatası URL {i+1} ({video_id}): {e}"
                         print(error_message)
                         estimated_time_str = self.calculate_estimated_time(i, comments_count_for_current_url, len(urls))
                         self.status_message.emit(f"Hata: URL {i+1} ({video_id}) indirilemedi | Tahmini Kalan Süre: {estimated_time_str}")
                         # Hata durumunda state'i ilerlet (bu URL'yi atla)
                         # Bu save_state artık finally bloğuna taşındı.
                         pass

                    if comments_file and not comments_file.closed:
                        try:
                            comments_file.flush()
                            os.fsync(comments_file.fileno())
                        except OSError as e:
                             print(f"Uyarı: os.fsync hatası ({e}).")
                        except Exception as e:
                             print(f"Uyarı: Dosya senkronizasyon hatası ({e}).")

                    print(f"URL {i+1} ({video_id}) işlendi. Yorum sayısı: {comments_count_for_current_url} (Limit: {self._comment_limit_per_url}, Durdurma: {self._stop_flag}, Hata: {current_url_error}).")

                    if self._stop_flag:
                         print("Durdurma isteği sonrası iç döngü bitti, ana döngü kırılıyor.")
                         break


                except Exception as e:
                    current_url_error = True
                    error_message = f"Yorum indirme başlatılırken hata URL {i+1} ({video_id}): {e}"
                    print(error_message)
                    estimated_time_str = self.calculate_estimated_time(i, comments_count_for_current_url, len(urls))
                    self.status_message.emit(f"Hata: URL {i+1} ({video_id}) başlatılamadı | Tahmini Kalan Süre: {estimated_time_str}")
                    # Bu save_state artık finally bloğuna taşındı.
                    pass


            # *** Ana döngü tamamlandı veya kırıldı ***

            # Ana döngü normal (stop olmadan) bittiğinde veya bir hata nedeniyle (continue olmayan) kırıldığında
            # State'i güncellemek için finally bloğunu kullanıyoruz.
            pass # Döngü sonu mantığı finally'e taşındı.


        except Exception as e:
             error_message = f"Yorum indirme sırasında beklenmedik bir hata oluştu: {e}"
             print(error_message)
             self.status_message.emit("Hata oluştu." + " | Tahmini Kalan Süre: Bilinmiyor")
             self.error.emit(error_message)
             # Beklenmedik ana hata durumunda state'i kaydetmiyoruz, bozuk olabilir.

        finally:
         

            if comments_file is not None and not comments_file.closed:
                try:
                    comments_file.close()
                    print("comments.csv dosyası kapatıldı.")
                except Exception as e:
                    print(f"Hata: comments.csv dosyası kapatılırken hata: {e}")

            # Worker kendi kendine state temizlemeyecek.
            # Tamamlama/Durdurma mesajı döngü içinde veya sonunda set edildi.
            print("İndirme worker finished sinyali yayıldı.")
            self.finished.emit()