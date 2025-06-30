import sys
import csv
import os
import glob
import json
from datetime import datetime
import time
import re
import pandas as pd

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QWidget, QLabel, QLineEdit, QPushButton,
                             QMessageBox, QSpinBox, QStyleFactory, QSizePolicy,
                             QFrame, QGridLayout, QScrollArea, QProgressBar,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer, pyqtSlot, QElapsedTimer, QCoreApplication
from PyQt6.QtGui import QIcon

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


import requests
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("Ollama kütüphanesi başarıyla yüklendi.")
except ImportError:
    print("Ollama kütüphanesi bulunamadı. Kategorizasyon özelliği devre devre dışı.")
    OLLAMA_AVAILABLE = False
    class CategorizationWorker(QObject):
        progress_updated = pyqtSignal(int, int, int, int) 
        plot_updated = pyqtSignal()
        categorization_finished = pyqtSignal()
        error_occurred = pyqtSignal(str)
        status_message = pyqtSignal(str)

        def __init__(self, input_csv_path, output_csv_path, comment_column, categories_data, ollama_model, batch_size):
             super().__init__()
             print("CategorizationWorker placeholder used. Ollama not available.")
             self.input_csv_path = input_csv_path
             self.output_csv_path = output_csv_path
             self.comment_column = comment_column
             self.categories_data = categories_data
             self.ollama_model = ollama_model
             self.batch_size = batch_size

             self.df = pd.DataFrame() 
             main_cat_names = [cat.get("category", f"Category_{i+1}") for i, cat in enumerate(categories_data) if cat.get("category", "").strip()]
             self.category_names = main_cat_names + ["Diğer"] if main_cat_names else ["Diğer"]
             self._main_category_names = main_cat_names
             self.total_rows = 0
             self.processed_count_at_start = 0
             self._is_running = False

        @pyqtSlot()
        def run_categorization(self):
             self.error_occurred.emit("Ollama kütüphanesi kurulu değil veya erişilemiyor. Kategorizasyon yapılamaz.");
             QTimer.singleShot(100, self.categorization_finished.emit)

        def stop(self):
             print("CategorizationWorker placeholder stop called.")
             self._is_running = False



try:
    from web_search import SearchWorker
    print("SearchWorker modülü yüklendi.")
except ImportError:
    print("web_search.py bulunamadı veya SearchWorker tanımı beklenenden farklı. Arama özelliği devre dışı olabilir.")
    class SearchWorker(QObject):
        search_finished = pyqtSignal(list)
        search_error = pyqtSignal(str)

        def __init__(self, query, limit):
            super().__init__()
            print("SearchWorker placeholder used.")
            self.query = query
            self.limit = limit
            self._is_running = False

        @pyqtSlot()
        def run(self):
             self._is_running = True
             print("SearchWorker placeholder: Simulating search...")
             QTimer.singleShot(500, lambda: self.search_error.emit("web_search.py modülü bulunamadı veya çalıştırılamadı."))
             self._is_running = False

        def stop(self):
             print("SearchWorker placeholder stop called.")
             self._is_running = False


try:
    from data_download import CommentDownloaderWorker, USE_COMMENT_DOWNLOADER
    if USE_COMMENT_DOWNLOADER:
         print("CommentDownloaderWorker modülü yüklendi ve aktif.")
    else:
         print("CommentDownloaderWorker modülü yüklendi ancak USE_COMMENT_DOWNLOADER False olarak ayarlanmış. Yorum indirme devre dışı.")
except ImportError:
    print("data_download.py bulunamadı veya CommentDownloaderWorker tanımı beklenenden farklı. Yorum indirme özelliği devre dışı.")
    USE_COMMENT_DOWNLOADER = False
    class CommentDownloaderWorker(QObject):
        status_message = pyqtSignal(str)
        finished = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, urls_filepath, comments_filepath, state_filepath, comment_limit_per_url):
             super().__init__()
             print("CommentDownloaderWorker placeholder used.")
             self.urls_filepath = urls_filepath
             self.comments_filepath = comments_filepath
             self.state_filepath = state_filepath
             self.comment_limit_per_url = comment_limit_per_url
             self._is_running = False

        @pyqtSlot()
        def run(self):
            self._is_running = True
            print("CommentDownloaderWorker placeholder: Simulating download error...")
            self.status_message.emit("Yorum indirme simülasyonu başlatıldı (Hata)...")
            QTimer.singleShot(500, lambda: self.error.emit("data_download.py modülü bulunamadı veya çalıştırılamadı."))
            self._is_running = False

        def stop(self):
            print("CommentDownloaderWorker placeholder stop called.")
            self._is_running = False


try:
    from bert_utils import perform_sentiment_analysis, SentimentAnalysisError
    print("bert_utils modülü yüklendi.")
    class SentimentAnalysisWorker(QObject):
        """BERT duygu analizi işlemini ayrı bir thread'de çalıştıran Worker."""
        finished = pyqtSignal(int, int)
        error = pyqtSignal(str)

        def __init__(self, csv_input_path, text_column_name, csv_output_path):
            super().__init__()
            self._csv_input_path = csv_input_path
            self._text_column_name = text_column_name
            self._csv_output_path = csv_output_path
            self._is_running = False

        @pyqtSlot()
        def run(self):
            self._is_running = True
            print("SentimentAnalysisWorker: Analiz başlatılıyor...")
            try:
                positive_count, negative_count = perform_sentiment_analysis(
                    self._csv_input_path,
                    self._text_column_name,
                    self._csv_output_path
                )
                if self._is_running:
                    print("SentimentAnalysisWorker: Analiz tamamlandı.")
                    self.finished.emit(positive_count, negative_count)
            except SentimentAnalysisError as e:
                if self._is_running:
                     print(f"SentimentAnalysisWorker: Analiz hatası: {e}")
                     self.error.emit(str(e))
            except Exception as e:
                if self._is_running:
                     print(f"SentimentAnalysisWorker: Beklenmedik hata: {e}")
                     self.error.emit(f"Beklenmedik analiz hatası: {e}")
            finally:
                 self._is_running = False
        def stop(self):
             print("SentimentAnalysisWorker stop called (not stoppable).")

except ImportError:
    print("bert_utils.py bulunamadı veya perform_sentiment_analysis tanımı beklenenden farklı. BERT analizi özelliği devre dışı.")
    class SentimentAnalysisError(Exception): pass
    def perform_sentiment_analysis(csv_input_path, text_column_name, csv_output_path):
        raise SentimentAnalysisError("bert_utils.py modülü veya perform_sentiment_analysis fonksiyonu beklendiği gibi değil.")
    
class SentimentAnalysisWorker(QObject):
    """BERT duygu analizi işlemini ayrı bir thread'de çalıştıran Worker."""
    finished = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, csv_input_path, text_column_name, csv_output_path):
        super().__init__()
        self._csv_input_path = csv_input_path
        self._text_column_name = text_column_name
        self._csv_output_path = csv_output_path
        self._is_running = False

    @pyqtSlot()
    def run(self):
        self._is_running = True
        print("SentimentAnalysisWorker: Analiz başlatılıyor...")
        try:
            positive_count, negative_count = perform_sentiment_analysis(
                self._csv_input_path,
                self._text_column_name,
                self._csv_output_path
            )
            if self._is_running:
                print("SentimentAnalysisWorker: Analiz tamamlandı.")
                self.finished.emit(positive_count, negative_count)
            else:
                print("SentimentAnalysisWorker: Analiz tamamlandı ancak işlem kullanıcı tarafından durdurulmuştu. Sonuç sinyali gönderilmiyor.")
        except SentimentAnalysisError as e:
            if self._is_running:
                 print(f"SentimentAnalysisWorker: Analiz hatası: {e}")
                 self.error.emit(str(e))
        except Exception as e:
            if self._is_running:
                 print(f"SentimentAnalysisWorker: Beklenmedik hata: {e}")
                 self.error.emit(f"Beklenmedik analiz hatası: {e}")
        finally:
             self._is_running = False # Her durumda bayrağı temizle

    def stop(self):
         print("SentimentAnalysisWorker: Durdurma sinyali alındı. Mevcut işlem bitince sonlanacak.")
         self._is_running = False

class ReportGenerationWorker(QObject):
    # progress_updated sinyaline artık ihtiyacımız yok.
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, csv_file_path, keyword, category_definitions):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.keyword = keyword
        self.category_definitions = category_definitions
        self._is_running = False # Durdurma isteğini kontrol etmek için

    @pyqtSlot()
    def run(self):
        self._is_running = True
        self.status_message.emit("Rapor oluşturma işlemi arka planda başlatıldı...")
        try:
            # Bu uzun süren, bloklayan çağrıdır.
            from rapor import YouTubeAnalysisReporter
            reporter = YouTubeAnalysisReporter(csv_file_path=self.csv_file_path,
                                             keyword=self.keyword,
                                             category_definitions=self.category_definitions)
            output_filename = f"output/{self.keyword.replace(' ', '_')}_analiz_raporu.docx"
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            reporter.generate_report(output_filename=output_filename)

            # Sadece işlem bittiğinde ve hala çalışıyorsa 'finished' sinyali gönder
            if self._is_running:
                self.finished.emit(output_filename)

        except ImportError:
            if self._is_running:
                self.error.emit("Rapor oluşturma modülü (rapor.py) bulunamadı.")
        except Exception as e:
            if self._is_running:
                self.error.emit(f"Rapor oluşturulurken hata: {str(e)}")
                import traceback
                traceback.print_exc()
        finally:
            self._is_running = False

    def stop(self):
        self._is_running = False
        self.status_message.emit("Rapor oluşturma durduruldu (mevcut işlem bittiğinde sonlanacak).")



if OLLAMA_AVAILABLE and 'CategorizationWorker' not in locals():
    class CategorizationWorker(QObject):
        progress_updated = pyqtSignal(int, int, int, int) 
        plot_updated = pyqtSignal() 
        categorization_finished = pyqtSignal()
        error_occurred = pyqtSignal(str)
        status_message = pyqtSignal(str)

        def __init__(self, input_csv_path, output_csv_path, comment_column, categories_data, ollama_model, batch_size):
            super().__init__()
            self.input_csv_path = input_csv_path
            self.output_csv_path = output_csv_path
            self.comment_column = comment_column
            self.categories_data = categories_data
            self.ollama_model = ollama_model
            self.batch_size = batch_size

            main_cat_names_raw = [cat.get("category", "").strip() for cat in categories_data]
            self._main_category_names = [name for name in main_cat_names_raw if name] 
            self.category_names = self._main_category_names + ["Diğer"] if self._main_category_names else ["Diğer"]
            self._main_category_desc_str = "\n".join([
                f"- {cat.get('category', f'Category_{i+1}')}: {cat.get('aciklama', '')}"
                for i, cat in enumerate(categories_data) if cat.get("category", "").strip() # Only include if name is not empty
            ]) or "Kategoriler: Hiç tanımlanmadı."


            self.client = None
            self.df = None

            self.total_rows = 0
            self.processed_count = 0 
            self.failed_count = 0 
            self.empty_comment_count = 0
            self.processed_count_at_start = 0 

            self._is_running = False # 


        def _initialize_ollama_client(self):
            """Ollama istemcisini başlatır ve erişilebilirlik kontrolü yapar."""
            self.status_message.emit("Ollama istemcisi başlatılıyor...")
            try:
                print("Ollama istemcisine bağlanılmaya çalışılıyor...")
                self.client = ollama.Client()
                
                response_object = self.client.list()
                print("Ollama istemcisi başarıyla bağlandı. Modellerin yanıt nesnesi:", response_object)

                models_list = [] 

                if isinstance(response_object, dict) and 'models' in response_object:
                    print("Yanıt bir sözlük formatında, 'models' anahtarı işleniyor.")
                    if isinstance(response_object.get('models'), list):
                        for model_dict in response_object.get('models', []):
                            if isinstance(model_dict, dict) and 'name' in model_dict:
                                class TempModel:
                                    def __init__(self, name):
                                        self.model = name 
                                models_list.append(TempModel(model_dict['name'])) 
                            else:
                                print(f"Uyarı: Sözlük formatındaki model listesinde beklenmedik formatta bir öğe: {model_dict}")
                    else:
                        msg = f"Ollama sunucusundan gelen sözlük formatındaki 'models' alanı bir liste değil. Yanıt: {response_object.get('models')}"
                        print("Ollama Başlatma Hatası:", msg)
                        self.error_occurred.emit(msg)
                        self.client = None
                        return False
                elif hasattr(response_object, 'models') and isinstance(response_object.models, list):
                    print("Yanıt bir nesne formatında, 'models' özelliği işleniyor.")
                    models_list = response_object.models 
                else:
                    msg = f"Ollama sunucusundan beklenmedik model listesi yanıt formatı alındı. Yanıt: {response_object}"
                    print("Ollama Başlatma Hatası:", msg)
                    self.error_occurred.emit(msg)
                    self.client = None
                    return False

                model_names = []
                for model_obj in models_list:
                    if hasattr(model_obj, 'model') and isinstance(model_obj.model, str):
                        model_names.append(model_obj.model)
                    elif hasattr(model_obj, 'name') and isinstance(model_obj.name, str):
                        model_names.append(model_obj.name)
                    elif isinstance(model_obj, dict) and 'name' in model_obj:
                         model_names.append(model_obj['name'])
                    else:
                        print(f"Uyarı: Model listesinde 'model' veya 'name' özelliği olmayan bir öğe: {model_obj}")
                
                print("Sunucudaki mevcut Ollama modelleri:", model_names)

                if not model_names and not models_list: 
                    print("Uyarı: Sunucuda hiç model bulunamadı veya model listesi işlenemedi.")

                if self.ollama_model not in model_names:
                     msg = f"Ollama modeli '{self.ollama_model}' bulunamadı. Lütfen 'ollama pull {self.ollama_model}' komutu ile modeli indirin veya doğru model adını girin. Sunucudaki modeller: {model_names}"
                     print("Ollama Başlatma Hatası:", msg)
                     self.error_occurred.emit(msg)
                     self.client = None
                     return False

                
                print("Ollama istemcisi başarıyla başlatıldı.")
                return True
            except ollama.ResponseError as e:
                msg = f"Ollama API Yanıt Hatası: {e}. Ollama sunucusu çalışıyor ve erişilebilir mi?"
                print("Ollama Başlatma Hatası (ResponseError):", msg)
                self.error_occurred.emit(msg)
                self.client = None
                return False
            except requests.exceptions.ConnectionError as e:
                msg = f"Ollama sunucusuna bağlanılamadı: {e}. Ollama sunucusu çalışıyor mu ve ağ ayarları doğru mu?"
                print("Ollama Başlatma Hatası (ConnectionError):", msg)
                self.error_occurred.emit(msg)
                self.client = None
                return False
            except Exception as e:
                msg = f"Ollama istemcisi başlatılırken veya modellere erişilirken genel hata. Ollama sunucusu çalışıyor mu? Hata Türü: {type(e).__name__}, Hata: {e}"
                print("Ollama Başlatma Hatası (Genel İstisna):", msg)
                import traceback
                traceback.print_exc()
                self.error_occurred.emit(msg)
                self.client = None
                return False

        def _get_batch_scores_from_ollama(self, comments_batch, batch_df_indices):
            """
            Sends a batch of comments to the Ollama model for categorization and parses the response.
            Returns a list of score lists (0/1) for each comment in the batch.
            Returns None for a comment if processing fails or parsing is unsuccessful for that comment.

            Args:
                comments_batch (list): A list of comment texts (strings) to process.
                batch_df_indices (list): A list of original DataFrame indices corresponding to the comments_batch.

            Returns:
                list of (list of int or None): A list where each element corresponds to a comment
                                               in comments_batch. It's either a list of 0/1 scores
                                               for main categories or None if parsing failed for that comment.
                                               The outer list always has the same length as comments_batch.
            """
            if not self.client or not self._main_category_names:
                 print(f"DEBUG: _get_batch_scores_from_ollama: Client not initialized or no main categories defined. Returning None for batch starting at DF index {batch_df_indices[0] if batch_df_indices else -1}")
                 return [None] * len(comments_batch)


            batch_prompt = f"""
Aşağıda {len(comments_batch)} adet yorum ve bir kategori listesi (isim ve açıklama ile) verilmiştir.
Her bir yorum için (numarası ile belirtilmiş, 0'dan başlar), yorumun her kategoriye UYGUN olup olmadığını belirleyin.
Açıklamalardan yararlanarak en doğru eşleşmeyi yapın.
Bir yorum birden fazla kategoriye uygun olabilir.
Eğer yorum belirtilen kategorilerin HİÇBİRİNE uygun değilse, onu "Diğer" diye düşünün (çıktıda bu durum 0 skorları ile gösterilecektir).
Çıktı formatı, yorum numarasını takiben ana kategorilerin 0/1 skorlarını içeren virgülle ayrılmış bir liste olmalıdır.

Her yorumun sonucu için:
"Comment [Yorum Numarası]: " şeklinde başlayın.
Ardından {len(self._main_category_names)} adet 0 veya 1 değerinden oluşan, virgülle ayrılmış bir liste döndürün.
0 = UYGUN DEĞİL, 1 = UYGUN.
Döndürdüğünüz 0/1 listesinin sırası, size verdiğim kategori listesinin sırasıyla AYNI OLMALIDIR.
YANLIZCA istenen formattaki çıktıları sağlayın. Başka hiçbir metin, açıklama veya ek karakter İÇERMEYİN. SADECE "Comment X: 0,1,0,..." formatını kullanın.

Kategoriler (İsim: Açıklama):
{self._main_category_desc_str}

Yorumlar:
"""
            for i, comment in enumerate(comments_batch):
                cleaned_comment = str(comment or '').strip().replace('\n', ' ').replace('\r', ' ')
                max_comment_len = 500 
                cleaned_comment = cleaned_comment[:max_comment_len]
                batch_prompt += f"Comment {i}: {cleaned_comment}\n"

            batch_prompt += "\nOutput:"

            batch_scores_results = [None] * len(comments_batch) 

            try:
                estimated_output_tokens = len(comments_batch) * (len(self._main_category_names) * 2 + 10)

                response = self.client.generate(
                    model=self.ollama_model,
                    prompt=batch_prompt,
                    options={'temperature': 0.0, 'num_predict': max(256, estimated_output_tokens)}
                )

                raw_output = response.get('response', '').strip()


                pattern = re.compile(r'Comment\s*(\d+):\s*([01]+(?:\s*,\s*[01]+)*)')


                parsed_count = 0
                for match in pattern.finditer(raw_output):
                     try:
                         comment_idx_in_batch = int(match.group(1)) 
                         scores_str = match.group(2) 

                         current_scores = [int(s.strip()) for s in scores_str.split(',') if s.strip() in ['0', '1']]

                         if len(current_scores) == len(self._main_category_names):
                             if comment_idx_in_batch >= 0 and comment_idx_in_batch < len(comments_batch):
                                  batch_scores_results[comment_idx_in_batch] = current_scores
                                  parsed_count += 1
                             else:
                                  print(f"Warning (DF Index Batch Start {batch_df_indices[0] if batch_df_indices else -1}): LLM returned invalid comment index ({comment_idx_in_batch}). Batch size: {len(comments_batch)}. Raw Output Start: '{raw_output[:100]}...'")
                         else:
                             print(f"Warning (DF Index Batch Start {batch_df_indices[0] if batch_df_indices else -1}): LLM returned unexpected number of scores for batch index {comment_idx_in_batch}. Expected: {len(self._main_category_names)}, Got: {len(current_scores)}. Raw Output Start: '{raw_output[:100]}...'")

                     except ValueError as e: 
                          print(f"Warning (DF Index Batch Start {batch_df_indices[0] if batch_df_indices else -1}): Parsing ValueError for batch index {match.group(1)}: {e}. Scores string: '{match.group(2)}'. Raw Output Start: '{raw_output[:100]}...'")
                     except Exception as e:
                          print(f"Warning (DF Index Batch Start {batch_df_indices[0] if batch_df_indices else -1}): Unexpected parsing error for batch index {match.group(1)}: {e}. Raw Output Start: '{raw_output[:100]}...'")


                return batch_scores_results

            except ollama.ResponseError as e:
                msg = f"Ollama API hatası (batch başlangıç index {batch_df_indices[0] if batch_df_indices else -1}): {e}. Sunucu çalışıyor mu? Model adı '{self.ollama_model}' doğru mu?"
                print("Ollama API Error:", msg)
                self.status_message.emit(msg)
                if "model '" in str(e) and "' not found" in str(e):
                     self.error_occurred.emit(f"Kritik Ollama Hatası: {e}. Lütfen modeli indirin veya doğru model adını kullanın.")
                return [None] * len(comments_batch)
            except Exception as e:
                msg = f"Ollama çağrısı sırasında beklenmedik bir hata (batch başlangıç index {batch_df_indices[0] if batch_df_indices else -1}): {e}"
                print("Ollama Call Error:", msg)
                self.status_message.emit(msg)
                self.error_occurred.emit(msg)
                return [None] * len(comments_batch)


        @pyqtSlot()
        def run_categorization(self):
            """
            Kategorizasyon işlemini ayrı bir iş parçacığında yürütür.
            Sentiment.csv dosyasını okur, kategorize eder ve categorization.csv dosyasına yazar.
            Bu versiyon kritik hataları ve mantıksal sorunları düzeltir.
            """
            self._is_running = True
            self.status_message.emit(f"Kategorizasyon: Başlatılıyor...")
            print(f"Kategorizasyon worker başlatıldı. Giriş: {self.input_csv_path}, Çıkış: {self.output_csv_path}")

            try:
                if os.path.exists(self.output_csv_path):
                    self.df = pd.read_csv(self.output_csv_path, encoding='utf-8', low_memory=False)
                    self.status_message.emit(f"Kategorizasyon: Mevcut çıktı dosyası '{self.output_csv_path}' yükleniyor.")
                    print(f"Kategorizasyon: Mevcut çıktı dosyası '{self.output_csv_path}' yüklendi.")
                elif os.path.exists(self.input_csv_path):
                    self.df = pd.read_csv(self.input_csv_path, encoding='utf-8', low_memory=False)
                    self.status_message.emit(f"Kategorizasyon: Giriş dosyası '{self.input_csv_path}' yükleniyor.")
                    print(f"Kategorizasyon: Giriş dosyası '{self.input_csv_path}' yüklendi.")
                else:
                    self.error_occurred.emit(f"Hata: Giriş veya devam dosyası bulunamadı: {self.input_csv_path} veya {self.output_csv_path}")
                    return

                if self.df.empty:
                    self.status_message.emit("Kategorizasyon: Giriş dosyası boş.")
                    return

                if self.comment_column not in self.df.columns:
                    self.error_occurred.emit(f"Hata: Giriş dosyasında yorum sütunu ('{self.comment_column}') bulunamadı.")
                    return

                if not self._main_category_names:
                    self.error_occurred.emit("Kategorizasyon: Tanımlanmış geçerli ana kategori adı yok. İşlem iptal edildi.")
                    return
                
                self.total_rows = len(self.df)
                cols_to_manage = self.category_names

                for category_name in cols_to_manage:
                    if category_name not in self.df.columns:
                        self.df[category_name] = -1
                    else:
                        self.df[category_name] = pd.to_numeric(self.df[category_name], errors='coerce').fillna(-1).astype(int)

                unprocessed_mask = (self.df[cols_to_manage] == -1).all(axis=1)
                unprocessed_indices_overall = self.df.index[unprocessed_mask].tolist()
                total_unprocessed_in_run = len(unprocessed_indices_overall)

                if total_unprocessed_in_run == 0:
                    self.status_message.emit("İşlenecek yeni yorum bulunmadı.")
                    self.progress_updated.emit(self.total_rows, self.total_rows, 0, 0)
                    return

                self.processed_count_at_start = self.total_rows - total_unprocessed_in_run
                self.failed_count = 0
                self.empty_comment_count = 0
                
                self.status_message.emit(f"Kategorizasyon: '{self.ollama_model}' modeli kullanılarak {total_unprocessed_in_run} yorum işlenecek.")
                
                if not self._initialize_ollama_client():
                    return 

                self.progress_updated.emit(self.total_rows, self.processed_count_at_start, 0, 0)
                self.plot_updated.emit()
                
                current_unprocessed_index_in_list = 0
                while current_unprocessed_index_in_list < total_unprocessed_in_run and self._is_running:
                    batch_df_indices = unprocessed_indices_overall[current_unprocessed_index_in_list : current_unprocessed_index_in_list + self.batch_size]
                    
                    comments_batch_series = self.df.loc[batch_df_indices, self.comment_column]
                    valid_comments_in_batch = []
                    valid_comments_batch_df_indices = []

                    for original_df_index, comment in comments_batch_series.items():
                        if pd.isna(comment) or not str(comment).strip():
                            for cat_name in cols_to_manage: self.df.at[original_df_index, cat_name] = 0 
                            self.empty_comment_count += 1
                        else:
                            valid_comments_in_batch.append(str(comment))
                            valid_comments_batch_df_indices.append(original_df_index)

                    if valid_comments_in_batch and self._is_running:
                        batch_scores_list = self._get_batch_scores_from_ollama(valid_comments_in_batch, valid_comments_batch_df_indices)

                        for batch_valid_idx, scores_or_none in enumerate(batch_scores_list):
                            original_df_index = valid_comments_batch_df_indices[batch_valid_idx]

                            if scores_or_none and len(scores_or_none) == len(self._main_category_names):
                                main_scores = scores_or_none
                                for i, cat_name in enumerate(self._main_category_names):
                                    self.df.at[original_df_index, cat_name] = main_scores[i]
                                
                                entered_any_main_category = any(score == 1 for score in main_scores)
                                self.df.at[original_df_index, "Diğer"] = 0 if entered_any_main_category else 1
                            else:
                                for cat_name in self._main_category_names: self.df.at[original_df_index, cat_name] = 0
                                self.df.at[original_df_index, "Diğer"] = 1
                                self.failed_count += 1

                    current_unprocessed_index_in_list += len(batch_df_indices)
                    processed_mask_after_batch = (self.df[cols_to_manage] != -1).any(axis=1)
                    self.processed_count = processed_mask_after_batch.sum()

                    self.progress_updated.emit(self.total_rows, self.processed_count, self.failed_count, self.empty_comment_count)

                    if not self.__save_results():
                        print("Kategorizasyon: Periyodik kaydetme başarısız oldu. İşlem durduruluyor.")
                        self.error_occurred.emit("Periyodik kaydetme sırasında hata oluştu.")
                        break # Döngüden çık

            except Exception as e:
                msg = f"Kategorizasyon sırasında beklenmedik bir hata: {e}"
                print(f"Kategorizasyon Hatası: {msg}")
                import traceback
                traceback.print_exc()
                self.error_occurred.emit(msg)
            finally:
                print("Kategorizasyon döngüsü tamamlandı veya durduruldu. Son temizlik yapılıyor.")
                
                if self._is_running:
                    self.status_message.emit("İşlem tamamlandı. Sonuçlar kaydediliyor.")
                else:
                    self.status_message.emit("İşlem durduruldu. Son sonuçlar kaydediliyor.")
                
                self.__save_results()  # Son bir kez kaydet
                self.plot_updated.emit() # Grafikleri son haliyle güncelle
                
                self._is_running = False # Durumu her zaman temizle
                self.categorization_finished.emit() # GUI'ye işin bittiğini haber ver
                
                print("Kategorizasyon worker 'run' metodu sonlanıyor.")




        def __save_results(self):
             """Saves the current DataFrame to the output CSV path."""
             if self.df is None:
                  print("Kategorizasyon: Kaydedilecek DataFrame yok.")
                  self.status_message.emit(f"Kategorizasyon: DataFrame boş olduğu için sonuç dosyası kaydedilmedi.")
                  return False 

             self.status_message.emit(f"Kategorizasyon: Sonuçlar '{self.output_csv_path}' dosyasına kaydediliyor...")

             try:
                 output_dir = os.path.dirname(self.output_csv_path)
                 if output_dir and not os.path.exists(output_dir):
                     os.makedirs(output_dir)
                     print(f"DEBUG: Output directory created: {output_dir}")

                 cols_to_manage_on_save = self.category_names
                 for cat_col in cols_to_manage_on_save:
                      if cat_col in self.df.columns:
                           self.df[cat_col] = pd.to_numeric(self.df[cat_col], errors='coerce').fillna(-1).astype(int)
                      else:
                           print(f"Warning: Managed category column '{cat_col}' missing during save. Adding with -1.") 
                           self.df[cat_col] = -1 
                           self.df[cat_col] = self.df[cat_col].astype(int) 


               

                 known_analysis_cols = ['duygu_tahmini', 'duygu_skoru', 'hesaplanan_tarih'] + cols_to_manage_on_save

                 original_cols_guess = [col for col in self.df.columns if col not in known_analysis_cols]
                 preferred_original_start = ['text', 'URL', 'video_id', 'comment_id']
                 ordered_original_cols = [col for col in preferred_original_start if col in original_cols_guess] + \
                                         [col for col in original_cols_guess if col not in preferred_original_start]

                 sentiment_cols_ordered = [col for col in ['duygu_tahmini', 'duygu_skoru', 'hesaplanan_tarih'] if col in self.df.columns]

                 category_cols_ordered = [col for col in self.category_names if col in self.df.columns]


                 final_cols_order_template = ordered_original_cols + sentiment_cols_ordered + category_cols_ordered
                 cols_to_save = final_cols_order_template + [col for col in self.df.columns if col not in final_cols_order_template]


                 df_to_save = self.df.reindex(columns=cols_to_save)

                 df_to_save.to_csv(self.output_csv_path, index=False, encoding='utf-8-sig')

                 self.status_message.emit(f"Kategorizasyon: Sonuçlar Başarıyla Kaydediliyor.")
                 print(f"DEBUG: Kategorizasyon: '{self.output_csv_path}' başarıyla kaydedildi.") 
                 return True
             except Exception as e:
                 msg = f"Kategorizasyon: Çıktı dosyası kaydedilirken hata: {e}"
                 print("Save Error:", msg)
                 self.status_message.emit(msg) 
                 self.error_occurred.emit(msg)
                 return False 


        def stop(self):
            self.status_message.emit("Kategorizasyon işlemi durduruluyor...")
            self._is_running = False
            print("CategorizationWorker stop sinyali alındı.")

class YoutubeToolGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("YtScraperS")
        self.setGeometry(100, 100, 1400, 900)

        self.setWindowIcon(QIcon("logo.png"))

        self._is_collecting: bool = False
        self._is_analyzing: bool = False
        self._is_categorizing: bool = False
        self._all_steps_completed_for_query: bool = False

        self._comments_collected: bool = False
        self._analysis_completed: bool = False
        self._categorization_completed: bool = False
        self._is_reporting = False 

        self._safe_query_name: str = ""
        self._urls_filepath: str = ""
        self._comments_filepath: str = ""
        self._state_filepath: str = ""
        self._sentiment_filepath: str = ""
        self._categorization_filepath: str = ""

        self._resume_available: bool = False
        self._report_completed = False 

        self.search_thread: QThread = None
        self.search_worker: SearchWorker = None
        self.download_thread: QThread = None
        self.download_worker: CommentDownloaderWorker = None
        self.sentiment_thread: QThread = None
        self.sentiment_worker: SentimentAnalysisWorker = None
        self._categorization_thread: QThread = None
        self._categorization_worker: CategorizationWorker = None

        self._bert_timer: QTimer = None # 
        self._bert_analysis_start_time: float = None
        self._bert_estimated_total_seconds: int = 0
        self._BERT_COMMENTS_PER_SECOND: float = 5.0 

        self._categorization_plot_timer: QTimer = None
        self._CATEGORIZATION_PLOT_UPDATE_INTERVAL_MS = 1000 


        self._categorization_start_time: float = None 


       
        self._sentiment_pie_figure = None
        self._sentiment_pie_canvas = None
        self._sentiment_line_figure = None
        self._sentiment_line_canvas = None

        self._categorization_table_widget: QTableWidget = None 

        self._category_sentiment_grid_layout: QGridLayout = None 
        self._category_sentiment_figures = [] 
        self._category_sentiment_canvases = [] 

        self.step1_button = None
        self.step2_button = None
        self.step3_button = None
        self.step4_button = None 
        self.line_label_1 = None 
        self.line_label_2 = None 
        self.line_label_3 = None
        self._step_buttons = []
        self._is_shutting_down = False

        # Raporlama için yeni GUI tabanlı zamanlayıcı
        self.report_timer = QTimer(self)
        self.report_timer.timeout.connect(self._update_report_progress_display)
        self.report_start_time = None
        self.REPORT_ESTIMATED_DURATION = 15 * 60

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        centering_layout = QHBoxLayout(central_widget)
        centering_layout.addStretch(1)

        content_container_widget = QWidget()
        content_container_widget.setFixedWidth(1100)
        centering_layout.addWidget(content_container_widget)
        centering_layout.addStretch(1)

        self.main_layout = QVBoxLayout(content_container_widget) 
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        top_fixed_widget = QWidget()
        top_fixed_layout = QVBoxLayout(top_fixed_widget)
        top_fixed_layout.setContentsMargins(0, 0, 0, 0)
        top_fixed_layout.setSpacing(10)

        input_layout = QVBoxLayout()
        input_layout.setSpacing(5)

        query_layout = QHBoxLayout()
        query_label = QLabel("Aranacak Kelime:")
        query_label.setMinimumWidth(150)
        query_layout.addWidget(query_label)
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Örn: Python programlama dersleri")
        query_layout.addWidget(self.query_input)
        input_layout.addLayout(query_layout)

        url_limit_layout = QHBoxLayout()
        url_limit_label = QLabel("Aranacak URL Sayısı:")
        url_limit_label.setMinimumWidth(150)
        url_limit_layout.addWidget(url_limit_label)
        self.search_limit_spinbox = QSpinBox()
        self.search_limit_spinbox.setMinimum(1)
        self.search_limit_spinbox.setMaximum(500)
        self.search_limit_spinbox.setValue(20)
        url_limit_layout.addWidget(self.search_limit_spinbox)
        input_layout.addLayout(url_limit_layout)

        comment_limit_layout = QHBoxLayout()
        comment_limit_label = QLabel("Her URL'den Yorum Sayısı:")
        comment_limit_label.setMinimumWidth(150)
        comment_limit_layout.addWidget(comment_limit_label)
        self.comment_limit_spinbox = QSpinBox()
        self.comment_limit_spinbox.setMinimum(1)
        self.comment_limit_spinbox.setMaximum(20000)
        self.comment_limit_spinbox.setValue(200)
        comment_limit_layout.addWidget(self.comment_limit_spinbox)
        input_layout.addLayout(comment_limit_layout)
        top_fixed_layout.addLayout(input_layout)

        buttons_container_layout = QHBoxLayout()
        buttons_container_layout.setSpacing(15)

        self.step_buttons_layout = QHBoxLayout() 
        self.step_buttons_layout.setSpacing(10)
        self.step_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.step1_button = QPushButton("1. Veri Topla")
        self.step1_button.setObjectName("step1_button") 
        self._step_buttons.append(self.step1_button)
        self.step_buttons_layout.addWidget(self.step1_button)
        self.step1_button.clicked.connect(self.start_process)

        self.line_label_1 = QLabel(">")
        self.line_label_1.setObjectName("line_label_1") 
        self.line_label_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_buttons_layout.addWidget(self.line_label_1)

        self.step2_button = QPushButton("2. Duygu Analizi)")
        self.step2_button.setObjectName("step2_button")
        self._step_buttons.append(self.step2_button)
        self.step_buttons_layout.addWidget(self.step2_button)
        self.step2_button.clicked.connect(self.start_sentiment_analysis) 


        self.line_label_2 = QLabel(">")
        self.line_label_2.setObjectName("line_label_2")
        self.line_label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_buttons_layout.addWidget(self.line_label_2)

        self.step3_button = QPushButton("3. Kategorize Et (Ollama)")
        self.step3_button.setObjectName("step3_button") 
        self._step_buttons.append(self.step3_button)
        self.step_buttons_layout.addWidget(self.step3_button)
        self.step3_button.clicked.connect(self.start_categorization)

        self.line_label_3 = QLabel(">")
        self.line_label_3.setObjectName("line_label_3") 
        self.line_label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_buttons_layout.addWidget(self.line_label_3)

        self.step4_button = QPushButton("4. Rapor Oluştur")
        self.step4_button.setObjectName("step4_button") 
        self._step_buttons.append(self.step4_button)
        self.step_buttons_layout.addWidget(self.step4_button)
        self.step4_button.clicked.connect(self.start_report_generation)

        buttons_container_layout.addLayout(self.step_buttons_layout)
        buttons_container_layout.addStretch(1) 

        control_buttons_layout = QHBoxLayout()
        control_buttons_layout.setSpacing(10)
        control_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.stop_button = QPushButton("Durdur")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.clicked.connect(self.stop_current_process)
        control_buttons_layout.addWidget(self.stop_button)

        self.clear_state_button = QPushButton("Temizle")
        self.clear_state_button.setObjectName("clear_button")
        self.clear_state_button.clicked.connect(self.clear_state_and_files)
        control_buttons_layout.addWidget(self.clear_state_button)

        buttons_container_layout.addLayout(control_buttons_layout)
        top_fixed_layout.addLayout(buttons_container_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("global_progress_bar")
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        top_fixed_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setObjectName("status_label")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.status_label.setWordWrap(True)
        top_fixed_layout.addWidget(self.status_label)

        self.main_layout.addWidget(top_fixed_widget)
        top_fixed_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVisible(False)

        scrollable_content_widget = QWidget()
        self.scroll_area.setWidget(scrollable_content_widget)
        self.scrollable_content_layout = QVBoxLayout(scrollable_content_widget)
        self.scrollable_content_layout.setContentsMargins(0, 0, 0, 0)
        self.scrollable_content_layout.setSpacing(20)
        self.main_layout.addWidget(self.scroll_area)

        self.sentiment_charts_group = QGroupBox("Duygu Analizi Grafikleri")
        self.sentiment_charts_layout = QVBoxLayout(self.sentiment_charts_group)
        self.sentiment_charts_layout.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.sentiment_charts_layout.setContentsMargins(10, 10, 10, 10)
        self.sentiment_charts_layout.setSpacing(20)
        self.sentiment_charts_group.setVisible(False)
        self.scrollable_content_layout.addWidget(self.sentiment_charts_group)

        self.category_settings_group = QGroupBox("Yapay Zeka Destekli Kategorizasyon Ayarları")
        category_settings_layout = QVBoxLayout(self.category_settings_group)
        category_settings_layout.setContentsMargins(10, 10, 10, 10)
        category_settings_layout.setSpacing(10)
        self.category_settings_group.setVisible(False) 
        self.scrollable_content_layout.addWidget(self.category_settings_group)

        category_input_grid_layout = QGridLayout()
        category_input_grid_layout.setSpacing(10)
        self.category_inputs = []
        num_categories = 4 
        for i in range(num_categories):
             cat_name_label = QLabel(f"Kategori {i+1} Adı:")
             cat_desc_label = QLabel(f"Kategori {i+1} Açıklaması:")
             cat_name_input = QLineEdit()
             cat_desc_input = QLineEdit()
             cat_name_input.setPlaceholderText(f"Örn: Teknik Sorunlar")
             cat_desc_input.setPlaceholderText(f"Örn: Kullanıcıların yaşadığı teknik problemler veya hatalar")
             category_input_grid_layout.addWidget(cat_name_label, i, 0)
             category_input_grid_layout.addWidget(cat_name_input, i, 1)
             category_input_grid_layout.addWidget(cat_desc_label, i, 2)
             category_input_grid_layout.addWidget(cat_desc_input, i, 3)
             self.category_inputs.append({"name_input": cat_name_input, "desc_input": cat_desc_input})
        category_settings_layout.addLayout(category_input_grid_layout)

        self.categorization_results_group = QGroupBox("Kategorizasyon Sonuçları")
        self.categorization_results_layout = QVBoxLayout(self.categorization_results_group)
        self.categorization_results_layout.setContentsMargins(10,10,10,10)
        self.categorization_results_layout.setSpacing(10)
        self.categorization_results_group.setVisible(False)
        self.scrollable_content_layout.addWidget(self.categorization_results_group)

       

        self.categorization_status_label = QLabel("Bekleniyor...")
        self.categorization_status_label.setObjectName("categorization_status_label")
        self.categorization_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.categorization_results_layout.addWidget(self.categorization_status_label)

        self.categorization_table_container_layout = QVBoxLayout()
        self.categorization_table_container_layout.setContentsMargins(0,0,0,0)
        self.categorization_table_container_layout.setSpacing(0)
        self.categorization_results_layout.addLayout(self.categorization_table_container_layout)


        self.sentiment_by_category_group = QGroupBox("Kategorilere Göre Duygu Dağılımı (Zaman Serisi)")
        self.sentiment_by_category_layout = QVBoxLayout(self.sentiment_by_category_group)
        self.sentiment_by_category_layout.setContentsMargins(10, 10, 10, 10)
        self.sentiment_by_category_layout.setSpacing(10)
        self.sentiment_by_category_group.setVisible(False) 
        self.scrollable_content_layout.addWidget(self.sentiment_by_category_group)

        self._category_sentiment_grid_layout = QGridLayout()
        self._category_sentiment_grid_layout.setSpacing(15)
        self.sentiment_by_category_layout.addLayout(self._category_sentiment_grid_layout)


        self.scrollable_content_layout.addStretch(1)

        self._bert_timer = QTimer(self)
        self._bert_timer.timeout.connect(self._update_bert_analysis_timer)

        self._categorization_plot_timer = QTimer(self)
        self._categorization_plot_timer.timeout.connect(self._periodic_plot_update)


        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.check_resume_state()


        self.setStyleSheet(QSS_STYLE)

    def _create_line(self):
        """Creates a horizontal line QFrame for the step indicator."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setFixedHeight(3) 
        line.setFixedWidth(50) 
        return line


    def _get_potential_category_names_from_inputs(self):
        """Collects non-empty category names from GUI input fields."""
        names = []
        for category_dict in self.category_inputs:
            name = category_dict["name_input"].text().strip()
            if name:
                names.append(name)
        return names

    def _get_all_managed_category_names(self):
        """Returns a list of all category names managed by the GUI, including 'Diğer'."""
        main_names = self._get_potential_category_names_from_inputs()
        return main_names + ["Diğer"] if main_names else [] 
    def _get_processed_categorization_df(self):
         """
         Retrieves the current categorization DataFrame (from worker or file).
         Identifies relevant category columns in the DataFrame.
         Filters for rows that have been processed (any managed category column is not -1).
         Ensures sentiment and date columns are in usable formats and filters rows with NaNs in these.

         Returns:
             tuple: (filtered_dataframe, list_of_category_column_names_found_in_df)
         """
         df_source = None
         category_names_in_df = [] 
         if self._is_categorizing and self._categorization_worker is not None and self._categorization_worker.df is not None:
             df_source = self._categorization_worker.df.copy() 
             category_names_to_check = self._categorization_worker.category_names if hasattr(self._categorization_worker, 'category_names') else []

         elif hasattr(self, '_categorization_filepath') and self._categorization_filepath and os.path.exists(self._categorization_filepath):
             try:
                 df_source = pd.read_csv(self._categorization_filepath, encoding='utf-8', low_memory=False)
                 known_non_category_cols = ['text', 'URL', 'video_id', 'comment_id',
                                            'duygu_tahmini', 'duygu_skoru', 'hesaplanan_tarih'] 
                 potential_cat_cols_in_file = [col for col in df_source.columns if col not in known_non_category_cols]

                 category_names_to_check = []
                 for col in potential_cat_cols_in_file:
                     if pd.api.types.is_numeric_dtype(df_source[col]):
                          unique_values = df_source[col].dropna().unique()
                          is_category_like = all(pd.api.types.is_integer_dtype(v) and -1 <= v <= 1 for v in unique_values if pd.notna(v))
                          if is_category_like or (unique_values.size == 0 and df_source[col].isnull().all()):
                               category_names_to_check.append(col)
     
                 if "Diğer" in df_source.columns and "Diğer" not in category_names_to_check:
                     category_names_to_check.append("Diğer")


             except Exception as e:
                 print(f"ERROR: _get_processed_categorization_df: Failed to load or inspect file '{self._categorization_filepath}': {e}")
                 df_source = None 


         if df_source is None or df_source.empty:
             return pd.DataFrame(), [] 
         category_names_in_df = [col for col in category_names_to_check if col in df_source.columns]

         if not category_names_in_df:
           
             return pd.DataFrame(), []


       
         for col in category_names_in_df:
              df_source[col] = pd.to_numeric(df_source[col], errors='coerce').fillna(-1).astype(int)

         processed_mask = (df_source[category_names_in_df] != -1).any(axis=1)
         df_processed = df_source.loc[processed_mask].copy()
         required_sentiment_cols = ['duygu_tahmini', 'duygu_skoru', 'hesaplanan_tarih']
         existing_sentiment_cols = [col for col in required_sentiment_cols if col in df_processed.columns]

         if existing_sentiment_cols:
             if 'hesaplanan_tarih' in df_processed.columns:
                 df_processed['hesaplanan_tarih'] = pd.to_datetime(df_processed['hesaplanan_tarih'], errors='coerce')

             if 'duygu_skoru' in df_processed.columns:
                 df_processed['duygu_skoru'] = pd.to_numeric(df_processed['duygu_skoru'], errors='coerce')

             cols_to_dropna = [col for col in ['hesaplanan_tarih', 'duygu_skoru'] if col in df_processed.columns] # Only drop if these cols exist
             if cols_to_dropna:
                  df_processed.dropna(subset=cols_to_dropna, inplace=True)

             if 'duygu_tahmini' in df_processed.columns:
                  df_processed = df_processed[df_processed['duygu_tahmini'].fillna('').str.strip() != ''].copy() # Filter and copy


         if df_processed.empty:
             return pd.DataFrame(), []
         return df_processed, category_names_in_df


    def clear_sentiment_charts(self):
        """Clears BERT sentiment analysis chart components."""
        while self.sentiment_charts_layout.count():
             item = self.sentiment_charts_layout.takeAt(0)
             if item.widget():
                  item.widget().deleteLater()
             elif item.layout():
                  self._clear_layout(item.layout())
                  item.layout().deleteLater()
             del item 

        if self._sentiment_pie_figure is not None:
            plt.close(self._sentiment_pie_figure)
            self._sentiment_pie_figure = None

        if self._sentiment_line_figure is not None:
            plt.close(self._sentiment_line_figure)
            self._sentiment_line_figure = None

        self._sentiment_pie_canvas = None
        self._sentiment_line_canvas = None


    def clear_categorization_table(self):
        """Clears Ollama categorization table component."""
        while self.categorization_table_container_layout.count():
            item = self.categorization_table_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                 self._clear_layout(item.layout())
                 item.layout().deleteLater()
            del item

        self._categorization_table_widget = None

        if hasattr(self, '_categorization_pie_figure') and self._categorization_pie_figure is not None:
            plt.close(self._categorization_pie_figure)
            self._categorization_pie_figure = None
            self._categorization_pie_canvas = None


    def clear_sentiment_by_category_charts(self):
         """Clears category-specific sentiment chart components."""
         if self._category_sentiment_grid_layout is not None:
              while self._category_sentiment_grid_layout.count():
                  item = self._category_sentiment_grid_layout.takeAt(0)
                  if item.widget():
                      item.widget().deleteLater()
                  elif item.layout():
                       self._clear_layout(item.layout())
                       item.layout().deleteLater()
                  del item

         for fig in self._category_sentiment_figures:
              if fig is not None:
                   plt.close(fig)

         self._category_sentiment_figures = []
         self._category_sentiment_canvases = []


    def clear_all_charts(self):
        """Clears all chart and table components."""
        self.clear_sentiment_charts()
        self.clear_categorization_table()
        self.clear_sentiment_by_category_charts()

    def _clear_layout(self, layout):
         """Helper function to clear widgets and items from a layout."""
         if layout is not None:
              while layout.count():
                   item = layout.takeAt(0)
                   if item.widget():
                       item.widget().deleteLater()
                   elif item.layout():
                       self._clear_layout(item.layout())
                       item.layout().deleteLater()
                   del item


    def show_existing_results_areas(self):
        """Manages the visibility of the results and settings GroupBoxes based on completed steps."""
        scroll_visible = False

        if self._analysis_completed or self._is_analyzing:
             self.sentiment_charts_group.setVisible(True)
             scroll_visible = True
        else:
             self.sentiment_charts_group.setVisible(False)
             self.clear_sentiment_charts()
        if self._analysis_completed or self._categorization_completed or self._is_categorizing:
             self.category_settings_group.setVisible(True)
             scroll_visible = True
             for cat_input in self.category_inputs:
                  cat_input["name_input"].setEnabled(OLLAMA_AVAILABLE and not self._is_categorizing)
                  cat_input["desc_input"].setEnabled(OLLAMA_AVAILABLE and not self._is_categorizing)
        else:
             self.category_settings_group.setVisible(False)
             for cat_input in self.category_inputs:
                 cat_input["name_input"].setEnabled(False)
                 cat_input["desc_input"].setEnabled(False)


        if self._categorization_completed or self._is_categorizing:
             self.categorization_results_group.setVisible(True)
             scroll_visible = True
        else:
             self.categorization_results_group.setVisible(False)
             self.clear_categorization_table()

        if self._categorization_completed or self._is_categorizing:
             self.sentiment_by_category_group.setVisible(True)
             scroll_visible = True
        else:
             self.sentiment_by_category_group.setVisible(False)
             self.clear_sentiment_by_category_charts() 
        self.scroll_area.setVisible(scroll_visible)



    def get_dynamic_filepaths(self, query):
        """Creates file paths based on the sanitized search query."""
        safe_query = sanitize_filename(query)
        urls_path = os.path.join(OUTPUT_DIR, URLS_FILE_TEMPLATE.format(query=safe_query))
        comments_path = os.path.join(OUTPUT_DIR, COMMENTS_FILE_TEMPLATE.format(query=safe_query))
        state_path = os.path.join(OUTPUT_DIR, STATE_FILE_TEMPLATE.format(query=safe_query))
        sentiment_path = os.path.join(OUTPUT_DIR, SENTIMENT_FILE_TEMPLATE.format(query=safe_query))
        categorization_path = os.path.join(OUTPUT_DIR, CATEGORIZATION_FILE_TEMPLATE.format(query=safe_query))
        self._safe_query_name = safe_query
        return safe_query, urls_path, comments_path, state_path, sentiment_path, categorization_path

    def check_resume_state(self):
        """Checks for existing state files and updates GUI accordingly."""
        print("check_resume_state başlatıldı.")

        self._is_collecting = False
        self._is_analyzing = False
        self._is_categorizing = False
        self._all_steps_completed_for_query = False
        self._comments_collected = False
        self._analysis_completed = False
        self._categorization_completed = False
        self._resume_available = False
        self._safe_query_name = ""
        self._urls_filepath = ""
        self._comments_filepath = ""
        self._state_filepath = ""
        self._sentiment_filepath = ""
        self._categorization_filepath = ""

        last_state_query = ""
        last_state_comment_limit = self.comment_limit_spinbox.value()

        state_files = glob.glob(os.path.join(OUTPUT_DIR, "*_state.json"))
        latest_state_file = None

        if state_files:
             try:
                latest_state_file = max(state_files, key=os.path.getmtime)
                print(f"En son state dosyası kontrol ediliyor: {latest_state_file}")

                with open(latest_state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    next_index = state.get("next_url_index", 0)
                    state_query = state.get("search_query", "").strip()
                    last_state_query = state_query
                    last_state_comment_limit = state.get("comment_limit_per_url", self.comment_limit_spinbox.value())
                    saved_categories = state.get("categories", [])
                    saved_ollama_model = state.get("ollama_model", DEFAULT_OLLAMA_MODEL)
                    saved_batch_size = state.get("ollama_batch_size", DEFAULT_BATCH_SIZE) 


                    self._safe_query_name, self._urls_filepath, self._comments_filepath, self._state_filepath, self._sentiment_filepath, self._categorization_filepath = self.get_dynamic_filepaths(state_query)

                    urls_exist = os.path.exists(self._urls_filepath)
                    comments_exist = os.path.exists(self._comments_filepath)
                    sentiment_exist = os.path.exists(self._sentiment_filepath)
                    categorization_exist = os.path.exists(self._categorization_filepath)

                    report_filename = f"{self._safe_query_name}_rapor.docx"
                    self._report_filepath = os.path.join(OUTPUT_DIR, report_filename)
                    report_exist = os.path.exists(self._report_filepath)


                    comment_count = 0
                    if comments_exist:
                        try:
                           df_comments_check = pd.read_csv(self._comments_filepath, encoding='utf-8', usecols=['text'], nrows=1)
                           if not df_comments_check.empty and 'text' in df_comments_check.columns:
                                df_comments_check_full = pd.read_csv(self._comments_filepath, encoding='utf-8', usecols=['text'])
                                comment_count = len(df_comments_check_full.dropna(subset=['text']))
                                if comment_count > 0:
                                     self._comments_collected = True
                                else:
                                     print(f"Uyarı: Yorum dosyası '{self._comments_filepath}' başlık dışında geçerli yorum içermiyor.")
                                     self._comments_collected = False
                           else:
                                print(f"Uyarı: Yorum dosyası '{self._comments_filepath}' boş, eksik sütun veya sadece başlık içeriyor.")
                                self._comments_collected = False
                        except pd.errors.EmptyDataError:
                             print(f"Uyarı: Yorum dosyası '{self._comments_filepath}' boş veya sadece başlık içeriyor.")
                             self._comments_collected = False
                        except KeyError:
                             print(f"Uyarı: Yorum dosyası '{self._comments_filepath}' 'text' sütunu içermiyor.")
                             self._comments_collected = False
                        except Exception as e:
                           print(f"check_resume_state: Yorum dosyası okunurken hata: {e}")
                           self._comments_collected = False


                    if sentiment_exist:
                         try:
                              df_sentiment_check = pd.read_csv(self._sentiment_filepath, encoding='utf-8', nrows=1)
                              required_sentiment_cols_header = ['duygu_tahmini', 'duygu_skoru', 'hesaplanan_tarih']
                              if not df_sentiment_check.empty and all(col in df_sentiment_check.columns for col in required_sentiment_cols_header):
                                   self._analysis_completed = True
                                   print(f"BERT analiz sonuç dosyası bulundu ve geçerli görünüyor: '{self._sentiment_filepath}'")
                              else:
                                   print(f"Uyarı: BERT analiz sonuç dosyası '{self._sentiment_filepath}' boş veya eksik sütun içeriyor.")
                                   self._analysis_completed = False
                         except pd.errors.EmptyDataError:
                              print(f"Uyarı: BERT analiz sonuç dosyası '{self._sentiment_filepath}' boş veya sadece başlık içeriyor.")
                              self._analysis_completed = False
                         except KeyError:
                              print(f"Uyarı: BERT analiz sonuç dosyası '{self._sentiment_filepath}' eksik sütun içeriyor.")
                              self._analysis_completed = False
                         except Exception as e:
                              print(f"check_resume_state: BERT analiz dosyası okunurken hata: {e}")
                              self._analysis_completed = False

                    if categorization_exist and saved_categories and OLLAMA_AVAILABLE:
                         try:
                              df_categorized_check, category_cols_found = self._get_processed_categorization_df()
                              total_input_rows = 0
                              if os.path.exists(self._sentiment_filepath):
                                   try:
                                        df_input_check = pd.read_csv(self._sentiment_filepath, encoding='utf-8', usecols=['text'], low_memory=False)
                                        total_input_rows = len(df_input_check) #
                                   except:
                                        pass

                              if not df_categorized_check.empty and category_cols_found:
                                self._categorization_completed = True
                                print(f"Ollama kategorizasyon sonuç dosyası bulundu ve geçerli görünüyor: '{self._categorization_filepath}' ({len(df_categorized_check)} işlenmiş yorum)")
                              else:
                                   print(f"Uyarı: Ollama kategorizasyon sonuç dosyası '{self._categorization_filepath}' boş, eksik kategori sütunu veya işlenmiş yorum içermiyor.")
                                   self._categorization_completed = False
                         except Exception as e:
                              print(f"check_resume_state: Ollama kategorizasyon dosyası okunurken hata: {e}")
                              self._categorization_completed = False
                    else:
                        self._categorization_completed = False

                    if report_exist:
                         self._report_completed = True
                         print(f"Rapor dosyası bulundu: '{self._report_filepath}'")
                    else:
                         self._report_completed = False
                         print(f"Rapor dosyası bulunamadı: '{self._report_filepath}'")


                    total_urls_in_file = 0
                    if urls_exist:
                         total_urls_in_file = self.count_urls_in_file(self._urls_filepath)

                    if urls_exist and next_index < total_urls_in_file:
                         self._resume_available = True
                         print(f"Resume available for query '{state_query}' from index {next_index}.")
                         self.status_label.setText(f"'{state_query}' için İşlem Bitmiştir.")
                         self.query_input.setText(state_query)
                         self.comment_limit_spinbox.setValue(last_state_comment_limit)

                         if saved_categories:
                              for i, cat_dict in enumerate(self.category_inputs):
                                   if i < len(saved_categories):
                                        cat_dict["name_input"].setText(saved_categories[i].get("category", ""))
                                        cat_dict["desc_input"].setText(saved_categories[i].get("aciklama", ""))
                                   else:
                                        cat_dict["name_input"].clear()
                                        cat_dict["desc_input"].clear()


                    else:
                         print(f"No resume for download for query '{state_query}'. URLs processed: {next_index}/{total_urls_in_file if urls_exist else 0}.")
                         if self._comments_collected:
                             self.status_label.setText(f"'{state_query}' için indirme tamamlandı ({total_urls_in_file} URL, {comment_count} yorum).")
                         else:
                             self.status_label.setText(f"'{state_query}' için indirme durumu kontrol edildi.")

                         self._resume_available = False
                         self.query_input.setText(state_query)
                         self.comment_limit_spinbox.setValue(last_state_comment_limit) 

                         if saved_categories:
                              for i, cat_dict in enumerate(self.category_inputs):
                                   if i < len(saved_categories):
                                        cat_dict["name_input"].setText(saved_categories[i].get("category", ""))
                                        cat_dict["desc_input"].setText(saved_categories[i].get("aciklama", ""))
                                   else:
                                        cat_dict["name_input"].clear()
                                        cat_dict["desc_input"].clear()


             except Exception as e:
                  print(f"State dosyası '{latest_state_file}' okunurken hata: {e}. State temizleniyor.")
                  if latest_state_file and os.path.exists(latest_state_file):
                       try:
                            os.remove(latest_state_file)
                            print(f"Hatalı state dosyası silindi: {latest_state_file}")
                       except Exception as rm_e:
                            print(f"Hata: Hatalı state dosyası silinemedi {latest_state_file}: {rm_e}")

                  self._safe_query_name, self._urls_filepath, self._comments_filepath, self._state_filepath, self._sentiment_filepath, self._categorization_filepath = self.get_dynamic_filepaths("") # Reset paths based on empty query
                  self.status_label.setText("Durum: Hazır. Yeni işlem başlatılabilir.")
                  self.query_input.setText("") 
                  self.search_limit_spinbox.setValue(20) 
                  self.comment_limit_spinbox.setValue(200)
                  for cat_dict in self.category_inputs:
                      cat_dict["name_input"].clear()
                      cat_dict["desc_input"].clear()
                  self._resume_available = False 


        if self._comments_collected and self._analysis_completed and self._categorization_completed and self._report_completed:
            self._all_steps_completed_for_query = True
            print("CHECK_RESUME_STATE: Tüm adımlar tamamlanmış görünüyor.")
        else:
            self._all_steps_completed_for_query = False

        self.show_existing_results_areas()
        if self._analysis_completed or self._is_analyzing:
             self.display_sentiment_charts(self._sentiment_filepath)
        else:
             self.clear_sentiment_charts()


        if self._categorization_completed or self._is_categorizing:
             self._update_categorization_table()
             if self._categorization_completed:
                  self.display_sentiment_by_category_charts()
             else:
                  self.clear_sentiment_by_category_charts()
        else:
             self.clear_categorization_table()
             self.clear_sentiment_by_category_charts()

        self.update_button_states()

        print("check_resume_state bitişi.")


    def count_urls_in_file(self, filepath):
        """Counts URLs in the given CSV file (excluding header), using pandas."""
        if not filepath or not os.path.exists(filepath):
            return 0
        try:
            if os.path.getsize(filepath) == 0:
                 return 0
            df = pd.read_csv(filepath, usecols=['URL'], encoding='utf-8', on_bad_lines='skip', low_memory=False)
            count = df['URL'].dropna().shape[0] 
            return count
        except FileNotFoundError:
             print(f"Error (count_urls_in_file): Dosya bulunamadı: {filepath}")
             return 0
        except KeyError:
             print(f"Error (count_urls_in_file): '{filepath}' dosyasında 'URL' sütunu bulunamadı.")
             return 0
        except pd.errors.EmptyDataError:
             print(f"Error (count_urls_in_file): '{filepath}' dosyası boş veya sadece başlık satırı var.")
             return 0
        except Exception as e:
            print(f"Error (count_urls_in_file): URL dosyası sayma hatası '{filepath}': {e}")
            return 0


    def start_process(self):
        """Starts or resumes the data collection process (Search -> Download)."""
        if self.step1_button is None or not self.step1_button.isEnabled():
             print("Start process requested but Step 1 button is disabled. Skipping.")
             return

        if self._is_collecting or self._is_analyzing or self._is_categorizing:
            self.status_label.setText("Durum: Zaten bir işlem devam ediyor.")
            return

        current_query = self.query_input.text().strip()

        if self._resume_available:
            print(f"Devam ediliyor: '{self._safe_query_name}'. İndirme işlemine geçiliyor.")
            self.start_download_process(is_resume=True)

        else: 
            if not current_query:
                 QMessageBox.warning(self, "Geçersiz Giriş", "Lütfen aranacak kelimeyi girin.")
                 return

            search_limit = self.search_limit_spinbox.value()
            comment_limit = self.comment_limit_spinbox.value()

            if search_limit <= 0 or comment_limit <= 0:
                QMessageBox.warning(self, "Geçersiz Giriş", "URL ve Yorum sayıları pozitif olmalı.")
                return

            self._safe_query_name, self._urls_filepath, self._comments_filepath, self._state_filepath, self._sentiment_filepath, self._categorization_filepath = self.get_dynamic_filepaths(current_query)
            
            print(f"Yeni arama başlatılıyor ('{self._safe_query_name}'). İlişkili eski çıktı dosyaları temizleniyor.")
            
            self.clear_all_charts()
            self.scroll_area.setVisible(False) # ... vb.

            self._clear_output_files(reset_attributes=False)

            self._is_collecting = True
            self._is_analyzing = False
            self._is_categorizing = False
            self._comments_collected = False
            self._analysis_completed = False
            self._categorization_completed = False
            self._all_steps_completed_for_query = False 
            self._resume_available = False

            self.status_label.setText(f"Durum: '{current_query}' için webde url aranıyor ({search_limit} adet)...")
            self.progress_bar.setValue(0) 

            self.search_thread = QThread()
            self.search_worker = SearchWorker(query=current_query, limit=search_limit)
            self.search_worker.moveToThread(self.search_thread)

            self.search_worker.search_finished.connect(self.handle_search_completion)
            self.search_worker.search_error.connect(self.handle_search_error)

            self.search_worker.search_finished.connect(self.search_thread.quit)
            self.search_worker.search_error.connect(self.search_thread.quit)

            self.search_thread.finished.connect(self.search_worker.deleteLater)
            self.search_thread.finished.connect(self.search_thread.deleteLater) 
            self.search_thread.finished.connect(self._cleanup_search_thread_refs)

            self.search_thread.started.connect(self.search_worker.run)
            self.search_thread.start()

            self.update_button_states()


    def _clear_output_files(self, reset_attributes=True):
        """Deletes urls.csv, comments.csv, state.json, sentiment.csv, and categorization.csv based on current paths."""
        files_to_clear = []
        paths_to_check = [self._state_filepath, self._urls_filepath, self._comments_filepath,
                          self._sentiment_filepath, self._categorization_filepath]

        for f_path in paths_to_check:
            if f_path and os.path.exists(f_path):
                 files_to_clear.append(f_path)

        if files_to_clear:
             print(f"Temizleniyor: {files_to_clear}")
             for f in files_to_clear:
                 try:
                     os.remove(f)
                     print(f"Dosya temizlendi: {f}")
                 except Exception as e:
                     print(f"Hata: Dosya temizlenemedi {f}: {e}")
        else:
             print("Temizlenecek herhangi bir çıktı dosyası bulunamadı.")

        if reset_attributes:
            self._safe_query_name = ""
            self._urls_filepath = ""
            self._comments_filepath = ""
            self._state_filepath = ""
            self._sentiment_filepath = ""
            self._categorization_filepath = ""

            self._is_collecting = False
            self._is_analyzing = False
            self._is_categorizing = False
            self._comments_collected = False
            self._analysis_completed = False
            self._categorization_completed = False
            self._resume_available = False
            self._all_steps_completed_for_query = False 

            self._cleanup_sentiment_thread_refs()
            self._cleanup_categorization_thread_refs()


    def clear_state_and_files(self):
        """Clears GUI state and deletes associated output files."""
        if self._is_collecting or self._is_analyzing or self._is_categorizing:
             QMessageBox.warning(self, "Uyarı", "İşlem devam ederken temizlik yapılamaz.")
             return

        current_query = self.query_input.text().strip()
        self._safe_query_name, self._urls_filepath, self._comments_filepath, self._state_filepath, self._sentiment_filepath, self._categorization_filepath = self.get_dynamic_filepaths(current_query)


        any_output_file_exists = bool(
            (hasattr(self, '_state_filepath') and self._state_filepath and os.path.exists(self._state_filepath)) or \
            (hasattr(self, '_urls_filepath') and self._urls_filepath and os.path.exists(self._urls_filepath)) or \
            (hasattr(self, '_comments_filepath') and self._comments_filepath and os.path.exists(self._comments_filepath)) or \
            (hasattr(self, '_sentiment_filepath') and self._sentiment_filepath and os.path.exists(self._sentiment_filepath)) or \
            (hasattr(self, '_categorization_filepath') and self._categorization_filepath and os.path.exists(self._categorization_filepath))
        )


        if not any_output_file_exists:
             self.status_label.setText(f"Durum: '{self._safe_query_name or 'varsayılan'}' için temizlenecek dosya bulunamadı.")
             print(f"Temizle isteği: '{self._safe_query_name or 'varsayılan'}' için temizlenecek dosya yok.")
             QTimer.singleShot(100, self.check_resume_state)
             return 
        reply = QMessageBox.question(self, "Temizle",
                                     f"'{self._safe_query_name or 'varsayılan'}' sorgusuna ait tüm ilişkili çıktı dosyalarını (state, URL'ler, yorumlar, analiz, kategorizasyon) temizlemek istediğinize emin misiniz?\n\nBu işlem geri alınamaz.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.clear_all_charts()
            self.scroll_area.setVisible(False)
            self.sentiment_charts_group.setVisible(False)
            self.category_settings_grou1p.setVisible(False)
            self.categorization_results_group.setVisible(False)
            self.sentiment_by_category_group.setVisible(False)


            self._clear_output_files(reset_attributes=True)

            self.query_input.setText("")
            self.search_limit_spinbox.setValue(20)
            self.comment_limit_spinbox.setValue(200)
            for category_dict in self.category_inputs:
                 category_dict["name_input"].clear()
                 category_dict["desc_input"].clear()


            self.status_label.setText("Durum: Temizlik tamamlandı. Hazır.") 
            self.categorization_status_label.setText("Bekleniyor...")

            QTimer.singleShot(100, self.check_resume_state)


    def update_button_states(self):
        """Updates the enabled/disabled state AND visual style of buttons based on the application's current status."""
        print(f"DEBUG: update_button_states çağrıldı. is_collecting={self._is_collecting}, is_analyzing={self._is_analyzing}, is_categorizing={self._is_categorizing}, is_reporting={self._is_reporting}, categorization_completed={self._categorization_completed}") # Debug print
        is_processing = self._is_collecting or self._is_analyzing or self._is_categorizing or self._is_reporting 

        self.query_input.setEnabled(not is_processing)
        self.search_limit_spinbox.setEnabled(not is_processing and not self._comments_collected)
        self.comment_limit_spinbox.setEnabled(not is_processing and not self._comments_collected)


        if self.step1_button:
             if self._comments_collected:
                 self.step1_button.setText("1. Veri Toplandı")
                 next_step_completed_or_active = self._analysis_completed or self._is_analyzing
                 self.step1_button.setEnabled(not is_processing and not next_step_completed_or_active)
             elif self._resume_available:
                 self.step1_button.setText("1. Devam Et")
                 self.step1_button.setEnabled(not is_processing)
             else:
                 self.step1_button.setText("1. Veri Topla")
                 self.step1_button.setEnabled(not is_processing)

        if self.step2_button:
            if self._all_steps_completed_for_query or self._analysis_completed:
                self.step2_button.setText("2. Analiz Tamamlandı")
            else:
                self.step2_button.setText("2. Duygu Analizi (BERT)")
            self.step2_button.setEnabled(not is_processing and self._comments_collected and not self._analysis_completed)

        if self.step3_button:
            if self._all_steps_completed_for_query or self._categorization_completed:
                self.step3_button.setText("3. Kategorizasyon Tamamlandı")
            else:
                self.step3_button.setText("3. Kategorize Et")
            self.step3_button.setEnabled(not is_processing and OLLAMA_AVAILABLE and self._analysis_completed and not self._categorization_completed)

        if self.step4_button:
            if self._all_steps_completed_for_query:
                self.step4_button.setText("4. Rapor Oluşturuldu")
            else:
                self.step4_button.setText("4. Rapor Oluştur")
            self.step4_button.setEnabled(not is_processing and self._categorization_completed)


        if self.stop_button: 
             self.stop_button.setEnabled(is_processing)

        any_output_file_exists = (hasattr(self, '_state_filepath') and self._state_filepath and os.path.exists(self._state_filepath)) or \
                                 (hasattr(self, '_urls_filepath') and self._urls_filepath and os.path.exists(self._urls_filepath)) or \
                                 (hasattr(self, '_comments_filepath') and self._comments_filepath and os.path.exists(self._comments_filepath)) or \
                                 (hasattr(self, '_sentiment_filepath') and self._sentiment_filepath and os.path.exists(self._sentiment_filepath)) or \
                                 (hasattr(self, '_categorization_filepath') and self._categorization_filepath and os.path.exists(self._categorization_filepath))

        if self.clear_state_button: 
             self.clear_state_button.setEnabled(not is_processing and bool(any_output_file_exists)) 


        if not is_processing:
            if self._all_steps_completed_for_query:
                self.status_label.setText(f"Durum: '{self._safe_query_name}' için BÜTÜN İŞLEMLER TAMAMLANDI.")
            elif self._categorization_completed:
                 self.status_label.setText(f"Durum: '{self._safe_query_name}' için kategorizasyon tamamlandı.")
            elif self._analysis_completed:
                 self.status_label.setText(f"Durum: '{self._safe_query_name}' için BERT analizi tamamlandı.")
            elif self._comments_collected:
                 comment_count_str = ""
                 if hasattr(self, '_comments_filepath') and self._comments_filepath and os.path.exists(self._comments_filepath):
                     try:
                         df_comments_check_full = pd.read_csv(self._comments_filepath, encoding='utf-8', usecols=['text'])
                         comment_count = len(df_comments_check_full.dropna(subset=['text']))
                         comment_count_str = f" ({comment_count} yorum)"
                     except: pass
                 self.status_label.setText(f"Durum: '{self._safe_query_name}' için yorum indirme tamamlandı{comment_count_str}.")
            elif self._resume_available:
                 self.status_label.setText(f"Durum: '{self._safe_query_name}' için devam etmeye hazır.")
            else:
                 self.status_label.setText("Durum: Hazır.")

        self._update_step_button_styles()


    def _update_step_button_styles(self):
        """Applies CSS classes to step buttons and line labels based on the application state."""
        step_states = {
            1: {
                "active": self._is_collecting,
                "completed": self._comments_collected,
                "button": self.step1_button
            },
            2: {
                "active": self._is_analyzing,
                "completed": self._analysis_completed,
                "button": self.step2_button
            },
            3: {
                "active": self._is_categorizing, 
                "completed": self._categorization_completed,
                "button": self.step3_button
            },
            4: {
                "active": self._is_reporting,
                "completed": self._report_completed,
                "button": self.step4_button
            }
        }

        line_label_states = {
            1: {
                "completed": self._comments_collected, 
                "label": self.line_label_1
            },
            2: {
                "completed": self._analysis_completed,
                "label": self.line_label_2
            },
            3: {
                "completed": self._categorization_completed,
                "label": self.line_label_3
            }
        }

        for step, state_info in step_states.items(): 
            button = state_info["button"]
            if button is None:
                 continue

            current_classes = ["step-button"]

            if state_info["active"]:
                current_classes.append("step-button-active")
            elif state_info["completed"]:
                 is_last_completed = True
                 for next_step_num in range(step + 1, len(step_states) + 1):
                      next_step_s = step_states.get(next_step_num, {})
                      if next_step_s and (next_step_s.get("active", False) or next_step_s.get("completed", False)):
                           is_last_completed = False
                           break
                 if is_last_completed:
                      current_classes.append("step-button-last-completed")
                 else:
                      current_classes.append("step-button-completed")
            else:
                 if button.isEnabled():
                     current_classes.append("step-button-pending")

            button.setProperty("class", " ".join(current_classes))
            button.style().polish(button)

        for i, label_state_info in line_label_states.items():
             label = label_state_info["label"]
             if label is None:
                  continue

             current_label_classes = ["step-separator-label"] 

             previous_step_completed = step_states.get(i, {}).get("completed", False)

             if previous_step_completed:
                  current_label_classes.append("separator-completed")
             else:
                  current_label_classes.append("separator-pending")

             label.setProperty("class", " ".join(current_label_classes))
             label.style().polish(label)


    def handle_search_completion(self, urls_list):
        """Handles the completion of the search process."""
        print("Arama tamamlandı (handle_search_completion).")

        if not urls_list:
             print("Arama tamamlandı, URL bulunamadı veya işlem durduruldu.")
             self._is_collecting = False
             self._comments_collected = False
             self.status_label.setText(f"Durum: '{self.query_input.text().strip()}' için url arama bitti (URL bulunamadı).")
             self.progress_bar.setValue(0)
             QTimer.singleShot(100, self.check_resume_state)
        else:
            print(f"Arama tamamlandı. Toplam {len(urls_list)} URL bulundu. '{self._urls_filepath}' dosyasına kaydediliyor.")
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(self._urls_filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['URL'])
                    for url in urls_list:
                        writer.writerow([url])
                print(f"{len(urls_list)} URL '{self._urls_filepath}' dosyasına kaydedildi.")

                initial_state = {
                    "next_url_index": 0,
                    "comment_limit_per_url": self.comment_limit_spinbox.value(),
                    "search_query": self.query_input.text().strip(),
                    "categories": self.get_categories_from_inputs(),
                    "ollama_model": DEFAULT_OLLAMA_MODEL,
                    "ollama_batch_size": DEFAULT_BATCH_SIZE
                }
                with open(self._state_filepath, 'w', encoding='utf-8') as f:
                     json.dump(initial_state, f, indent=4)
                print(f"Initial state kaydedildi: '{self._state_filepath}'")

            except Exception as e:
                error_message = f"URL listesi veya state dosyası kaydedilirken hata oluştu: {e}"
                print("File Save Error (Search Completion):", error_message)
                self.status_label.setText("Durum: Hata - Dosya kaydı yapılamadı.")
                self.progress_bar.setValue(0)
                QMessageBox.critical(self, "Dosya Kayıt Hatası", error_message)
                self._is_collecting = False
                self._comments_collected = False
                QTimer.singleShot(100, self.check_resume_state)
                return
            self.start_download_process()

    def handle_search_error(self, message):
        """Handles errors during the search process."""
        print(f"Arama Hatası (handle_search_error): {message}")
        self._is_collecting = False
        self._comments_collected = False 

        self.status_label.setText(f"Durum: Arama Hatası - {message}")
        self.progress_bar.setValue(0) 
        QMessageBox.critical(self, "Arama Hatası", message)

        QTimer.singleShot(100, self.check_resume_state)

   

        QTimer.singleShot(100, self.check_resume_state)

    def _cleanup_search_thread_refs(self):
        """Cleans up SearchWorker thread and worker references AFTER thread finishes."""
        print("Search thread referansları temizleniyor.")
        self.search_worker = None
        self.search_thread = None
        print("Search thread referansları temizlendi.")


    def start_download_process(self, is_resume=False):
        """Starts the comment download process."""
        if not is_resume and not self._is_collecting and (self.step1_button is None or not self.step1_button.isEnabled()):
             print("Start download requested (not resume, not collecting) but Step 1 button is disabled. Skipping.")
             return

        if (self._is_analyzing or self._is_categorizing) or \
           (self._is_collecting and not is_resume and self.download_worker is not None): #
            if not (self._is_collecting and self.download_worker is None): 
                self.status_label.setText("Durum: Zaten başka bir işlem devam ediyor.")
                return


        if not USE_COMMENT_DOWNLOADER:
            msg = "Hata: Yorum indirme kütüphanesi kurulu değil veya devre dışı bırakılmış."
            self.status_label.setText(msg)
            QMessageBox.critical(self, "Kütüphane Eksik", msg + "\nLütfen 'youtube-comment-downloader' kurun veya data_download.py'yi kontrol edin.")
            self._is_collecting = False
            self._comments_collected = False
            QTimer.singleShot(100, self.check_resume_state)
            return

        self._is_collecting = True
        comment_limit = self.comment_limit_spinbox.value()
        self.status_label.setText("Durum: Yorumlar alınıyor...")
        self.progress_bar.setValue(0)

        self.update_button_states() 

        self.download_thread = QThread()
        self.download_worker = CommentDownloaderWorker(
            urls_filepath=self._urls_filepath,
            comments_filepath=self._comments_filepath,
            state_filepath=self._state_filepath,
            comment_limit_per_url=comment_limit
        )

        self.download_worker.moveToThread(self.download_thread)

        self.download_worker.status_message.connect(self.status_label.setText)
        self.download_worker.finished.connect(self.handle_download_completion)
        self.download_worker.error.connect(self.handle_download_error)

        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_worker.error.connect(self.download_thread.quit)
        self.download_thread.finished.connect(self.download_worker.deleteLater)
        self.download_thread.finished.connect(self.download_thread.deleteLater)
        self.download_thread.finished.connect(self._cleanup_download_thread_refs)

        self.download_thread.start()


    def _cleanup_download_thread_refs(self):
        """Cleans up CommentDownloaderWorker thread and worker references AFTER thread finishes."""
        print("Download thread referansları temizleniyor.")
        self.download_worker = None
        self.download_thread = None
        print("Download thread referansları temizlendi.")

    def handle_download_completion(self):
        """Handles the completion of the download process."""
        print("İndirme işlemi tamamlandı veya durduruldu (handle_download_completion).")
        self._is_collecting = False 
        comments_file_valid = False
        comment_count = 0
        if hasattr(self, '_comments_filepath') and self._comments_filepath and os.path.exists(self._comments_filepath):
             try:
                  df_comments_check = pd.read_csv(self._comments_filepath, encoding='utf-8', usecols=['text'], nrows=1) #
                  if not df_comments_check.empty and 'text' in df_comments_check.columns:
                      df_comments_check_full = pd.read_csv(self._comments_filepath, encoding='utf-8', usecols=['text'])
                      comment_count = len(df_comments_check_full.dropna(subset=['text'])) 
                      if comment_count > 0:
                           comments_file_valid = True
                           self._comments_collected = True 
                           print(f"İndirme tamamlandı. Yorum dosyası geçerli: '{self._comments_filepath}' ({comment_count} yorum)")
                      else:
                           print(f"İndirme tamamlandı. Yorum dosyası '{self._comments_filepath}' geçerli yorum içermiyor.")
                           self._comments_collected = False
                  else:
                      print(f"İndirme tamamlandı. Yorum dosyası '{self._comments_filepath}' boş, eksik sütun veya sadece başlık içeriyor.")
                      self._comments_collected = False
             except pd.errors.EmptyDataError:
                 print(f"İndirme tamamlandı. Yorum dosyası '{self._comments_filepath}' boş veya sadece başlık içeriyor.")
                 self._comments_collected = False
             except KeyError:
                 print(f"İndirme tamamlandı. Yorum dosyası '{self._comments_filepath}' 'text' sütunu içermiyor.")
                 self._comments_collected = False
             except Exception as e:
                  print(f"handle_download_completion: Yorum dosyası okunurken hata: {e}")
                  self._comments_collected = False
        else:
             print("İndirme tamamlandı ama yorum dosyası bulunamadı.")
             self._comments_collected = False


        if self._comments_collected:
             self.status_label.setText(f"Durum: Yorum indirme tamamlandı ({comment_count} yorum bulundu).")
             self.progress_bar.setValue(100)
        else:
             self.status_label.setText("Durum: Yorum indirme tamamlandı ancak yorum bulunamadı.")
             self.progress_bar.setValue(0)

        try:
            if hasattr(self, '_state_filepath') and self._state_filepath and os.path.exists(self._state_filepath):
                 with open(self._state_filepath, 'r', encoding='utf-8') as f:
                     state = json.load(f)
                 pass
            print("Relying on worker to have updated the state file on completion.")

        except Exception as e:
             print(f"Warning: handle_download_completion: Final state file saving/checking failed: {e}")


        QTimer.singleShot(100, self.check_resume_state)


    def handle_download_error(self, message):
        """Handles errors during the download process."""
        print(f"Yorum İndirme Hatası (handle_download_error): {message}")
        self._is_collecting = False 
        self._comments_collected = False 

        self.status_label.setText(f"Durum: Yorum İndirme Hatası - {message}")
        self.progress_bar.setValue(0) 
        QMessageBox.critical(self, "Yorum İndirme Hatası", message)

        QTimer.singleShot(100, self.check_resume_state)

    def stop_current_process(self):
        """Çalışan mevcut işlemi durdurmak için sinyal gönderir."""
        print("GUI'den durdurma isteği alındı.")
        self.stop_button.setEnabled(False) # Butonu hemen devre dışı bırak

        worker_stopped = False
        if self._is_collecting and self.download_worker:
            self.status_label.setText("Durum: Veri toplama durduruluyor...")
            self.download_worker.stop()
            worker_stopped = True
        elif self._is_analyzing and self.sentiment_worker:
            self.status_label.setText("Durum: Duygu analizi durduruluyor (mevcut işlem bittiğinde)...")
            self.sentiment_worker.stop()
            worker_stopped = True
        elif self._is_categorizing and self._categorization_worker:
            self.status_label.setText("Durum: Kategorizasyon durduruluyor...")
            self._categorization_worker.stop()
            worker_stopped = True
        elif self._is_reporting and hasattr(self, '_report_worker') and self._report_worker:
            self.status_label.setText("Durum: Rapor oluşturma durduruluyor...")
            self._report_worker.stop()
            worker_stopped = True
            
        if not worker_stopped:
            self.status_label.setText("Durum: Durdurulacak aktif bir işlem bulunamadı.")
            self.update_button_states() # Butonları tekrar doğru duruma getir

    def start_sentiment_analysis(self):
        """Starts the BERT sentiment analysis process."""
        if self.step2_button is None or not self.step2_button.isEnabled():
             print("Start sentiment analysis requested but Step 2 button is disabled. Skipping.")
             return 
        if self._is_collecting or self._is_analyzing or self._is_categorizing:
            self.status_label.setText("Durum: Zaten başka bir işlem devam ediyor.")
            return

        current_query = self.query_input.text().strip()
        if not current_query:
             QMessageBox.warning(self, "Geçersiz Giriş", "Lütfen analiz edilecek yorumların ait olduğu arama kelimesini girin.")
             return

        self._safe_query_name, _, comments_filepath_input, _, sentiment_filepath_output, _ = self.get_dynamic_filepaths(current_query)

        comments_file_has_data = False
        total_comments = 0
        if hasattr(self, '_comments_filepath') and self._comments_filepath and os.path.exists(self._comments_filepath):
             try:
                 df_check = pd.read_csv(self._comments_filepath, encoding='utf-8', usecols=['text'], low_memory=False)
                 total_comments = len(df_check['text'].fillna('').str.strip().loc[lambda x: x != ''])
                 if total_comments > 0:
                      comments_file_has_data = True
                 else:
                      print(f"BERT Analiz için yorum dosyası bulundu ama boş veya 'text' sütunu eksik: '{comments_filepath_input}'")
             except pd.errors.EmptyDataError:
                 print(f"BERT Analiz için yorum dosyası '{comments_filepath_input}' boş veya sadece başlık içeriyor.")
             except KeyError:
                  print(f"BERT Analiz için yorum dosyası '{comments_filepath_input}' 'text' sütunu içermiyor.")
             except Exception as e:
                  print(f"BERT Analiz için yorum dosyası '{comments_filepath_input}' okunurken hata: {e}")
                  pass 


        if not comments_file_has_data:
             QMessageBox.warning(self, "Veri Yok", f"Analiz edilecek yorum dosyası ('{comments_filepath_input}') bulunamadı veya boş/geçersiz.\nLütfen önce o arama kelimesi için yorumları indirin veya BERT analizini yapın.")
             QTimer.singleShot(100, self.check_resume_state)
             return 


        if self._analysis_completed:
            print(f"BERT Analiz sonuç dosyası bulundu ve tamamlanmış: '{sentiment_filepath_output}'. Grafikler çiziliyor.")
            self.status_label.setText("Durum: Mevcut BERT analiz sonuçları yükleniyor...")

            self.display_sentiment_charts(sentiment_filepath_output)

            QTimer.singleShot(100, self.check_resume_state)
            self.status_label.setText("Durum: BERT analiz sonuçları yüklendi ve gösterildi.")
            return 

        print(f"BERT Analiz başlatılıyor. Giriş: '{comments_filepath_input}', Çıkış: '{sentiment_filepath_output}'")
        self._is_analyzing = True
        self._analysis_completed = False 
        self._bert_estimated_total_seconds = 0
        if self._BERT_COMMENTS_PER_SECOND > 0 and total_comments > 0:
            self._bert_estimated_total_seconds = int(total_comments / self._BERT_COMMENTS_PER_SECOND)
        self._bert_analysis_start_time = time.time()

        if hasattr(self, '_bert_timer') and self._bert_timer is not None:
            if self._bert_timer.isActive(): self._bert_timer.stop()
            self._bert_timer.deleteLater()
        self._bert_timer = QTimer(self)
        self._bert_timer.timeout.connect(self._update_bert_analysis_timer)
        self._bert_timer.start(1000) 

        self._update_bert_analysis_timer() 
        self.clear_sentiment_charts()
        self.sentiment_charts_group.setVisible(False)
        self.category_settings_group.setVisible(False)
        self.categorization_results_group.setVisible(False)
        self.sentiment_by_category_group.setVisible(False)
        self.clear_categorization_table()
        self.clear_sentiment_by_category_charts() 
        self.update_button_states()


        self.sentiment_thread = QThread()
        self.sentiment_worker = SentimentAnalysisWorker(
            csv_input_path=comments_filepath_input,
            text_column_name='text',
            csv_output_path=sentiment_filepath_output
        )

        self.sentiment_worker.moveToThread(self.sentiment_thread)

        self.sentiment_worker.finished.connect(self.handle_sentiment_completion)
        self.sentiment_worker.error.connect(self.handle_sentiment_error)
        self.sentiment_thread.started.connect(self.sentiment_worker.run)
        self.sentiment_worker.finished.connect(self.sentiment_thread.quit)
        self.sentiment_worker.error.connect(self.sentiment_thread.quit)
        self.sentiment_thread.finished.connect(self.sentiment_thread.deleteLater)
        self.sentiment_thread.finished.connect(self._cleanup_sentiment_thread_refs)


        self.sentiment_thread.start()


    @pyqtSlot()
    def _update_bert_analysis_timer(self):
        """Updates the BERT analysis timer display in the status label."""
        if self._is_analyzing and self._bert_analysis_start_time is not None and self._bert_estimated_total_seconds >= 0:
            elapsed_time = time.time() - self._bert_analysis_start_time
            remaining_time = max(0, self._bert_estimated_total_seconds - int(elapsed_time))

            mins_rem = int(remaining_time // 60)
            secs_rem = int(remaining_time % 60)
            time_rem_str = f"{mins_rem:02d}:{secs_rem:02d}"

            self.status_label.setText(f"Durum: BERT Analizi Yapılıyor... (Tahmini Kalan Süre: {time_rem_str})")

        elif self._is_analyzing:
             self.status_label.setText("Durum: BERT Analizi Yapılıyor...")


    def _cleanup_sentiment_thread_refs(self):
        """Cleans up SentimentAnalysisWorker thread and worker references and stops the timer AFTER thread finishes."""
        print("BERT sentiment thread referansları temizleniyor.")
        if hasattr(self, '_bert_timer') and self._bert_timer is not None:
             if self._bert_timer.isActive():
                self._bert_timer.stop()
             self._bert_timer.deleteLater()
             self._bert_timer = None
        self.sentiment_worker = None
        self.sentiment_thread = None
        print("BERT sentiment thread referansları temizlendi.")

        self._bert_analysis_start_time = None 
        self._bert_estimated_total_seconds = 0 


    @pyqtSlot(int, int)
    def handle_sentiment_completion(self, positive_count, negative_count):
        """Handles the successful completion of the BERT analysis."""
        print("BERT analizi tamamlandı (handle_sentiment_completion).")
        self._is_analyzing = False
        self._analysis_completed = True

        if hasattr(self, '_bert_timer') and self._bert_timer is not None and self._bert_timer.isActive():
             self._bert_timer.stop()


        self.status_label.setText(f"Durum: BERT analizi tamamlandı. Pozitif: {positive_count}, Negatif: {negative_count}")
        self.progress_bar.setValue(100)
        QTimer.singleShot(100, self.check_resume_state)

       
        QApplication.processEvents()

        QTimer.singleShot(100, self.check_resume_state)


    @pyqtSlot(str)
    def handle_sentiment_error(self, message):
        """Handles errors during the BERT analysis."""
        print(f"BERT Analizi Hatası (handle_sentiment_error): {message}")
        self._is_analyzing = False
        self._analysis_completed = False 

        if hasattr(self, '_bert_timer') and self._bert_timer is not None and self._bert_timer.isActive():
             self._bert_timer.stop()

        self.status_label.setText("Durum: BERT analizi hatası.")
        self.progress_bar.setValue(0) 
        QMessageBox.critical(self, "Analiz Hatası", message)

        QTimer.singleShot(100, self.check_resume_state)



    def get_categories_from_inputs(self):
         """Collects category name and description from GUI input fields."""
         categories = []
         for category_dict in self.category_inputs:
              name = category_dict["name_input"].text().strip()
              description = category_dict["desc_input"].text().strip()
              if name:
                   categories.append({"category": name, "aciklama": description})
         return categories


    def start_categorization(self):
        """Starts the Ollama categorization process."""
        if self.step3_button is None or not self.step3_button.isEnabled(): 
             print("Start categorization requested but Step 3 button is disabled. Skipping.")
             return 
        if self._is_collecting or self._is_analyzing or self._is_categorizing:
            self.status_label.setText("Durum: Zaten başka bir işlem devam ediyor.")
            return

        if not OLLAMA_AVAILABLE:
            msg = "Hata: Ollama kütüphanesi kurulu değil. Kategorizasyon yapılamaz."
            QMessageBox.warning(self, "Kütüphane Eksik", msg)
            self.status_label.setText(msg)
            self.update_button_states()
            return 

        current_query = self.query_input.text().strip()
        if not current_query:
             QMessageBox.warning(self, "Geçersiz Giriş", "Lütfen kategorize edilecek yorumların ait olduğu arama kelimesini girin.")
             return


        self._safe_query_name, _, _, _, sentiment_filepath_input, categorization_filepath_output = self.get_dynamic_filepaths(current_query)

        categories_data = self.get_categories_from_inputs()
        main_category_names = [cat.get("category", "").strip() for cat in categories_data if cat.get("category", "").strip()]

        if not main_category_names:
            QMessageBox.warning(self, "Kategori Tanımlanmadı", "Lütfen Ollama kategorizasyonu için en least bir ana kategori adı girin.")
            self.category_settings_group.setVisible(True)
            self.scroll_area.setVisible(True)
            QTimer.singleShot(100, self.check_resume_state)
            return

        sentiment_input_file_valid = False
        if hasattr(self, '_sentiment_filepath') and self._sentiment_filepath and os.path.exists(self._sentiment_filepath):
             try:
                 df_check = pd.read_csv(self._sentiment_filepath, encoding='utf-8', nrows=1, low_memory=False) 
                 required_cols_for_cat_input = ['text', 'duygu_tahmini', 'duygu_skoru', 'hesaplanan_tarih']
                 if not df_check.empty and all(col in df_check.columns for col in required_cols_for_cat_input):
                      sentiment_input_file_valid = True
                 else:
                      print(f"Ollama Kategorizasyon için girdi dosyası bulundu ama boş veya gerekli sütunlar eksik: '{sentiment_filepath_input}'")
             except pd.errors.EmptyDataError:
                  print(f"Ollama Kategorizasyon için girdi dosyası '{sentiment_filepath_input}' boş veya sadece başlık içeriyor.")
             except KeyError:
                  print(f"Ollama Kategorizasyon için girdi dosyası '{sentiment_filepath_input}' gerekli sütunları içermiyor.")
             except Exception as e:
                  print(f"Ollama Kategorizasyon için girdi dosyası '{sentiment_filepath_input}' okunurken hata: {e}")
                  pass


        if not sentiment_input_file_valid:
             QMessageBox.warning(self, "Girdi Verisi Yok", f"Kategorize edilecek girdi dosyası ('{sentiment_filepath_input}') bulunamadı veya boş/geçersiz.\nLütfen önce o arama kelimesi için yorumları indirin ve BERT analizini yapın.")
             QTimer.singleShot(100, self.check_resume_state)
             return

        ollama_model = DEFAULT_OLLAMA_MODEL
        batch_size = DEFAULT_BATCH_SIZE

        try:
             if hasattr(self, '_state_filepath') and self._state_filepath:
                  state_to_save = {
                       "next_url_index": 0,
                       "comment_limit_per_url": self.comment_limit_spinbox.value(),
                       "search_query": self.query_input.text().strip(),
                       "categories": categories_data, 
                       "ollama_model": ollama_model, 
                       "ollama_batch_size": batch_size 
                   }
                  output_dir = os.path.dirname(self._state_filepath)
                  if output_dir and not os.path.exists(output_dir):
                      os.makedirs(output_dir)

                  with open(self._state_filepath, 'w', encoding='utf-8') as f:
                      json.dump(state_to_save, f, indent=4)
                  print(f"Current state and settings kaydedildi: '{self._state_filepath}'")
             else:
                  print("Warning: Cannot save state, _state_filepath is not set.")

        except Exception as e:
             print(f"Warning: State and settings kaydedilirken hata oluştu: {e}")
             QMessageBox.warning(self, "State Kayıt Hatası", f"Ayarlar kaydedilirken hata oluştu: {e}\nİşleme devam edilecek.")

        print(f"Ollama Kategorizasyon başlatılıyor. Girdi: '{sentiment_filepath_input}', Çıktı: '{categorization_filepath_output}', Model: '{ollama_model}', Batch: {batch_size}")
        self._is_categorizing = True 
        self._categorization_completed = False 

        self.status_label.setText(f"Durum: Ollama kategorizasyonu başlatılıyor... ('{ollama_model}' modeli kullanılıyor)")
        self.progress_bar.setValue(0) 


        self.categorization_results_group.setVisible(True)
        self.sentiment_by_category_group.setVisible(True)
        self.scroll_area.setVisible(True) 
        self.clear_categorization_table() 
        self.clear_sentiment_by_category_charts()

        self.categorization_status_label.setText("Hazırlanıyor...")

        self.update_button_states()

        self._categorization_thread = QThread()
        self._categorization_worker = CategorizationWorker(
            input_csv_path=sentiment_filepath_input,
            output_csv_path=categorization_filepath_output,
            comment_column='text',
            categories_data=categories_data, 
            ollama_model=ollama_model,
            batch_size=batch_size 
        )

        self._categorization_worker.moveToThread(self._categorization_thread)

        self._categorization_worker.progress_updated.connect(self.on_categorization_progress_updated)
        self._categorization_worker.plot_updated.connect(self.on_categorization_plot_updated) 
        self._categorization_worker.categorization_finished.connect(self.on_categorization_finished)
        self._categorization_worker.error_occurred.connect(self.on_categorization_error_occurred)
        self._categorization_worker.status_message.connect(self.status_label.setText)


        self._categorization_thread.started.connect(self._categorization_worker.run_categorization)
        self._categorization_worker.categorization_finished.connect(self._categorization_thread.quit)
        self._categorization_worker.error_occurred.connect(self._categorization_thread.quit)
        self._categorization_thread.finished.connect(self._categorization_thread.deleteLater)
        self._categorization_thread.finished.connect(self._cleanup_categorization_thread_refs) 


        if hasattr(self, '_categorization_plot_timer') and self._categorization_plot_timer is not None:
             if self._categorization_plot_timer.isActive(): self._categorization_plot_timer.stop()
             self._categorization_plot_timer.start(self._CATEGORIZATION_PLOT_UPDATE_INTERVAL_MS)


        self._categorization_start_time = time.time()

        self._categorization_thread.start()

        QTimer.singleShot(50, self._periodic_plot_update)


    def _cleanup_categorization_thread_refs(self):
        """Cleans up CategorizationWorker thread and worker references and stops the timer AFTER thread finishes."""
        print("Ollama kategorizasyon thread referansları temizleniyor.")
        if hasattr(self, '_categorization_plot_timer') and self._categorization_plot_timer is not None:
             if self._categorization_plot_timer.isActive():
                self._categorization_plot_timer.stop()
             self._categorization_plot_timer.deleteLater()
             self._categorization_plot_timer = None 

        self._categorization_worker = None
        self._categorization_thread = None
        self._categorization_start_time = None
        print(f"DEBUG: _cleanup_categorization_thread_refs: _is_categorizing bayrağı False olarak ayarlanıyor. Mevcut değeri: {self._is_categorizing}")
        self._is_categorizing = False 
        print(f"DEBUG: _cleanup_categorization_thread_refs: _is_categorizing bayrağı False olarak ayarlandı. Yeni değeri: {self._is_categorizing}")
        print("Ollama kategorizasyon thread referansları temizlendi.")

        QTimer.singleShot(100, self.check_resume_state)


    @pyqtSlot()
    def _periodic_plot_update(self):
        """Periodically updates the categorization table and sentiment-by-category charts."""
        if self._is_categorizing and self._categorization_worker is not None:
             self._update_categorization_table()
             self.display_sentiment_by_category_charts()


    @pyqtSlot(int, int, int, int)
    def on_categorization_progress_updated(self, total, processed, failed, empty):
        """Updates progress bars and status label based on worker signal."""
        if total > 0:
            percentage = int((processed / total) * 100)
            self.progress_bar.setValue(percentage)

            time_elapsed = time.time() - self._categorization_start_time if self._categorization_start_time is not None else 0

            processed_at_start = self._categorization_worker.processed_count_at_start if self._categorization_worker is not None else 0
            processed_this_run = processed - processed_at_start

            time_rem_str = "Hesaplanıyor..."
            if processed_this_run > 0 and time_elapsed > 1:
                 avg_time_per_comment = time_elapsed / processed_this_run
                 remaining_comments = total - processed
                 estimated_remaining_time = remaining_comments * avg_time_per_comment

                 mins_rem = int(estimated_remaining_time // 60)
                 secs_rem = int(estimated_remaining_time % 60)
                 time_rem_str = f"{mins_rem:02d}:{secs_rem:02d}"

            self.categorization_status_label.setText(
                f"İşleniyor: {processed}/{total} (Bu Çalıştırmada Hatalı: {failed}, Boş: {empty}) | Kalan Süre (Tahmini): {time_rem_str}"
            )
        else:
            self.progress_bar.setValue(0)
            self.categorization_progress_bar.setValue(0)
            self.categorization_status_label.setText("Ollama Kategorizasyon: İşlem başlatıldı. Veri yok.")


    @pyqtSlot()
    def on_categorization_plot_updated(self):
        """Handles plot/table update requests from the worker (can be used for non-periodic updates)."""
        self._update_categorization_table()
        self.display_sentiment_by_category_charts()

        self.categorization_results_group.setVisible(True)
        self.sentiment_by_category_group.setVisible(True)
        self.scroll_area.setVisible(True)


    @pyqtSlot()
    def on_categorization_finished(self):
        """Handles the successful completion of the categorization process."""
        print("Ollama kategorizasyon tamamlandı (on_categorization_finished).")
        self._is_categorizing = False 
        self._categorization_completed = True  

        if hasattr(self, '_categorization_plot_timer') and self._categorization_plot_timer is not None and self._categorization_plot_timer.isActive():
             self._categorization_plot_timer.stop()


        self.status_label.setText(f"Durum: '{self._safe_query_name}' için kategorizasyon tamamlandı.")
        self.progress_bar.setValue(100)
        QTimer.singleShot(100, self.check_resume_state)

        QApplication.processEvents()

        QTimer.singleShot(100, self.check_resume_state)


    @pyqtSlot(str)
    def on_categorization_error_occurred(self, message):
        """Handles errors during the categorization process."""
        print(f"Ollama Kategorizasyon Hatası (on_categorization_error_occurred): {message}")
        self._is_categorizing = False 

        if hasattr(self, '_categorization_plot_timer') and self._categorization_plot_timer is not None and self._categorization_plot_timer.isActive():
             self._categorization_plot_timer.stop()


        self.status_label.setText("Durum: Ollama kategorizasyon hatası.")
        self.progress_bar.setValue(0)
        self.categorization_progress_bar.setValue(0)
        self.categorization_status_label.setText("Hata Oluştu.")

        QMessageBox.critical(self, "Kategorizasyon Hatası", message)

        self._update_categorization_table()
        self.display_sentiment_by_category_charts()

        QTimer.singleShot(100, self.check_resume_state)


    def display_sentiment_charts(self, sentiment_filepath_to_display):
        """Reads sentiment analysis results and draws the overall sentiment pie and line charts."""
        self.clear_sentiment_charts() 
        if not sentiment_filepath_to_display or not os.path.exists(sentiment_filepath_to_display):
            self.sentiment_charts_group.setVisible(False)
            return

        try:
            df_results = pd.read_csv(sentiment_filepath_to_display, encoding='utf-8', low_memory=False)
            required_cols_sentiment = ['duygu_tahmini', 'duygu_skoru', 'hesaplanan_tarih']
            if not all(col in df_results.columns for col in required_cols_sentiment):
                missing_cols = [col for col in required_cols_sentiment if col not in df_results.columns]
                error_msg = f"BERT Grafik çizilemiyor: Analiz sonuç dosyasında gerekli sütunlar eksik ({missing_cols}). Dosya: '{sentiment_filepath_to_display}'"
                print(error_msg)
                self.status_label.setText("Durum: BERT Grafik Hatası - Eksik sütunlar.")
                self.sentiment_charts_group.setVisible(False) 
                return

            df_results['hesaplanan_tarih'] = pd.to_datetime(df_results['hesaplanan_tarih'], errors='coerce')
            df_results['duygu_skoru'] = pd.to_numeric(df_results['duygu_skoru'], errors='coerce')
            df_results['duygu_tahmini'] = df_results['duygu_tahmini'].astype(str).str.lower().replace('nan', '').replace('none', '').str.strip()


            df_plot_data = df_results.dropna(subset=['hesaplanan_tarih', 'duygu_skoru']).loc[df_results['duygu_tahmini'] != ''].copy()


            if df_plot_data.empty:
                info_msg = f"BERT Grafik çizilecek geçerli veri (tarih, skor, duygu tahmini) bulunamadı: '{sentiment_filepath_to_display}'."
                print(info_msg)
                self.status_label.setText("Durum: BERT Analizi - Grafik verisi yok.")
                self.sentiment_charts_group.setVisible(False) 
                return

            sentiment_pie_plotted = False
            sentiment_counts = df_plot_data['duygu_tahmini'].value_counts().reindex(['positive', 'negative']).fillna(0)

            counts_to_plot = sentiment_counts[sentiment_counts > 0]
            if not counts_to_plot.empty:
                self._sentiment_pie_figure, ax1 = plt.subplots(figsize=(5, 5))
                colors = []
                if 'positive' in counts_to_plot.index: colors.append('#2ecc71') 
                if 'negative' in counts_to_plot.index: colors.append('#e74c3c') 

                labels = [label.capitalize() for label in counts_to_plot.index.tolist()]

                ax1.pie(counts_to_plot, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, shadow=True)
                ax1.axis('equal')
                ax1.set_title(f'Duygu Dağılımı - {self._safe_query_name or "Bilinmiyor"}', fontsize=12)

                self._sentiment_pie_canvas = FigureCanvasQTAgg(self._sentiment_pie_figure)
                self._sentiment_pie_canvas.setMinimumSize(300, 350)
                self._sentiment_pie_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.sentiment_charts_layout.addWidget(self._sentiment_pie_canvas) 
                sentiment_pie_plotted = True


            sentiment_line_chart_plotted = False

            df_plot_data['hesaplanan_tarih_date'] = df_plot_data['hesaplanan_tarih'].dt.date

            df_pos = df_plot_data[df_plot_data['duygu_tahmini'].str.lower() == 'positive'].copy()
            df_neg = df_plot_data[df_plot_data['duygu_tahmini'].str.lower() == 'negative'].copy()

            df_pos_agg = df_pos.groupby('hesaplanan_tarih_date')['duygu_skoru'].mean().reset_index()
            df_neg_agg = df_neg.groupby('hesaplanan_tarih_date')['duygu_skoru'].mean().reset_index()

            df_pos_agg['hesaplanan_tarih_date'] = pd.to_datetime(df_pos_agg['hesaplanan_tarih_date'])
            df_pos_agg = df_pos_agg.sort_values('hesaplanan_tarih_date')

            df_neg_agg['hesaplanan_tarih_date'] = pd.to_datetime(df_neg_agg['hesaplanan_tarih_date'])
            df_neg_agg = df_neg_agg.sort_values('hesaplanan_tarih_date')

            if not df_pos_agg.empty or not df_neg_agg.empty:
                self._sentiment_line_figure, ax2 = plt.subplots(figsize=(8, 4))

                if not df_pos_agg.empty:
                    ax2.plot(df_pos_agg['hesaplanan_tarih_date'], df_pos_agg['duygu_skoru'], marker='o', linestyle='-', color='green', label='Pozitif Ortalama Skor')
                    sentiment_line_chart_plotted = True

                if not df_neg_agg.empty:
                    ax2.plot(df_neg_agg['hesaplanan_tarih_date'], df_neg_agg['duygu_skoru'], marker='o', linestyle='-', color='red', label='Negatif Ortalama Skor')
                    sentiment_line_chart_plotted = True

                if sentiment_line_chart_plotted:
                    ax2.set_title(f'Günlük Ortalama Duygu Skoru - {self._safe_query_name or "Bilinmiyor"}', fontsize=12)
                    ax2.set_xlabel('Tarih', fontsize=10)
                    ax2.set_ylabel('Ortalama Duygu Skoru', fontsize=10)
                    ax2.legend(fontsize=8)

                    self._sentiment_line_figure.autofmt_xdate()

                    ax2.set_ylim(0.1, 1.1) 
                    self._sentiment_line_canvas = FigureCanvasQTAgg(self._sentiment_line_figure)
                    self._sentiment_line_canvas.setMinimumSize(300, 350)
                    self._sentiment_line_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    self.sentiment_charts_layout.addWidget(self._sentiment_line_canvas) 
            if sentiment_pie_plotted or sentiment_line_chart_plotted:
                 self.sentiment_charts_group.setVisible(True)
                 self.scroll_area.setVisible(True) 
            else:
                 self.sentiment_charts_group.setVisible(False)


        except FileNotFoundError:
            print(f"Error (display_sentiment_charts): '{sentiment_filepath_to_display}' dosyası bulunamadı.")
            self.status_label.setText("Durum: BERT Grafik Hatası - Dosya bulunamadı.")
            self.sentiment_charts_group.setVisible(False)
        except pd.errors.EmptyDataError:
            print(f"Error (display_sentiment_charts): '{sentiment_filepath_to_display}' dosyası boş veya sadece başlık içeriyor.")
            self.status_label.setText("Durum: BERT Grafik Hatası - Dosya boş.")
            self.sentiment_charts_group.setVisible(False)
        except KeyError as e:
            print(f"Error (display_sentiment_charts): Gerekli bir sütun bulunamadı: {e}")
            self.status_label.setText(f"Durum: BERT Grafik Hatası - Gerekli sütun '{e}' bulunamadı.")
            self.sentiment_charts_group.setVisible(False)
        except Exception as e:
            error_msg = f"BERT Grafik çizilirken beklenmedik bir hata oluştu: {e}"
            print("BERT Plotting Error:", error_msg)
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Durum: BERT Grafik Hatası - {e}")
            self.sentiment_charts_group.setVisible(False)


    def _update_categorization_table(self):
        """
        Updates the categorization results table.
        It gets the data from _get_processed_categorization_df.
        This method runs in the GUI thread.
        """
        self.clear_categorization_table()

        df_processed, category_names_in_df = self._get_processed_categorization_df()

        self.categorization_results_group.setVisible(True)
        self.scroll_area.setVisible(True)

        total_processed_comments = df_processed.shape[0] 

        category_counts = {
            cat: df_processed[cat].eq(1).sum()
            for cat in category_names_in_df if cat in df_processed.columns
        }

        categories_to_show = [(name, count) for name, count in category_counts.items() if count > 0]


        preferred_category_order = self._get_all_managed_category_names()
        category_order_map = {name: i for i, name in enumerate(preferred_category_order)}
        categories_to_show.sort(key=lambda item: category_order_map.get(item[0], len(preferred_category_order) + 1))


        num_rows = len(categories_to_show) + (1 if total_processed_comments > 0 else 0)
        num_rows = max(num_rows, 1)
        self._categorization_table_widget = QTableWidget(num_rows, 3)
        self._categorization_table_widget.setObjectName("categorization_table")
        self._categorization_table_widget.setHorizontalHeaderLabels(["Kategori Adı", "Yorum Sayısı", "Yüzde (%)"])
        self._categorization_table_widget.verticalHeader().setVisible(False)
        self._categorization_table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) 
        self._categorization_table_widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows) 
        self._categorization_table_widget.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._categorization_table_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._categorization_table_widget.setMinimumHeight(220) 

        if total_processed_comments == 0 and not categories_to_show:
             item_no_data = QTableWidgetItem("Kategorize Edilmiş Geçerli Veri Yok")
             item_no_data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
             item_no_data.setFlags(item_no_data.flags() & ~Qt.ItemFlag.ItemIsSelectable) 
             self._categorization_table_widget.setItem(0, 0, item_no_data)
             self._categorization_table_widget.setSpan(0, 0, 1, 3) 

        else:
            row = 0
            for cat_name, count in categories_to_show:
                 percentage = (count / total_processed_comments) * 100 if total_processed_comments > 0 else 0

                 item_name = QTableWidgetItem(cat_name)
                 item_name.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                 self._categorization_table_widget.setItem(row, 0, item_name)

                 item_count = QTableWidgetItem(str(count))
                 item_count.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                 item_count.setFlags(item_count.flags() & ~Qt.ItemFlag.ItemIsSelectable) 
                 self._categorization_table_widget.setItem(row, 1, item_count)

                 item_percent = QTableWidgetItem(f"{percentage:.1f}%")
                 item_percent.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                 item_percent.setFlags(item_percent.flags() & ~Qt.ItemFlag.ItemIsSelectable) 
                 self._categorization_table_widget.setItem(row, 2, item_percent)

                 row += 1

           

        header = self._categorization_table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) 
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) 

        self.categorization_table_container_layout.addWidget(self._categorization_table_widget)

    def display_sentiment_by_category_charts(self):
        """
        Draws time-series sentiment charts for each category based on categorization results.
        It gets the data from _get_processed_categorization_df.
        This method runs in the GUI thread.
        """
        self.clear_sentiment_by_category_charts()

        df_processed, category_names_in_df = self._get_processed_categorization_df()

        self.sentiment_by_category_group.setVisible(True)
        self.scroll_area.setVisible(True) 

        categories_to_plot = [
             cat for cat in category_names_in_df
             if cat in df_processed.columns and (df_processed[cat].eq(1)).any()
        ]

        if df_processed.empty or not categories_to_plot:
             self.sentiment_by_category_group.setVisible(False)
             return 

        preferred_category_order = self._get_all_managed_category_names()
        category_order_map = {name: i for i, name in enumerate(preferred_category_order)}
        categories_to_plot.sort(key=lambda name: category_order_map.get(name, len(preferred_category_order) + 1))


        plots_per_row = 2 
        current_row = 0
        current_col = 0
        min_comments_for_plot = 3 

        plotted_count = 0 
        for cat_name in categories_to_plot:
             try:
                  df_cat_subset = df_processed[df_processed[cat_name].eq(1)].copy()

                  df_pos_cat = df_cat_subset[df_cat_subset['duygu_tahmini'].str.lower() == 'positive'].copy()
                  df_neg_cat = df_cat_subset[df_cat_subset['duygu_tahmini'].str.lower() == 'negative'].copy()

                  df_pos_agg_cat = df_pos_cat.groupby(df_pos_cat['hesaplanan_tarih'].dt.date)['duygu_skoru'].mean()
                  df_neg_agg_cat = df_neg_cat.groupby(df_neg_cat['hesaplanan_tarih'].dt.date)['duygu_skoru'].mean()

                  df_pos_agg_cat = df_pos_agg_cat.sort_index()
                  df_neg_agg_cat = df_neg_agg_cat.sort_index()

                  if not df_pos_agg_cat.empty or not df_neg_agg_cat.empty:
                      fig, ax = plt.subplots(figsize=(5, 4)) 

                      if not df_pos_agg_cat.empty:
                          ax.plot(df_pos_agg_cat.index, df_pos_agg_cat.values, marker='o', linestyle='-', color='green', label='Pozitif Ortalama')

                      if not df_neg_agg_cat.empty:
                          ax.plot(df_neg_agg_cat.index, df_neg_agg_cat.values, marker='o', linestyle='-', color='red', label='Negatif Ortalama')

                      ax.set_title(f'{cat_name} Duygu Skoru', fontsize=10) 
                      ax.set_xlabel('Tarih', fontsize=8) 
                      ax.set_ylabel('Ortalama Skor', fontsize=8)
                      ax.legend(fontsize=6)

                      fig.autofmt_xdate()

                      ax.set_ylim(0.1, 1.1) 

                      canvas = FigureCanvasQTAgg(fig)
                      canvas.setMinimumSize(300, 350) 
                      canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                      self._category_sentiment_figures.append(fig)
                      self._category_sentiment_canvases.append(canvas)

                      self._category_sentiment_grid_layout.addWidget(canvas, current_row, current_col)
                      plotted_count += 1

                      current_col += 1
                      if current_col >= plots_per_row:
                          current_col = 0
                          current_row += 1

             except Exception as e:
                 print(f"Hata: Kategori '{cat_name}' için grafik çizilirken beklenmedik hata oluştu: {e}")
                 import traceback
                 traceback.print_exc() 

        if plotted_count > 0:
             self.sentiment_by_category_group.setVisible(True)
             self.scroll_area.setVisible(True)
        else:
             self.sentiment_by_category_group.setVisible(False)


    def closeEvent(self, event):
        """Pencere kapatıldığında çağrılır. Kontrollü kapatma sağlar."""
        # Eğer zaten kapatma sürecindeysek ve bu metod tekrar çağrıldıysa, direkt kabul et.
        if self._is_shutting_down:
            event.accept()
            return

        # Hangi işlemlerin çalıştığını tespit et
        active_workers = []
        if self._is_collecting and self.download_worker:
            active_workers.append((self.download_worker, self.download_thread, "Veri Toplama"))
        if self._is_analyzing and self.sentiment_worker:
            active_workers.append((self.sentiment_worker, self.sentiment_thread, "Duygu Analizi"))
        if self._is_categorizing and self._categorization_worker:
            active_workers.append((self._categorization_worker, self._categorization_thread, "Kategorizasyon"))
        if self._is_reporting and hasattr(self, '_report_worker') and self._report_worker:
            active_workers.append((self._report_worker, self._report_thread, "Rapor Oluşturma"))

        # Eğer çalışan bir işlem yoksa, normal şekilde kapat
        if not active_workers:
            print("Çalışan işlem yok, pencere kapatılıyor.")
            plt.close('all') # Tüm matplotlib figürlerini temizle
            event.accept()
            return

        # Çalışan işlem varsa, kullanıcıyı uyar
        running_processes_str = "\n - ".join([name for _, _, name in active_workers])
        reply = QMessageBox.warning(self, "İşlem Devam Ediyor",
                                    f"Aşağıdaki işlemler devam ediyor:\n - {running_processes_str}\n\nYine de uygulamayı kapatmak istiyor musunuz?\n(İşlemler durdurulmaya çalışılacak)",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            print("Pencere kapatma işlemi kullanıcı tarafından iptal edildi.")
            event.ignore() # Kapatma işlemini iptal et
            return

        # Kullanıcı "Evet" derse, kontrollü kapatmayı başlat
        print("Kullanıcı kapatmayı onayladı. Kapatma süreci başlatılıyor.")
        self._is_shutting_down = True
        event.ignore() # Olayı şimdilik yoksay, biz kendimiz kapatacağız

        self.status_label.setText("Durum: Uygulama kapatılıyor, işlemler durduruluyor...")
        self.stop_button.setEnabled(False)
        QApplication.processEvents() # Arayüzün güncellenmesini sağla

        # Tüm aktif worker'lara durma sinyali gönder
        for worker, _, name in active_workers:
            if hasattr(worker, 'stop'):
                print(f"Durdurma sinyali gönderiliyor: {name}")
                worker.stop()

        # Worker'ların sonlanmasını beklemek için zamanlayıcı kullan
        # Bu, GUI'nin donmasını engeller
        QTimer.singleShot(100, lambda: self._wait_for_threads_and_exit(active_workers))


    def _wait_for_threads_and_exit(self, workers_to_wait_for):
        """
        Worker thread'lerinin bitmesini bekler ve ardından uygulamayı kapatır.
        Bu metod, GUI thread'ini uzun süre bloklamaz.
        """
        all_finished = True
        for _, thread, name in workers_to_wait_for:
            if thread and thread.isRunning():
                # Thread'e bitmesi için kısa bir süre bekle (en fazla 2 saniye)
                if not thread.wait(2000):
                    print(f"UYARI: '{name}' thread'i zamanında sonlanmadı.")
                    all_finished = False

        if all_finished:
            print("Tüm thread'ler başarıyla sonlandı.")
        else:
            print("Bazı thread'ler zamanında sonlanmadı, yine de çıkılıyor.")

        print("Kaynaklar temizleniyor ve uygulama kapatılıyor...")
        plt.close('all') # Kalan tüm matplotlib figürlerini kapat
        QCoreApplication.instance().quit() # Uygulamanın olay döngüsünü sonlandır

    def _wait_for_threads_and_cleanup(self, stoppable_workers, event):
        """Waits for stoppable threads and performs cleanup after user confirms closing."""
        wait_time_ms = 3000
        print(f"Pencere kapatılıyor: Durdurulabilir worker thread'lerinin sonlanması bekleniyor (maks {wait_time_ms}ms/thread)...")
        
        for worker, thread, name in stoppable_workers:
             if thread is not None and thread.isRunning():
                  print(f"  {name} thread bekleniyor...")
                  if not thread.wait(wait_time_ms):
                       print(f"Uyarı: {name} thread zaman aşımına rağmen sonlanmadı.")
                  else:
                       print(f"  {name} thread sonlandı.")

        print("Temizlik işlemleri başlatılıyor (thread referanslarının None yapılması)...")
        self._cleanup_search_thread_refs()
        self._cleanup_download_thread_refs()
        self._cleanup_sentiment_thread_refs()
        self._cleanup_categorization_thread_refs() 
        print("Thread referansları temizlendi.")


        print("Matplotlib figürleri kapatılıyor...")
        self.clear_all_charts()
        print("Matplotlib figürleri kapatıldı.")

        print("Bekleme ve temizlik tamamlandı. Pencere kapatma kabul ediliyor.")
        event.accept() 


    def start_report_generation(self):
        """Rapor oluşturma işlemini başlatır."""
        if self.step4_button is None or not self.step4_button.isEnabled():
            print("Rapor oluşturma başlatılamadı: buton devre dışı.")
            return

        if self._is_reporting:
            self.status_label.setText("Durum: Rapor oluşturma işlemi zaten devam ediyor.")
            return

        if not self._categorization_completed:
            self.status_label.setText("Durum: Önce kategorizasyon işlemini tamamlayın.")
            return

        try:
            csv_file_path = self._categorization_filepath
            if not csv_file_path or not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"Kategorize edilmiş CSV dosyası bulunamadı: {csv_file_path}")

            df = pd.read_csv(csv_file_path, encoding='utf-8', nrows=1)
            existing_columns = set(df.columns)

            keyword = self._safe_query_name.replace('_', ' ').title()

            raw_categories = self.get_categories_from_inputs()
            if not raw_categories:
                raise ValueError("Kategori tanımları bulunamadı")

            category_definitions = []
            for cat in raw_categories:
                category_name = cat["category"].strip()
                if not category_name: 
                    continue

                possible_column_names = [
                    category_name.upper(),  # UPPERCASE
                    category_name.upper().replace(" ", "_"),  
                    category_name.lower(),  # lowercase
                    category_name.lower().replace(" ", "_"),  
                    category_name.title(),  # Title Case
                    category_name.title().replace(" ", "_"),  # 
                ]

                matching_column = None
                for col_name in possible_column_names:
                    if col_name in existing_columns:
                        matching_column = col_name
                        break

                if matching_column:
                    category_definitions.append({
                        "csv_col": matching_column,
                        "display_name": category_name
                    })
                else:
                    print(f"UYARI: '{category_name}' kategorisi için CSV'de eşleşen sütun bulunamadı. Olası sütunlar: {possible_column_names}")

            if not category_definitions:
                raise ValueError("Hiçbir kategori için CSV'de eşleşen sütun bulunamadı")

            self._report_worker = ReportGenerationWorker(
                csv_file_path=csv_file_path,
                keyword=keyword,
                category_definitions=category_definitions
            )
            
            self._report_worker.finished.connect(self._handle_report_completion)
            self._report_worker.error.connect(self._handle_report_error)
            self._report_worker.status_message.connect(lambda msg: self.status_label.setText(f"Durum: {msg}"))
            # progress_updated sinyali artık worker'da yok, o yüzden bu satırı kaldırın.

            self._report_thread = QThread()
            self._report_worker.moveToThread(self._report_thread)
            self._report_thread.started.connect(self._report_worker.run)
            self._report_thread.finished.connect(self._report_thread.deleteLater)
            self._report_thread.finished.connect(self._cleanup_report_thread_refs)
            self._report_thread.start()

            # --- YENİ KISIM: GUI ZAMANLAYICISINI BAŞLAT ---
            self.report_start_time = time.time()
            self.report_timer.start(1000) # Her saniye _update_report_progress_display metodunu çağır
            # ----------------------------------------------

            self._is_reporting = True
            self.update_button_states()
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Rapor Oluşturuluyor: %p%")
            self.status_label.setText("Durum: Rapor oluşturma başlatıldı...")
            
        except Exception as e:
            self.status_label.setText(f"Durum: Rapor oluşturma başlatılamadı: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Rapor oluşturma başlatılamadı: {str(e)}")
            self._is_reporting = False
            self.update_button_states() 
    
    def _update_report_progress_display(self):
        """GUI zamanlayıcısı tarafından çağrılır ve raporlama ilerlemesini günceller."""
        if not self._is_reporting or self.report_start_time is None:
            self.report_timer.stop()
            return

        elapsed_seconds = time.time() - self.report_start_time
        progress = min(int((elapsed_seconds / self.REPORT_ESTIMATED_DURATION) * 100), 99)
        self.progress_bar.setValue(progress)

        remaining_seconds = max(0, self.REPORT_ESTIMATED_DURATION - elapsed_seconds)
        mins, secs = divmod(int(remaining_seconds), 60)
        self.status_label.setText(f"Durum: Rapor oluşturuluyor... Tahmini kalan süre: {mins:02d}:{secs:02d}")

    def _handle_report_completion(self, output_filename):
        """Rapor oluşturma tamamlandığında çağrılır.
        # Note: _is_reporting is set to False at the end of this function in the finally block
        # No need to set it here
        """
        try:
            self.report_timer.stop() # Zamanlayıcıyı durdur
            self._report_completed = True
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("%p%") 
            self.status_label.setText("Durum: Rapor oluşturma tamamlandı!")
            
            QMessageBox.information(self, "İşlem Tamamlandı", 
                                  f"Rapor başarıyla oluşturuldu!\n\n"
                                  f"Dosya: {output_filename}\n\n"
                                  "Bütün işlemler bitti!")
            
        except Exception as e:
            self.status_label.setText(f"Durum: Rapor tamamlandı ancak bir hata oluştu: {str(e)}")
            QMessageBox.warning(self, "Uyarı", f"Rapor oluşturuldu ancak bir hata oluştu: {str(e)}")
        finally:
            self._is_reporting = False 
            self.update_button_states()
            self._cleanup_report_thread_refs()

    def _handle_report_error(self, error_message):
        """Rapor oluşturma sırasında hata oluştuğunda çağrılır."""
        try:
            self.report_timer.stop() # Zamanlayıcıyı durdur
            self.progress_bar.setValue(0) #
            self.progress_bar.setFormat("%p%") #
            self.status_label.setText("Durum: Rapor oluşturulurken hata.")

            QMessageBox.critical(self, "Hata", f"Rapor oluşturulurken hata oluştu:\n{error_message}")
        except Exception as e:
            print(f"Hata mesajı gösterilirken hata oluştu: {e}")
        finally:
            self._is_reporting = False 
            self.update_button_states()
            self._cleanup_report_thread_refs() 

    def _cleanup_report_thread_refs(self):
        """Rapor oluşturma thread'ini temizler."""
        if hasattr(self, '_report_thread') and self._report_thread is not None:
            if self._report_thread.isRunning():
                print("Warning: Report thread is still running during cleanup.")
            self._report_thread.deleteLater()
            self._report_thread = None
        if hasattr(self, '_report_worker') and self._report_worker is not None:
            self._report_worker.deleteLater()
            self._report_worker = None
        if hasattr(self, 'report_progress_bar'):
            pass 


OUTPUT_DIR = "output"
URLS_FILE_TEMPLATE = "{query}_urls.csv"
COMMENTS_FILE_TEMPLATE = "{query}_comments.csv"
STATE_FILE_TEMPLATE = "{query}_state.json"
SENTIMENT_FILE_TEMPLATE = "{query}_sentiment.csv"
CATEGORIZATION_FILE_TEMPLATE = "{query}_categorization.csv"

DEFAULT_OLLAMA_MODEL = "gemma3:12b" 
DEFAULT_BATCH_SIZE = 5 

QSS_STYLE = """
QMainWindow {
    background-color: #ecf0f1;
}

QWidget {
    background-color: #ecf0f1;
    font-family: Arial, sans-serif;
    color: #2c3e50;
}

QLabel {
    color: #2c3e50;
    padding: 2px 0; /* Adjust padding */
}

QLineEdit, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 5px;
    selection-background-color: #3498db;
    selection-color: #ffffff;
}

QLineEdit:focus, QSpinBox:focus {
    border-color: #3498db;
}

/* General Button Style (Applies to control buttons like Stop, Clear) */
QPushButton {
    background-color: #3498db;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 15px;
    margin: 4px;
    min-width: 90px; /* Slightly wider buttons */
}

QPushButton:hover {
    background-color: #2980b9;
}

QPushButton:pressed {
    background-color: #2471a3;
}

QPushButton:disabled { /* Generic disabled style for all QPushButtons */
    background-color: #bdc3c7;
    color: #ecf0f1;
}

/* --- Styles for the Step Buttons --- */
/* Common style for all step buttons */
QPushButton[class~="step-button"] {
    border-radius: 10px; /* More rounded corners for step buttons */
    padding: 8px;
    min-width: 120px; /* Ensure step buttons are wide enough */
    text-align: center;
    font-weight: normal; /* Default normal weight */
    border: 1px solid #bdc3c7; /* Add a subtle border */
}

/* Pending state for step buttons */
QPushButton[class~="step-button"][class~="step-button-pending"] {
    background-color: #ffffff; /* White background */
    color:rgb(44, 80, 52); /* Dark text */
    border-color: #3498db; /* Accent border */
    font-weight: bold; /* Highlight next pending step */
}

/* Active state for step buttons */
QPushButton[class~="step-button"][class~="step-button-active"] {
    background-color: #0b74de; /* StepUI's active color */
    color: white;
    font-weight: bold;
    border-color: #0b74de;
}

/* Completed state for step buttons */
QPushButton[class~="step-button"][class~="step-button-completed"] {
    background-color:rgb(0, 255, 8); /* Black for completed steps */
    color: #ffffff; /* White text on black */
    font-weight: normal;
    border-color: #000000;
}

/* Last-completed state for step buttons */
QPushButton[class~="step-button"][class~="step-button-last-completed"] {
     background-color:rgb(0, 255, 8); /* Black for final completed step */
     color: #ffffff; /* White text on black */
     font-weight: bold;
     border-color: #333333; /* Slightly lighter border */
}

/* Specific disabled style for step buttons. Overrides generic QPushButton:disabled */
QPushButton[class~="step-button"]:disabled {
    background-color: #eeeeee; /* Lighter disabled color for steps */
    color: #777777; /* Darker text for contrast on light bg */
    border-color: #dddddd; /* Lighter border */
    font-weight: normal;
    /* cursor: default; is implicitly handled by Qt for disabled widgets */
}


/* --- Styles for the Lines between Steps --- */
QFrame { /* Base for all QFrames if needed, but step-lines are specific */
    border: none; 
    margin: 0; 
    padding: 0; 
}

/* Base style for step lines */
QFrame[class~="step-line"] {
    height: 3px; /* Match _create_line height */
    background-color: #ccc; /* StepUI's default line color */
}

/* Completed state for step lines */
QFrame[class~="step-line"][class~="line-completed"] {
    background-color: #0b74de; /* StepUI's completed line color */
}

/* Pending state for step lines (when adjacent step is active) */
QFrame[class~="step-line"][class~="line-pending"] {
    background-color: #3498db; /* Use an active color */
}

/* --- Styles for the Step Separator Labels ('>') --- */
QLabel[class~="step-separator-label"] {
    font-size: 20px; /* Adjust size as needed */
    font-weight: bold;
    color: #bdc3c7; /* Default greyish color */
    margin: 0 5px; /* Adjust spacing around the label */
}

QLabel[class~="step-separator-label"][class~="separator-completed"] {
    color:rgb(0, 255, 8); /* Black color when previous step is completed */
}

QLabel[class~="step-separator-label"][class~="separator-pending"] {
     color: #3498db; /* Blue color when adjacent step is active */
}


/* Style for the Stop Button (Optional: make it red) */
QPushButton#stop_button {
    background-color: #e74c3c; /* Red color */
}
QPushButton#stop_button:hover {
    background-color: #c0392b;
}
QPushButton#stop_button:pressed {
    background-color: #a93226;
}

QLabel#status_label {
    font-weight: bold;
    text-align: center; /* Centered text */
    padding: 10px;
    color: #2c3e50; /* Ensure color is set */
}

QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 5px;
    text-align: center;
    height: 25px;
    margin: 5px 0;
    color: #2c3e50; /* Progress text color */
    background-color: #ffffff; /* Background of the bar itself */
}

QProgressBar::chunk {
    background-color: #2ecc71; /* Green fill color (can customize for global vs categorization bar) */
    width: 10px; /* Note: width property on chunk is often for fixed-size chunks, not continuous fill */
}

/* Specific style for the categorization progress bar if needed */
QProgressBar#categorization_progress_bar::chunk {
    background-color: #3498db; /* Blue fill color */
}


QTableWidget {
    background-color: #ffffff; 
    alternate-background-color: #f0faff; 
    selection-background-color: #a6cee3; 
    selection-color: #333333; 
    gridline-color: #e0e0e0; 
    border: 1px solid #d3d3d3; 
    border-radius: 5px; 
    padding: 0px; 
}

QHeaderView::section {
    background-color: #e0e0e0; 
    color: #333333; 
    padding: 6px; 
    border: none; 
    border-bottom: 2px solid #a6cee3; 
    font-weight: bold; 
    text-align: center; 
}

QTableWidget::item {
    padding: 5px; 
    border: none; 
    border-bottom: 1px solid #eeeeee; 
}

QTableWidget::item:selected {
    /* selection-background-color and selection-color QTableWidget'ta ayarlandı */
}

QScrollBar:vertical {
    border: none;
    background: #f0f0f0;
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #c0c0c0;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical {
    border: none;
    background: none;
    height: 0px;
    subcontrol-position: bottom;
    subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical {
    border: none;
    background: none;
    height: 0px;
    subcontrol-position: top;
    subcontrol-origin: margin;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QGroupBox {
    color: #2c3e50;
    font-weight: bold;
    margin-top: 10px;
    border: 1px solid #bdc3c7; /* Add border to group box */
    border-radius: 5px;
    padding-top: 15px; /* Space for title */
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #2c3e50;
    background-color: #ecf0f1; /* Match window background */
}

QScrollArea {
    border: none; /* Remove border from scroll area */
}
"""

def sanitize_filename(query):
    """Arama sorgusunu dosya adı için güvenli hale getirir."""
    safe_chars = re.compile(r'[^\w\s.-_çğıİöşüÇĞIÖŞÜ]')
    sanitized = safe_chars.sub('_', query).strip()
    sanitized = sanitized.replace(' ', '_')
    sanitized = re.sub(r'_{2,}', '_', sanitized)
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized) 
    sanitized = sanitized.strip('_')

    if len(sanitized) > 100:
        sanitized = sanitized[:100].rsplit('_', 1)[0] or sanitized[:100]
        sanitized = sanitized.strip('_')

    return sanitized or "default_query"


if __name__ == '__main__':
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    available_styles = QStyleFactory.keys()
    if 'Fusion' in available_styles:
        QApplication.setStyle('Fusion')

    app.setStyleSheet(QSS_STYLE)

    window = YoutubeToolGUI()
    window.show()

    sys.exit(app.exec())
