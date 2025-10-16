# -*- coding: utf-8 -*-
"""
Bu betik, Reddit API'sini kullanarak belirli bir subreddit içerisinde arama yapar,
toplanan yorumların duygu analizini gerçekleştirir ve sonuçları görselleştirir.
API anahtarları, güvenlik için .env dosyasından okunur.
"""

# --- GEREKLİ KÜTÜPHANELERİN İÇERİ AKTARILMASI ---
import praw                     # Reddit API'si için PRAW kütüphanesi
import pandas as pd             # Veri işleme ve analiz için Pandas
import re                       # Metin temizleme için Regular Expressions
import os                       # İşletim sistemiyle etkileşim (ortam değişkenleri)
from dotenv import load_dotenv  # .env dosyasını okumak için
from transformers import pipeline # HuggingFace duygu analizi modeli için
import matplotlib.pyplot as plt # Görselleştirme için Matplotlib
from wordcloud import WordCloud # Kelime bulutu oluşturmak için
import seaborn as sns           # Daha estetik grafikler için Seaborn


# --- ADIM 1: ORTAM DEĞİŞKENLERİNİ YÜKLEME VE API BAĞLANTISI ---

print("Adım 1: API anahtarları yükleniyor ve Reddit'e bağlanılıyor...")
# .env dosyasındaki değişkenleri sisteme yükle
load_dotenv()

# API anahtarlarını .env dosyasından güvenli bir şekilde oku
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Anahtarların başarıyla okunup okunmadığını kontrol et
if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    raise ValueError("Lütfen .env dosyanızdaki Reddit API bilgilerinizi kontrol edin!")

# PRAW kullanarak Reddit API'sine bağlan
try:
    reddit = praw.Reddit(client_id=CLIENT_ID,
                         client_secret=CLIENT_SECRET,
                         user_agent=USER_AGENT)
    print("✓ Reddit API bağlantısı başarıyla kuruldu.")
except Exception as e:
    print(f"HATA: Reddit'e bağlanılamadı. Hata detayı: {e}")
    exit()


# --- ADIM 2: VERİ ÇEKME AYARLARI VE İŞLEMİ ---

print("\nAdım 2: Reddit'ten veriler çekiliyor...")
# Arama yapılacak subreddit, anahtar kelime ve limitleri belirle
SUBREDDIT_NAME = 'Turkey'
SEARCH_QUERY = 'yapay zeka'
POST_LIMIT = 25  # Aranacak gönderi sayısı
COMMENT_LIMIT_PER_POST = 20 # Her gönderiden alınacak maksimum yorum sayısı

comment_list = []

try:
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    # Belirtilen subredditte arama yap ve en popüler gönderileri al
    for submission in subreddit.search(SEARCH_QUERY, sort='top', limit=POST_LIMIT):
        submission.comments.replace_more(limit=0) # "devamını yükle" yorumlarını atla
        # Her gönderideki yorumları listeye ekle
        for comment in submission.comments.list()[:COMMENT_LIMIT_PER_POST]:
            comment_list.append(comment.body)
    
    # Toplanan yorumları bir Pandas DataFrame'e dönüştür
    df = pd.DataFrame(comment_list, columns=['Yorumlar'])
    print(f"✓ {len(df)} adet yorum başarıyla çekildi.")

except Exception as e:
    print(f"HATA: Veri çekme sırasında bir sorun oluştu. Hata detayı: {e}")
    df = pd.DataFrame() # Hata durumunda boş bir DataFrame oluştur


# --- ADIM 3: VERİ TEMİZLEME ---

# Eğer hiç yorum çekilemediyse sonraki adımları atla
if not df.empty:
    print("\nAdım 3: Metin verileri temizleniyor...")
    def clean_text(text):
        """Metinleri link, kullanıcı adı, hashtag ve özel karakterlerden temizler."""
        text = re.sub(r'http\S+', '', text)    # Linkleri kaldır
        text = re.sub(r'u/\w+|r/\w+', '', text) # Reddit kullanıcı/subreddit adlarını kaldır
        text = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]', '', text) # Harf ve boşluk dışındaki her şeyi kaldır
        text = text.lower()                   # Tüm metni küçük harfe çevir
        text = text.strip()                   # Başındaki ve sonundaki boşlukları kaldır
        return text

    df['Temiz_Yorumlar'] = df['Yorumlar'].apply(clean_text)
    # Temizlendikten sonra boş kalan satırları kaldır
    df.dropna(subset=['Temiz_Yorumlar'], inplace=True)
    df = df[df['Temiz_Yorumlar'] != '']
    print("✓ Temizleme işlemi tamamlandı.")


# --- ADIM 4: DUYGU ANALİZİ (HUGGINGFACE TRANSFORMERS) ---

if not df.empty:
    print("\nAdım 4: Duygu analizi yapılıyor... (Bu işlem biraz zaman alabilir)")
    try:
        # Türkçe için önceden eğitilmiş duygu analizi modelini yükle
        sentiment_pipeline = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased", truncation = True)
        
        # Temizlenmiş yorumlara modeli uygula ve sadece duygu etiketini ('positive', 'negative') al
        df['Duygu'] = df['Temiz_Yorumlar'].apply(lambda text: sentiment_pipeline(text)[0]['label'])
        
        print("✓ Duygu analizi tamamlandı.")
        print("\nDuygu Dağılımı Sonuçları:")
        print(df['Duygu'].value_counts())
    except Exception as e:
        print(f"HATA: Duygu analizi sırasında bir sorun oluştu. Hata detayı: {e}")


# --- ADIM 5: SONUÇLARI GÖRSELLEŞTİRME ---

if not df.empty and 'Duygu' in df.columns:
    print("\nAdım 5: Sonuçlar görselleştiriliyor...")
    
    # a) Kelime Bulutu (WordCloud)
    all_comments = " ".join(comment for comment in df['Temiz_Yorumlar'])
    stopwords_tr = ["ve", "ile", "ama", "bir", "cok", "daha", "bu", "icin", "mi", "ki", "ne", "de", "da", "şey", "çok", "olarak", "ben", "o", "var", "yok"]
    
    wordcloud = WordCloud(width=1200, height=600, 
                          background_color='white', 
                          stopwords=stopwords_tr,
                          min_font_size=10).generate(all_comments)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"'{SEARCH_QUERY}' Konusunda En Sık Geçen Kelimeler", fontsize=16)
    plt.show()

    # b) Duygu Dağılımı Grafiği
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Duygu', data=df, palette="viridis", order = df['Duygu'].value_counts().index)
    plt.title(f"'{SEARCH_QUERY}' Hakkındaki Yorumların Duygu Dağılımı", fontsize=16)
    plt.xlabel('Duygu Durumu', fontsize=12)
    plt.ylabel('Yorum Sayısı', fontsize=12)
    plt.show()
    
    print("✓ Görselleştirme tamamlandı. Proje bitti!")

else:
    print("\nAnaliz edilecek veya görselleştirilecek veri bulunamadı. Lütfen arama kriterlerinizi kontrol edin.")