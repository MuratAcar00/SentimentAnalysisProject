# Reddit Sentiment Analysis Project / Reddit Duygu Analizi Projesi

This project is a Python script that fetches comments from a specified subreddit based on a keyword, performs sentiment analysis on them using a Hugging Face Transformers model, and visualizes the results.

Bu proje, belirtilen bir subredditten bir anahtar kelimeye göre yorumları çeken, Hugging Face Transformers modeli kullanarak bu yorumlar üzerinde duygu analizi yapan ve sonuçları görselleştiren bir Python betiğidir.

---

## ![UK Flag](https://flagcdn.com/w20/gb.png) English Documentation

### 🚀 Description

This script connects to the Reddit API via PRAW to search for submissions related to a specific query within a given subreddit. It then collects the top comments from these submissions, cleans the text data, and performs sentiment analysis using the `savasy/bert-base-turkish-sentiment-cased` model, which is pre-trained for the Turkish language.

Finally, it generates two visualizations:
1.  A **Word Cloud** to show the most frequently used words in the comments.
2.  A **Bar Chart** to display the distribution of sentiments (positive, negative, neutral).

### ✨ Features

-   Securely loads API credentials using a `.env` file.
-   Fetches data from any public subreddit.
-   Cleans text data by removing URLs, mentions, and special characters.
-   Performs advanced sentiment analysis with a pre-trained Turkish BERT model.
-   Visualizes results with a Word Cloud and a sentiment distribution chart.

### 🛠️ Tech Stack

-   **Python 3.x**
-   **PRAW** (Python Reddit API Wrapper)
-   **Pandas** (For data manipulation)
-   **Hugging Face Transformers** (For NLP and Sentiment Analysis)
-   **PyTorch** (As the backend for Transformers)
-   **WordCloud** (For visualization)
-   **Matplotlib & Seaborn** (For plotting charts)
-   **python-dotenv** (For managing environment variables)

### ⚙️ Setup and Installation

#### 1. Prerequisites
- Make sure you have [Python 3.8+](https://www.python.org/downloads/) installed on your system.

#### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

#### 3. Create a Virtual Environment (Recommended)
It's a best practice to create a virtual environment to keep project dependencies isolated.
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 4. Install Dependencies
Install all the required libraries from a `requirements.txt` file.
```bash
pip install -r requirements.txt
```
*(If you don't have a `requirements.txt` file, you can create one with `pip freeze > requirements.txt` after installing the libraries manually.)*
The required libraries are: `praw pandas transformers torch wordcloud matplotlib seaborn python-dotenv`


### 🔑 Configuration

The script requires Reddit API credentials to function.

1.  **Get Reddit API Keys:**
    -   Go to [Reddit's App Preferences](https://www.reddit.com/prefs/apps).
    -   Click "are you a developer? create an app...".
    -   Fill out the form:
        -   **name**: `SentimentAnalysisProject` (or any name)
        -   Select **script**.
        -   **redirect uri**: `http://localhost:8080`
    -   Click "create app". You will get a `CLIENT_ID` (under the app name) and a `CLIENT_SECRET`.

2.  **Create the `.env` File:**
    -   In the root directory of the project, create a file named `.env`.
    -   Add your credentials to this file as follows:
    ```ini
    REDDIT_CLIENT_ID="YOUR_CLIENT_ID"
    REDDIT_CLIENT_SECRET="YOUR_CLIENT_SECRET"
    REDDIT_USER_AGENT="SentimentAnalysisProject by u/YOUR_REDDIT_USERNAME"
    ```
    -   Replace the placeholders with your actual credentials.

### ▶️ Usage

1.  **Customize the Search:**
    -   Open the `SentimentAnalysisProject.py` file.
    -   Modify the following variables to change the search scope:
    ```python
    SUBREDDIT_NAME = 'Turkey'
    SEARCH_QUERY = 'yapay zeka'
    POST_LIMIT = 25
    ```

2.  **Run the Script:**
    -   Execute the script from your terminal:
    ```bash
    python SentimentAnalysisProject.py
    ```
The script will run, and upon completion, it will display the generated Word Cloud and sentiment distribution chart.

---

## ![Turkish Flag](https://flagcdn.com/w20/tr.png) Türkçe Dokümantasyon

### 🚀 Açıklama

Bu betik, PRAW aracılığıyla Reddit API'sine bağlanarak belirli bir subreddit içinde bir anahtar kelime ile ilgili gönderileri arar. Daha sonra bu gönderilerden en popüler yorumları toplar, metin verilerini temizler ve Türkçe dili için önceden eğitilmiş olan `savasy/bert-base-turkish-sentiment-cased` modelini kullanarak duygu analizi gerçekleştirir.

Son olarak, iki adet görselleştirme oluşturur:
1.  Yorumlarda en sık kullanılan kelimeleri göstermek için bir **Kelime Bulutu**.
2.  Duygu dağılımını (pozitif, negatif, nötr) göstermek için bir **Sütun Grafiği**.

### ✨ Özellikler

-   `.env` dosyası kullanarak API anahtarlarını güvenli bir şekilde yükler.
-   Herkese açık herhangi bir subredditten veri çeker.
-   URL'leri, etiketleri ve özel karakterleri kaldırarak metin verilerini temizler.
-   Önceden eğitilmiş bir Türkçe BERT modeli ile gelişmiş duygu analizi yapar.
-   Sonuçları bir Kelime Bulutu ve duygu dağılım grafiği ile görselleştirir.

### 🛠️ Kullanılan Teknolojiler

-   **Python 3.x**
-   **PRAW** (Python Reddit API Wrapper)
-   **Pandas** (Veri manipülasyonu için)
-   **Hugging Face Transformers** (NLP ve Duygu Analizi için)
-   **PyTorch** (Transformers için altyapı)
-   **WordCloud** (Görselleştirme için)
-   **Matplotlib & Seaborn** (Grafik çizimi için)
-   **python-dotenv** (Ortam değişkenlerini yönetmek için)

### ⚙️ Kurulum

#### 1. Ön Gereksinimler
- Sisteminizde [Python 3.8+](https://www.python.org/downloads/) sürümünün kurulu olduğundan emin olun.

#### 2. Projeyi Klonlama
```bash
git clone [https://github.com/kullanici-adiniz/proje-adiniz.git](https://github.com/kullanici-adiniz/proje-adiniz.git)
cd proje-adiniz
```

#### 3. Sanal Ortam Oluşturma (Tavsiye Edilir)
Proje bağımlılıklarını izole tutmak için bir sanal ortam oluşturmak en iyi yöntemdir.
```bash
# Windows için
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux için
python3 -m venv venv
source venv/bin/activate
```

#### 4. Bağımlılıkları Yükleme
Gerekli tüm kütüphaneleri bir `requirements.txt` dosyasından yükleyin.
```bash
pip install -r requirements.txt
```
*(Eğer bir `requirements.txt` dosyanız yoksa, kütüphaneleri manuel olarak kurduktan sonra `pip freeze > requirements.txt` komutuyla oluşturabilirsiniz.)*
Gerekli kütüphaneler: `praw pandas transformers torch wordcloud matplotlib seaborn python-dotenv`

### 🔑 Yapılandırma

Betiğin çalışması için Reddit API anahtarları gereklidir.

1.  **Reddit API Anahtarlarını Alma:**
    -   [Reddit'in Uygulama Tercihleri](https://www.reddit.com/prefs/apps) sayfasına gidin.
    -   "are you a developer? create an app..." butonuna tıklayın.
    -   Formu doldurun:
        -   **name**: `SentimentAnalysisProject` (veya herhangi bir isim)
        -   **script** seçeneğini işaretleyin.
        -   **redirect uri**: `http://localhost:8080`
    -   "create app" butonuna tıklayın. Size bir `CLIENT_ID` (uygulama adının altında) ve bir `CLIENT_SECRET` verilecektir.

2.  **`.env` Dosyasını Oluşturma:**
    -   Projenin ana dizininde `.env` adında bir dosya oluşturun.
    -   Anahtarlarınızı bu dosyaya aşağıdaki gibi ekleyin:
    ```ini
    REDDIT_CLIENT_ID="CLIENT_ID_BILGINIZ"
    REDDIT_CLIENT_SECRET="CLIENT_SECRET_BILGINIZ"
    REDDIT_USER_AGENT="SentimentAnalysisProject by u/REDDIT_KULLANICI_ADINIZ"
    ```
    -   Boşlukları kendi gerçek bilgilerinizle doldurun.

### ▶️ Kullanım

1.  **Aramayı Özelleştirme:**
    -   `SentimentAnalysisProject.py` dosyasını açın.
    -   Arama kapsamını değiştirmek için aşağıdaki değişkenleri düzenleyin:
    ```python
    SUBREDDIT_NAME = 'Turkey'
    SEARCH_QUERY = 'yapay zeka'
    POST_LIMIT = 25
    ```

2.  **Betiği Çalıştırma:**
    -   Terminalinizden betiği çalıştırın:
    ```bash
    python SentimentAnalysisProject.py
    ```
Betik çalışacak ve tamamlandığında oluşturulan Kelime Bulutu ile duygu dağılımı grafiğini gösterecektir.
