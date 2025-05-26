# Prediksi_cuaca
Aplikasi web sederhana ini dibangun menggunakan Flask untuk memprediksi tipe cuaca (misalnya, Cerah, Hujan, Berawan, Bersalju) berdasarkan berbagai fitur meteorologi yang dimasukkan oleh pengguna. Prediksi dilakukan menggunakan model machine learning Regresi Logistik yang telah dilatih sebelumnya pada dataset cuaca dan disimpan dalam format

## 📝 Daftar Isi

* [Fitur Utama](#-fitur-utama)
* [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
* [Instalasi dan Setup](#-instalasi-dan-setup)
    * [Prasyarat](#prasyarat)
    * [Langkah-langkah Instalasi](#langkah-langkah-instalasi)
* [Cara Menjalankan Aplikasi](#-cara-menjalankan-aplikasi)
* [Cara Menggunakan Aplikasi](#-cara-menggunakan-aplikasi)
* [Detail Model](#-detail-model)
* [Kontributor](#-kontributor)

## 🛠️ Teknologi yang Digunakan

* **Python**: Bahasa pemrograman utama.
* **Flask**: Framework web micro untuk backend.
* **Scikit-learn**: Library untuk model machine learning (Regresi Logistik, LabelEncoder).
* **Pandas**: Untuk manipulasi dan analisis data.
* **Joblib**: Untuk menyimpan dan memuat model machine learning.
* **Numpy**: Untuk operasi numerik.
* **HTML/CSS**: Untuk struktur dan styling halaman web.
* **JavaScript**: Untuk interaktivitas di sisi klien.
* **Chart.js**: Library JavaScript untuk membuat grafik interaktif.

## ✨ Fitur Utama

* **Prediksi Cuaca**: Memprediksi tipe cuaca (misalnya, Cerah, Hujan, Berawan, Bersalju) berdasarkan input pengguna.
* **Antarmuka Web**: Formulir input yang ramah pengguna untuk memasukkan data fitur cuaca.
* **Visualisasi Data**: Menampilkan grafik batang probabilitas untuk setiap tipe cuaca menggunakan Chart.js.
* **Backend Flask**: Dibangun dengan framework Flask yang ringan dan mudah dikembangkan.
* **Model Regresi Logistik**: Menggunakan model machine learning yang telah dilatih untuk melakukan prediksi.

### Prasyarat

* Python 3.x (disarankan versi 3.8 atau lebih baru)
* PIP (Python package installer)
* Git (untuk clone repositori)
* Web browser (Chrome, Firefox, dll.)

### Langkah-langkah Instalasi

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/EskelandLab/ANDA](https://github.com/EskelandLab/ANDA)
    cd [nama_folder_proyek_anda]
    ```

2.  **Buat dan aktifkan virtual environment (disarankan):**
    ```bash
    python -m venv .venv
    ```
    * Untuk Windows:
        ```bash
        .venv\Scripts\activate
        ```
    * Untuk macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Instal semua dependensi yang dibutuhkan:**
    (Anda bisa membuat file `requirements.txt` dengan menjalankan `pip freeze > requirements.txt` setelah menginstal semua library di virtual environment Anda, lalu pengguna lain bisa menginstal dengan `pip install -r requirements.txt`)
    ```bash
    pip install Flask pandas scikit-learn joblib numpy
    ```
## ▶️ Cara Menjalankan Aplikasi

1.  Pastikan Anda berada di direktori utama proyek dan virtual environment sudah aktif (jika menggunakan).
2.  Jalankan aplikasi Flask menggunakan perintah berikut di terminal:
    ```bash
    python app.py
    ```
3.  Jika berhasil, server pengembangan Flask akan berjalan. Anda akan melihat output seperti:
    ```
    * Serving Flask app 'app.py'
    * Debug mode: on
    * Running on [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```
4.  Buka web browser Anda dan akses alamat `http://127.0.0.1:5000/`.

*(Jika Anda menggunakan port yang berbeda, sesuaikan alamat di atas.)*

## 🚀 Cara Menggunakan Aplikasi

1.  Setelah aplikasi terbuka di browser, Anda akan melihat formulir input.
2.  Isi semua field yang diminta dengan data meteorologi yang relevan (Suhu, Kelembapan, Kecepatan Angin, dll.).
3.  Klik tombol "**Prediksi Cuaca**".
4.  Aplikasi akan menampilkan hasil prediksi tipe cuaca beserta tabel probabilitas per kelas dan grafik batang yang memvisualisasikan probabilitas tersebut.

## 🤖 Detail Model

* **Algoritma**: Regresi Logistik (dari Scikit-learn).
* **Dataset**: (https://www.kaggle.com/datasets/nikhil7280/weather-type-classification).
* **Fitur yang Digunakan**: Temperature, Humidity, Wind_Speed, Precipitation (%), Cloud_Cover, Atmospheric_Pressure, UV_Index, Season, Visibility_KM, Location.
* **Target Variabel**: Weather_Type (misalnya: Rainy, Cloudy, Sunny, Snowy).
* **Pra-pemrosesan**: Fitur kategorikal (Cloud_Cover, Season, Location, Weather_Type) diubah menjadi representasi numerik menggunakan `LabelEncoder`.

## 🧑‍💻 Kontributor

* **[Agil Irman Fadri]** - [12250314181]
* **[Nur Futri Ayu Jelita]** - [12250320374]

