from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import numpy as np # Ditambahkan untuk menangani tipe data jika diperlukan

app = Flask(__name__)

# --- Memuat Model dan Encoders saat aplikasi dimulai ---
try:
    model_lr_yang_dimuat = joblib.load('model_regresi_logistik_cuaca.joblib')
    weather_type_encoder_yang_dimuat = joblib.load('Weather_Type_label_encoder.joblib')

    kolom_objek_fitur_yang_dimuat = ['Cloud_Cover', 'Season', 'Location']
    encoders_yang_dimuat = {}
    for kolom in kolom_objek_fitur_yang_dimuat:
        encoders_yang_dimuat[kolom] = joblib.load(f'{kolom}_label_encoder.joblib')
    print("Model dan semua encoder berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: Salah satu file model atau encoder tidak ditemukan: {e}")
    # Anda mungkin ingin menghentikan aplikasi atau menangani ini dengan cara lain
    model_lr_yang_dimuat = None
    weather_type_encoder_yang_dimuat = None
    encoders_yang_dimuat = None
except Exception as e:
    print(f"Terjadi error saat memuat model atau encoder: {e}")
    model_lr_yang_dimuat = None
    weather_type_encoder_yang_dimuat = None
    encoders_yang_dimuat = None


# Urutan fitur yang diharapkan oleh model (sesuai dengan saat pelatihan X)
URUTAN_FITUR = ['Temperature', 'Humidity', 'Wind_Speed', 'Precipitation (%)',
                'Cloud_Cover', 'Atmospheric_Pressure', 'UV_Index', 'Season',
                'Visibility_KM', 'Location']

def praproses_data_input_flask(data_input_dict):
    """
    Melakukan pra-pemrosesan pada data input dari form Flask.
    Mengubah nilai kategorikal menjadi bentuk numerik menggunakan encoder yang dimuat.
    """
    df_input = pd.DataFrame([data_input_dict])

    # Konversi tipe data untuk kolom numerik dan tangani error jika ada
    kolom_numerik = ['Temperature', 'Humidity', 'Wind_Speed', 'Precipitation (%)',
                     'Atmospheric_Pressure', 'UV_Index', 'Visibility_KM']
    for kolom in kolom_numerik:
        try:
            df_input[kolom] = pd.to_numeric(df_input[kolom])
        except ValueError:
            raise ValueError(f"Nilai untuk '{kolom}' harus berupa angka.")

    # Encoding fitur kategorikal
    for kolom, encoder in encoders_yang_dimuat.items():
        if kolom in df_input.columns:
            try:
                # Pastikan input untuk encoder adalah array 1D dari string
                nilai_input_kategorikal = df_input[kolom].astype(str).values
                df_input[kolom] = encoder.transform(nilai_input_kategorikal)
            except ValueError as e:
                # Menangani kasus di mana label tidak dikenal oleh encoder
                raise ValueError(f"Nilai '{df_input[kolom].iloc[0]}' untuk kolom '{kolom}' tidak dikenali. Error: {e}")
        else:
            raise ValueError(f"Kolom kategorikal '{kolom}' yang diharapkan tidak ada dalam input.")

    # Pastikan urutan kolom sesuai dengan yang diharapkan model
    try:
        df_final = df_input[URUTAN_FITUR]
    except KeyError as e:
        raise ValueError(f"Kolom fitur yang hilang dari input: {e}. Pastikan semua fitur ada: {URUTAN_FITUR}")

    return df_final


@app.route('/')
def home():
    """Menampilkan halaman utama dengan formulir input."""
    return render_template('index.html', fitur_kategorikal=encoders_yang_dimuat)

@app.route('/predict', methods=['POST'])
def predict():
    """Menerima input dari form, melakukan prediksi, dan menampilkan hasilnya."""
    if not all([model_lr_yang_dimuat, weather_type_encoder_yang_dimuat, encoders_yang_dimuat]):
        return render_template('index.html',
                               prediksi_teks="Error: Model atau encoder tidak berhasil dimuat. Silakan cek log server.",
                               fitur_kategorikal=encoders_yang_dimuat if encoders_yang_dimuat else {})

    try:
        # Mengambil semua data dari form sebagai dictionary
        data_input_pengguna = request.form.to_dict()

        # Pra-pemrosesan data input
        data_yang_diproses = praproses_data_input_flask(data_input_pengguna)

        # Melakukan prediksi
        prediksi_encoded = model_lr_yang_dimuat.predict(data_yang_diproses)
        prediksi_proba_encoded = model_lr_yang_dimuat.predict_proba(data_yang_diproses)

        # Decode hasil prediksi ke label asli
        label_prediksi_cuaca = weather_type_encoder_yang_dimuat.inverse_transform(prediksi_encoded)
        hasil_prediksi_teks = f"Prediksi Tipe Cuaca: {label_prediksi_cuaca[0]}"

        # Menyiapkan probabilitas untuk ditampilkan (opsional)
        probabilitas_label = {}
        if hasattr(weather_type_encoder_yang_dimuat, 'classes_'):
            for i, kelas in enumerate(weather_type_encoder_yang_dimuat.classes_):
                probabilitas_label[kelas] = f"{prediksi_proba_encoded[0][i]*100:.2f}%"
        else: # Fallback jika classes_ tidak tersedia atau encoder target bermasalah
            probabilitas_label = {"Info": "Tidak dapat menampilkan probabilitas per kelas."}


        return render_template('index.html',
                               prediksi_teks=hasil_prediksi_teks,
                               probabilitas=probabilitas_label,
                               data_input_sebelumnya=data_input_pengguna, # Untuk mengisi kembali form
                               fitur_kategorikal=encoders_yang_dimuat)

    except ValueError as ve: # Menangkap error validasi dari praproses_data_input_flask
        return render_template('index.html',
                               prediksi_teks=f"Error Validasi Input: {str(ve)}",
                               data_input_sebelumnya=request.form.to_dict(),
                               fitur_kategorikal=encoders_yang_dimuat)
    except Exception as e:
        # Menangkap error umum lainnya
        print(f"Terjadi error saat prediksi: {e}") # Log error ke konsol server
        return render_template('index.html',
                               prediksi_teks=f"Terjadi error internal: {str(e)}",
                               data_input_sebelumnya=request.form.to_dict(),
                               fitur_kategorikal=encoders_yang_dimuat)

if __name__ == '__main__':
    app.run(debug=True) # debug=True hanya untuk pengembangan