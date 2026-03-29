# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
Institusi pendidikan sering menghadapi kesulitan dalam memonitor dan meningkatkan performa siswa secara konsisten. Dengan banyaknya variabel yang mempengaruhi hasil akademik (seperti kebiasaan belajar, kehadiran, dan faktor sosial), diperlukan pendekatan berbasis data untuk membantu pengambilan keputusan yang lebih akurat.

Proyek ini bertujuan untuk membangun sistem machine learning yang mampu memprediksi performa siswa serta menyediakan dashboard untuk monitoring dan analisis.

### Permasalahan Bisnis
Permasalahan Bisnis
- Sulit mengidentifikasi siswa yang berpotensi memiliki performa rendah sejak dini
- Tidak adanya sistem berbasis data untuk mendukung keputusan akademik
- Kurangnya visibilitas terhadap faktor-faktor yang mempengaruhi performa siswa

### Cakupan Proyek
- Exploratory Data Analysis (EDA) terhadap data performa siswa
- Feature engineering untuk meningkatkan kualitas model
- Training model machine learning (Random Forest)
- Penyimpanan model dan artifacts (encoder, feature list, bounds)
- Pembuatan prototype aplikasi prediksi (app.py)
- Pembuatan dashboard analitik (dashboard.py) menggunakan Streamlit

### Persiapan

Sumber data: [Students_Performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Setup environment:
```
# Clone repository
git clone <your-repo-url>
cd <your-project-folder>

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit apps
streamlit run app.py
streamlit run dashboard.py
```

## Business Dashboard
Dashboard dibuat menggunakan Streamlit untuk memberikan insight terkait performa siswa secara interaktif.

Access Public : [Students_Performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Fitur utama dashboard: 

- Distribusi nilai siswa
- Analisis faktor-faktor yang mempengaruhi performa
- Visualisasi data (histogram, correlation, dll)
- Monitoring performa secara keseluruhan

## Menjalankan Sistem Machine Learning
Prototype sistem machine learning dibuat pada app.py

Fungsi utama:

Input data siswa
- Preprocessing otomatis:
Imputation (median)
Winsorization (berdasarkan bounds training)
Feature engineering (approval rate)
- Prediksi status siswa menggunakan model Random Forest terbaik

Cara menjalankan:

```
streamlit run dashboard.py
```

deployment notes
```
Pastikan file berikut tersedia:
best_model.pkl
bounds.pkl
numeric_cols.pkl
feature_names.pkl
label_encoder.pkl
```

## Conclusion
Dengan memanfaatkan machine learning, institusi pendidikan dapat:

- Memprediksi performa siswa secara lebih akurat
- Mengidentifikasi risiko lebih awal
- Mengoptimalkan strategi pembelajaran

Model yang dibangun mampu menangkap pola dari data historis dan memberikan insight yang actionable.

### Rekomendasi Action Items
- Implementasikan sistem monitoring performa siswa secara real-time
- Gunakan hasil prediksi untuk intervensi dini pada siswa berisiko
- Integrasikan sistem dengan database akademik internal
