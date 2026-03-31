# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
Institusi pendidikan sering menghadapi kesulitan dalam memonitor dan meningkatkan performa siswa secara konsisten. Dengan banyaknya variabel yang mempengaruhi hasil akademik (seperti kebiasaan belajar, kehadiran, dan faktor sosial), diperlukan pendekatan berbasis data untuk membantu pengambilan keputusan yang lebih akurat.

Proyek ini bertujuan untuk membangun sistem machine learning yang mampu :
1. Mengidentifikasi siswa yang berisiko dropout lebih awal, 
2. Memberikan bimbingan atau intervensi tepat waktu, 
3. Meningkatkan retensi dan kualitas lulusan

### Permasalahan Bisnis
- Sulit mendeteksi siswa yang berpotensi dropout
- Proses Identifikasi secara manual kurang efisien dan memakan waktu

### Cakupan Proyek
- Exploratory Data Analysis (EDA) terhadap data siswa
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

Access Public : [Students_Performance](http://dicoding-dt-student-performance-ehpxtef5qtgfuxztaclsf5.streamlit.app)

Fitur utama dashboard: 

- Distribusi nilai siswa
- Analisis faktor-faktor yang mempengaruhi performa
- Visualisasi data (histogram, correlation, dll)
- Monitoring performa secara keseluruhan

## Menjalankan Sistem Machine Learning
Prototype sistem machine learning dibuat pada app.py

Fungsi utama:
Prediksi kemungkinan siswa dropout menggunakan model Random Forest terbaik

Cara menjalankan:
pastikan environment sudah disetup
Generate File Prediksi terlebih dahulu
```
streamlit run app.py
```
Untuk Dashboard dapat gunakan
```
streamlit run dashboard.py
```

deployment notes
```
Pastikan file berikut tersedia pada folder:
best_model.pkl
bounds.pkl
numeric_cols.pkl
feature_names.pkl
label_encoder.pkl
```

## Conclusion
- Random Forest baseline menunjukkan performa baik dengan ROC AUC > 0.95
- Threshold prediksi dapat disesuaikan untuk memaksimalkan deteksi Dropout
- Dashboard dan prototype memungkinkan institusi untuk memantau siswa dan memberikan intervensi tepat waktu sebelum mereka dropout 

### Rekomendasi Action Items
- Implementasikan sistem monitoring performa siswa secara real-time
- Gunakan hasil prediksi untuk intervensi dini pada siswa berisiko
- Integrasikan sistem dengan database akademik internal
