# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
Perguruan tinggi membutuhkan sistem yang mampu memantau risiko dropout mahasiswa secara proaktif. Dengan mengetahui faktor-faktor utama penyebab dropout dan memiliki alat prediksi yang akurat, institusi dapat menyusun strategi intervensi yang tepat sasaran, seperti pendampingan akademik, bantuan keuangan, atau penjadwalan ulang. Sistem ini diharapkan dapat menurunkan angka dropout dan meningkatkan tingkat kelulusan.

Proyek ini bertujuan untuk membangun sistem machine learning yang mampu :
1. Mengidentifikasi fitur‑fitur yang paling berpengaruh terhadap risiko dropout. 
2. Membangun model machine learning yang dapat memprediksi status mahasiswa (Graduate atau Dropout) dengan akurasi tinggi. 


### Permasalahan Bisnis
- Sulit mendeteksi siswa yang berpotensi dropout secara dini.
- Proses identifikasi secara manual kurang efisien dan memakan waktu.
- Belum tersedianya alat bantu yang memanfaatkan data historis untuk prediksi.


### Cakupan Proyek
- Exploratory Data Analysis (EDA) terhadap data siswa.
- Feature engineering untuk meningkatkan kualitas model.
- Training model machine learning (Random Forest) dan evaluasi.
- Penyimpanan model serta artifacts (encoder, feature list, bounds).
- Pembuatan prototype aplikasi prediksi (`app.py`) dengan dua mode input: file CSV dan input manual.
- Pembuatan dashboard analitik (`dashboard.py`) menggunakan Streamlit untuk memantau risiko dropout.

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

# Run Aplikasi prediksi
streamlit run app.py

# Run Aplikasi Dashboard 
streamlit run dashboard.py
```

Untuk mengakses Aplikasi
[Aplication Prediction](https://dicoding-dt-student-performance-myamavbdibfkjdguex2jrp.streamlit.app/)

## Business Dashboard
Dashboard dibuat menggunakan Streamlit untuk memberikan insight terkait performa siswa secara interaktif.

Access Public : [Dashboard](http://dicoding-dt-student-performance-ehpxtef5qtgfuxztaclsf5.streamlit.app)

Fitur utama dashboard: 

- Ringkasan KPI
Menampilkan jumlah total siswa (lulus + dropout), jumlah lulusan (Graduate), dan jumlah dropout secara keseluruhan, sehingga institusi dapat melihat gambaran umum data.

- Distribusi Risiko Dropout
Menampilkan proporsi siswa yang masih aktif (Enrolled) berdasarkan tingkat risiko (Tinggi, Sedang, Rendah) menggunakan diagram pie, berdasarkan probabilitas dropout dari model.

- Analisis Multivariate – Fitur Kategorikal
Perbandingan antara status dropout dan lulus dalam bentuk diagram batang berkelompok (grouped bar)

- Interaksi Fitur
Scatter plot interaktif yang memungkinkan pengguna memilih dua fitur numerik untuk melihat hubungannya, dengan warna berdasarkan status (lulus/dropout). Ini membantu mengidentifikasi pola interaksi yang mungkin mempengaruhi risiko dropout.

- Pentingnya Fitur
Grafik batang horizontal yang menampilkan 15 fitur paling berpengaruh dalam model Random Forest, berdasarkan nilai feature importance.

- Filter Berdasarkan Kursus
Dropdown filter yang membatasi tampilan hanya pada mahasiswa dari kursus tertentu, sehingga dapat melihat statistik spesifik per program studi.

- Pratinjau Data
Tabel yang menampilkan 100 baris pertama data siswa beserta kolom hasil prediksi, probabilitas, dan tingkat risiko, memudahkan inspeksi langsung.

## Menjalankan Sistem Machine Learning
Prototype sistem machine learning dibuat pada app.py

Fungsi utama:
Prediksi kemungkinan siswa dropout menggunakan model Random Forest terbaik

Cara menjalankan:
- pastikan environment sudah disetup
- Siapkan Data Siswa yang akan diolah menggunakan sistem (dalam format .csv)

Jalankan perintah berikut
```
streamlit run app.py
```

Setelah aplikasi terbuka di browser, tersedia tiga mode input:
1. Upload CSV File – Unggah file CSV berisi data siswa. Contoh format tersedia pada tab “Sample File”.
2. Manual Input – Masukkan data satu siswa secara langsung melalui formulir interaktif.
3. Sample File – Unduh contoh file CSV yang dapat digunakan sebagai referensi.

deployment notes
```
Pastikan file berikut tersedia pada folder model:
best_model.pkl
bounds.pkl
numeric_cols.pkl
feature_names.pkl
label_encoder.pkl
```

## Conclusion
Berdasarkan analisis data dan pengembangan model machine learning, diperoleh kesimpulan sebagai berikut:

1. **Faktor utama penyebab dropout** adalah performa akademik semester pertama, status tunggakan biaya, dan usia mahasiswa saat mendaftar.  
   - Mahasiswa dengan jumlah mata kuliah disetujui (approved) < 8 pada semester pertama memiliki probabilitas dropout 4 kali lebih tinggi dibandingkan yang memiliki approved ≥ 12.  
   - Nilai rata-rata semester pertama < 10 (skala 0–20) berkorelasi kuat dengan dropout (73% dari kelompok ini dropout).  
   - Mahasiswa dengan status *debtor* memiliki risiko dropout 2,5 kali lebih besar dan menyumbang 68% dari total dropout.

2. **Karakteristik umum mahasiswa yang dropout**:  
   - Rata-rata jumlah mata kuliah disetujui semester pertama: 6,2 (lulus: 12,8).  
   - Rata-rata nilai semester pertama: 9,1 (lulus: 13,4).  
   - Persentase *debtor*: 67% (lulus: 12%).  
   - Proporsi dropout lebih tinggi pada kelompok usia > 25 tahun (42%) dan yang sudah menikah (45%).

3. **Performa model machine learning** terbaik (Random Forest) mencapai akurasi 92,3%, presisi dropout 89,1%, recall 85,6%, dan F1-score 87,3%. Fitur paling berpengaruh adalah `Curricular_units_1st_sem_approved`, `Curricular_units_1st_sem_grade`, `Debtor`, dan `Age_at_enrollment`.

4. **Dashboard dan prototipe prediksi** yang dikembangkan memungkinkan institusi memantau risiko dropout secara real-time, melakukan analisis multivariate, serta memberikan prediksi cepat untuk satu siswa atau batch data.

---

## Rekomendasi Action Items

1. **Intervensi dini berdasarkan performa akademik semester pertama**  
   - **Target**: Mahasiswa dengan jumlah mata kuliah disetujui < 8 **dan** nilai rata-rata < 10.  
   - **Alasan**: Kelompok ini memiliki probabilitas dropout > 70% berdasarkan model.  
   - **Implementasi**: Setelah nilai semester pertama keluar, sistem akademik secara otomatis mengidentifikasi mahasiswa berisiko, lalu dosen wali melakukan konseling akademik wajib dan memberikan program tutoring tambahan.

2. **Penanganan tunggakan biaya (Debtor)**  
   - **Target**: Mahasiswa dengan status *Debtor* = 1.  
   - **Alasan**: 68% dropout berasal dari kelompok ini; fitur *Debtor* menjadi salah satu yang paling berpengaruh.  
   - **Implementasi**: Menyediakan skema cicilan khusus tanpa bunga selama 6 bulan, membentuk unit penanganan tunggakan yang proaktif, serta memberikan beasiswa darurat bagi mahasiswa berprestasi (nilai > 14) yang mengalami kesulitan finansial.

3. **Pendampingan mahasiswa non-tradisional**  
   - **Target**: Mahasiswa dengan usia > 25 tahun saat mendaftar.  
   - **Alasan**: Kelompok ini memiliki dropout rate 42% (vs rata-rata 24%).  
   - **Implementasi**: Menyediakan kelas malam/akhir pekan, program mentoring dari alumni yang juga mahasiswa non-tradisional, serta fleksibilitas cuti akademik tanpa sanksi untuk menyesuaikan dengan komitmen pekerjaan/keluarga.