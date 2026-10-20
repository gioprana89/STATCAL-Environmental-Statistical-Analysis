# STATCAL ONLINE — Environmental Statistical Analysis (Versi 1)

Aplikasi Streamlit berbahasa Indonesia untuk pembelajaran dan penelitian: **analisis korelasi, regresi linear sederhana/berganda, dan uji beda parametrik/nonparametrik**. Dirancang sebagai aplikasi ketiga dalam seri laboratorium Statistika Lingkungan STATCAL ONLINE.

## Fitur utama

- **Sumber data:** Excel (`.xlsx`, pilih sheet dan nomor baris header), CSV, atau dua dataset simulasi bawaan.
- **Pengaturan global:** filter satu atau lebih kategori, taraf signifikansi α = 0,01 / 0,05 / 0,10, resolusi PNG 150 / 200 / 300 DPI.
- **Split:** jalankan analisis terpisah menurut maksimal dua kategori, termasuk kombinasi Tahun × Jenis Hari. Maksimal 16 kombinasi untuk menjaga performa.
- **Korelasi:** Pearson, Spearman, Kendall; matriks pasangan dengan koefisien, N, p-value; Fisher-z confidence interval untuk Pearson jika memenuhi syarat; correlation heatmap, scatterplot pendamping.
- **Regresi:** satu atau banyak prediktor numerik (OLS); opsi robust standard errors HC3; koefisien, t, p, confidence intervals, R², adjusted R², RMSE, ANOVA OLS, VIF/tolerance, uji Shapiro–Wilk residual, Breusch–Pagan, Durbin–Watson, grafik residual dan Q–Q.
- **Uji beda:** paired t-test, Wilcoxon signed-rank, independent t-test (pooled), Welch independent t-test, Mann–Whitney U, one-way ANOVA, Welch ANOVA, Kruskal–Wallis. Post-hoc ketika uji keseluruhan signifikan: Tukey HSD, Games–Howell, Dunn–Holm.
- **Desain paired:** data `Long` (ID pasangan, kelompok, variabel Y) atau `Wide` (dua kolom Y). Data long harus satu nilai per ID–kelompok; aplikasi menolak duplikasi yang dapat membuat pasangan salah.
- **Output:** tabel bernomor/judul yang dapat diedit, laporan Excel multihalaman berformat, catatan metodologi, grafik PNG resolusi tinggi. Untuk analisis split, seluruh PNG dapat diunduh sekaligus sebagai ZIP.
- Grafik dan laporan dibentuk di server. **Tidak perlu Chrome/Kaleido untuk PNG.**

## Cara menjalankan pada komputer lokal

Python 3.10 atau lebih baru direkomendasikan. Buka terminal pada folder aplikasi, kemudian:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Buka alamat lokal yang ditampilkan oleh Streamlit, umumnya `http://localhost:8501`.

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru, misalnya `statcal-environmental-statistical-analysis`.
2. Unggah **isi folder proyek** (bukan file ZIP) ke root repository, termasuk `app.py`, `stat_core.py`, `plots.py`, `data_io.py`, `report.py`, `requirements.txt`, dan folder `data/`.
3. Masuk ke <https://share.streamlit.io/>, klik **Create app**, pilih repository dan branch.
4. Set **Main file path** ke `app.py`, lalu **Deploy**.
5. Setelah terbit, uji unggah dataset dan unduh Excel/PNG.

Jika muncul `ModuleNotFoundError`, pastikan `requirements.txt` ikut terunggah. Jika tampilan kosong, cek log di `Manage app → Logs`.

## Praktikum Kasus 1 — Sepeda Motor dan Karbon Monoksida

Berkas contoh: `data/Kasus_1_Sepeda_Motor_CO.csv` dan `data/Kasus_1_Sepeda_Motor_CO.xlsx`.

Dataset **simulasi** 48 baris (24 bulan × 2 kelompok, Jan 2024–Des 2025), 11 kolom. **Satu baris adalah ringkasan 4 hari sampel, bukan satu hari atau satu kendaraan.**

### 1. Korelasi

- Pilih **Contoh Kasus 1** → tab **Korelasi**.
- Pilih `Motor_rata2_15menit`, `CO_rata2_ppm`, dan bila perlu `Suhu_rata2_C`, `Angin_rata2_mps`.
- Metode: Pearson. Tanpa split: hubungan pada 48 ringkasan pengamatan. Dengan split `Tahun`: korelasi 2024 dan 2025 dihitung terpisah.
- Korelasi Pearson antara motor dan CO untuk seluruh data simulasi ≈ **0,8029** (p ≈ 6,68 × 10⁻¹²).

### 2. Regresi

- Variabel Y: `CO_rata2_ppm`.
- Regresi sederhana: pilih X `Motor_rata2_15menit`.
- Regresi berganda: pilih juga `Suhu_rata2_C` dan `Angin_rata2_mps`.
- Perhatikan perbedaan koefisien, R², uji asumsi, dan makna variabel penjelas. Untuk contoh tiga X, R² sekitar **0,7153** (data simulasi).

### 3. Uji beda berpasangan

- Jenis uji: `Paired Samples t-Test` atau `Wilcoxon Signed-Rank`.
- Variabel Y: `CO_rata2_ppm`.
- Bentuk data: **Long (ID + kategori)**.
- Kolom kelompok: `Jenis_hari`.
- ID pasangan: `Periode`.
- Pilih `Hari kerja` dan `Akhir pekan`.
- Tanpa split: N = **24 pasangan bulanan**. Dengan split `Tahun`: masing-masing N = 12 pasangan.
- Nilai rata-rata CO pada simulasi: hari kerja ≈ **2,9421 ppm** dan akhir pekan ≈ **2,1929 ppm**. Uji paired t pada simulasi: t ≈ **8,257**, p ≈ **2,48 × 10⁻⁸**.

### 4. ANOVA / Kruskal–Wallis

Pilih **Contoh Kasus 2 — Kualitas Air** (data simulasi 90 baris, 3 lokasi: Hulu, Tengah, Hilir).

- Variabel hasil: `BOD_mg_L`.
- Variabel kelompok: `Lokasi`.
- Uji: `One-Way ANOVA`, `Welch ANOVA`, atau `Kruskal–Wallis`.
- Perhatikan diagnostik, effect size, dan tabel post-hoc bila uji keseluruhan signifikan.

## Berkas proyek

```text
STATCAL_Environmental_Statistical_Analysis/
  app.py                         # Streamlit UI
  stat_core.py                   # Perhitungan dan validasi statistik
  data_io.py                     # Pembaca, deteksi header, filter
  plots.py                       # Ekspor PNG Matplotlib
  report.py                      # Ekspor laporan Excel
  requirements.txt               # Dependensi Streamlit Cloud
  README.md                      # Panduan ini
  data/
    Kasus_1_Sepeda_Motor_CO.csv
    Kasus_1_Sepeda_Motor_CO.xlsx
    Kasus_2_Kualitas_Air_Tiga_Lokasi.csv
  examples/
    Contoh_Korelasi.png
    Contoh_Regresi.png
    Contoh_Uji_Beda.png
  tests/
    test_engine.py
```

## Catatan metodologi penting

1. **Observasi independen adalah asumsi desain**, bukan fitur yang otomatis dapat dikenali program. Data berulang per bulan/lokasi dapat memiliki autokorelasi atau struktur hierarkis. Hasil uji klasik pada data demikian hanya ilustrasi jika asumsi independensi tidak dipenuhi.
2. **Uji paired membutuhkan pasangan yang benar**. Untuk data berbentuk long, pilih ID yang sama muncul satu kali per kelompok. ID berulang pada kelompok yang sama menyebabkan error, bukan digabungkan diam-diam.
3. **Uji nonparametrik tidak selalu menguji perbedaan median.** Mann–Whitney dan Kruskal–Wallis secara umum terkait perbedaan distribusi/peringkat; klaim median membutuhkan asumsi tambahan.
4. **Normalitas regresi** menyangkut residual untuk inferensi normal klasik; data X atau Y tidak harus normal. Shapiro–Wilk bukan satu-satunya cara mengevaluasi model.
5. **Multikolinearitas sempurna** ditolak ketika persamaan tidak dapat diestimasi. VIF membantu membaca multikolinearitas nonsempurna.
6. **Uji beda dua sampel independen** tidak otomatis sah untuk sampel berpasangan atau data agregat yang direkam pada bulan yang sama.
7. **Split** menyediakan hasil terpisah, tetapi tidak dengan sendirinya merupakan uji formal apakah koefisien antarkelompok berbeda. Jika pengguna menguji banyak subset, risiko multiple testing perlu dipertimbangkan.
8. **Post-hoc otomatis** tersedia saat omnibus p < α; tidak semua kasus memerlukan atau memungkinkan post-hoc yang sah, terutama pada sampel kecil atau varians nol.
9. Confidence interval Fisher-z Pearson bersifat pendekatan dan dapat kurang andal pada data yang tidak memenuhi asumsi.
10. Semua data contoh adalah **simulasi**, tidak boleh digunakan sebagai bukti empiris atau penilaian kualitas udara/air sebenarnya.

## Status verifikasi

Mesin perhitungan, split, metode uji, grafik PNG, serta pembuatan Excel diverifikasi melalui unit test yang dapat dijalankan dengan:

```bash
python -m pip install pytest
python -m pytest tests -q
```

> UI Streamlit belum dapat diuji secara live dalam lingkungan pembuatan ini karena paket Streamlit tidak terinstal di sana. Uji end-to-end setelah deploy direkomendasikan.
