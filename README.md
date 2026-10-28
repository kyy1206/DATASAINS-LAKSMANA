# LAKSAMANA — Prototipe Aplikasi Python

Aplikasi Streamlit untuk:
- membaca dataset pangan dan iklim yang sudah dibersihkan;
- mengeksplorasi korelasi indikator iklim dengan surplus/defisit pangan;
- menampilkan ringkasan hasil analisis dan kesimpulan yang dihitung langsung dari dataset;
- mengunduh ringkasan hasil analisis untuk bahan laporan;
- melatih K-Nearest Neighbors (KNN) untuk mengklasifikasikan risiko defisit/surplus dari label historis;
- melatih Regresi Linier untuk mengestimasi surplus dan kebutuhan bulanan;
- mengevaluasi regresi dengan MAE, RMSE, dan R² serta membandingkannya dengan baseline rata-rata data latih;
- menampilkan grafik tren komoditas/iklim, aktual-vs-prediksi, dan koefisien regresi;
- mengatur persentase data latih dengan pembagian kronologis agar periode uji tetap berada di masa depan;
- menerima input komoditas, kondisi iklim, lama gangguan, dan stok layak pakai;
- menghitung simulasi buffer minimum dari kebutuhan terprediksi, durasi gangguan, dan stok tersedia.

## Menjalankan di Windows

1. Ekstrak ZIP ke folder, misalnya `D:\LAKSAMANA_App`.
2. Buka folder tersebut di VS Code.
3. Pastikan Python 3.10 atau lebih baru terpasang.
4. Buka terminal VS Code dan jalankan:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

5. Browser akan membuka aplikasi Streamlit. Jika tidak terbuka otomatis, buka alamat lokal yang muncul di terminal (biasanya `http://localhost:8501`).

## File
- `app.py`: kode aplikasi.
- `requirements.txt`: dependensi Python.
- `Dataset_Pangan_Iklim_Cleaned_Normalized.xlsx`: dataset yang digunakan aplikasi.

## Catatan metodologi
Aplikasi ini prototipe penelitian, bukan sistem operasional resmi. KNN belajar label dari surplus bulanan aktual (surplus < 0 berarti defisit); label ini bukan catatan gangguan pelayaran. Regresi Linier belajar surplus dan kebutuhan dari data historis pangan yang dipasangkan dengan curah hujan dan angin. Evaluasi menahan periode terbaru agar model diuji pada waktu yang tidak digunakan saat pelatihan.

Dataset iklim hanya mencakup sekitar 24 bulan dan tidak menyediakan catatan gangguan maritim maupun stok cadangan aktual. Karena itu, hasil risiko adalah pola historis, bukan prediksi sebab-akibat gangguan logistik. Label defisit juga lebih jarang daripada label surplus, dan kondisi iklim yang sama digunakan oleh beberapa komoditas; skor evaluasi tinggi pada sedikit bulan uji tidak membuktikan performa umum. Buffer yang ditampilkan merupakan simulasi sederhana: `max(0, kebutuhan bulanan terprediksi / 30 × hari gangguan − stok layak pakai)`. Pastikan satuan input konsisten dengan dataset; angka tersebut bukan standar buffer resmi dan tidak boleh menjadi satu-satunya dasar keputusan.

MAE dan RMSE mengukur besar kesalahan dalam satuan target; semakin kecil semakin baik. R² membandingkan kecocokan terhadap variasi data uji dan bisa bernilai negatif. Pembanding baseline memprediksi nilai rata-rata target dari data latih. Grafik aktual-vs-prediksi dan koefisien membantu membaca perilaku model, tetapi tidak membuktikan hubungan sebab-akibat.

Menu **Ringkasan Hasil** menampilkan jumlah data, korelasi gabungan dan per komoditas, serta evaluasi model dengan pembagian kronologis 80/20. Halaman ini juga menyediakan ringkasan Markdown yang dapat diunduh. Angka dan kesimpulan dihitung dari dataset saat aplikasi dijalankan; bila dataset diganti, hasilnya ikut berubah.
