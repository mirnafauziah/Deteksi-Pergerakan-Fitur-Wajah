# Deteksi-Pergerakan-Fitur-Wajah

Pada perancancangan sistem ini, proses dibagi menjadi tiga tahap, yaitu pre-processing, training dan detecting.

![Tesis-Keseluruhan-Flowchart-v2](https://user-images.githubusercontent.com/60698877/107319440-70e99f00-6ad1-11eb-9fcc-d790d3d7dfbc.png)

## Pre-Processing
Dataset yang digunakan merupakan dataset yang dikumpulkan terhadap 64 wajah yang berbeda dengan kondisi wajah dan pencahayaan yang berbeda-beda. 
Dataset kemudian melalui proses pre-processing untuk diseragamkan agar dapat melalui proses pelatihan pada tahap training.
Gambar-gambar diubah menjadi berskala abu-abu dan berukuran 100x100.

![preprocesing-wajah](https://user-images.githubusercontent.com/60698877/107319696-ea818d00-6ad1-11eb-961e-fc2672703845.JPG)

## Training
Tahap training menggunakan arsitektur CNN untuk mengolah sekumpulan gambar (dataset) agar menghasilkan model deteksi pergerakan fitur wajah. 

![arsitektur CNN wajah-FACE](https://user-images.githubusercontent.com/60698877/107319664-d76ebd00-6ad1-11eb-9e89-3d73cb07a9bd.png)

## Detecting
Setelah model selesai dilatih, model digunakan untuk mendeteksi pergerakan fitur wajah yang terekam pada kamera webcam.

![V3_Tesis-Keseluruhan+LOG](https://user-images.githubusercontent.com/60698877/107320470-51537600-6ad3-11eb-8957-575d3874e2d4.png)

Video demo dapat dilihat pada halaman: https://www.youtube.com/watch?v=QFMU-_Iw0GU&lc
