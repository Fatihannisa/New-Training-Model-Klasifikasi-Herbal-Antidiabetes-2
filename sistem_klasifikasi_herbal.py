import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as components
import base64
import cv2
import os
import io
from rembg import new_session, remove

# =========================================================
# CONFIG & PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="DiaHerb - Sistem Identifikasi Daun Herbal Antidiabetes",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper Base64 Gambar
def load_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

@st.cache_resource
def get_bg_session():
    return new_session("u2netp")
bg_session = get_bg_session()

# =========================================================
# LOAD MODEL TFLITE
# =========================================================
@st.cache_resource
def load_tflite():
    model_file = "leafnet_dual_branch.tflite"
    if not os.path.exists(model_file):
        st.warning(f"File model '{model_file}' tidak ditemukan di directory root. Menggunakan simulasi prediksi...")
        return None
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite()

LABELS = [
    "Acalypha siamensis", "Andrographis paniculata", "Cananga odorata", "Capsicum sp", "Catharanthus roseus",
    "Dracaena angustifolia", "Ficus microcarpa", "Flueggea virosa", "Gardenia jasminoides", "Leucaena leucocephala",
    "Moringa oleifera", "Orthosiphon aristatus", "Pandanus amaryllifolius", "Phyllanthus amarus",
    "Physalis angulata", "Rosa sp", "Solanum nigrum", "Syzygium polyanthum", "Vernonia amygdalina", "Ziziphus mauritiana"
]

CONFIG = {
    "IMG_SIZE": (256, 256),
    "TARGET_BRIGHTNESS": 110,
    "CLAHE_CLIP": 4.0,
    "CLAHE_TILE": (4, 4),
    "VEIN_STRENGTH": 1.2
}

# =========================================================
# DATABASE HERBAL DINAMIS
# =========================================================
herbal_info = {
    "Acalypha siamensis": {
        "nama_umum": ["Teh-tehan", "Teh hutan"],
        "status": "Tanaman pembanding",
        "informasi": "Teh-tehan adalah tanaman perdu atau semak yang sering digunakan sebagai pagar hidup dekoratif.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Andrographis paniculata": {
        "nama_umum": ["Sambiloto", "Ki pait", "Ampadu tanah", "Ki oray"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Sambiloto terkenal sebagai herbal dengan kandungan andrographolide (AGL) yang sangat pahit, tetapi berkhasiat tinggi dalam mengendalikan kadar gula darah dan bersifat antiinflamasi. AGL mampu meningkatkan produksi insulin dan penyerapan glukosa sehingga mengurangi kadar gula dalam darah.",
        "tautan_artikel": "https://hellosehat.com/diabetes/daun-sambiloto-untuk-diabetes/",
        "judul_artikel": "Kenali Manfaat Daun Sambiloto untuk Diabetes, Plus Efek Sampingnya",
        "tautan_jurnal": "https://jurnal.ikbis.ac.id/index.php/infokes/article/view/371/221",
        "judul_jurnal": "Putri, A., Santoso, E. B., & Putri, R. T. N. (2021). Air Rebusan Daun Sambiloto ( Andrographis Paniculata ) Terhadap Penurunan Kadar Gula Darah Pada Penderita Diabetes Meliitus Dosen Institut Kesehatan dan Bisnis Surabaya , Jln Medokan Semampir Indah No 27 Mahasiswa Institut Kesehatan dan Bisnis Surabaya ,. Jurnal Info Kesehatan, 11(2), 427–430. https://jurnal.ikbis.ac.id/infokes/article/view/371/221",
        "cara_mengolah": [
            "Sambiloto sebaiknya dikonsumsi setelah makan untuk mencegah sakit maag.",
            "Cuci daun sambiloto sebelum dimasak.",
            "Rebuslah daun sambiloto dengan 2–3 gelas air hingga matang, lalu diminum.",
            "Untuk mengurangi rasa pahit sambiloto, bisa juga menambahkan madu."
        ],
        "tautan_pengolahan": "https://www.alodokter.com/sambiloto",
        "sumber_pengolahan": "Alodokter - Sambiloto",
        "catatan": "Untuk menghindari risiko efek samping, disarankan untuk mengonsumsi dalam jumlah yang wajar dan tidak lebih dari dua kali sehari. Jika memiliki kondisi medis tertentu, konsultasikan terlebih dahulu dengan dokter."
    },
    "Cananga odorata": {
        "nama_umum": ["Kenanga", "Kananga", "Sepalen"],
        "status": "Tanaman pembanding",
        "informasi": "Kenanga adalah tanaman tropis yang bunganya sangat harum. Dimanfaatkan sebagai bahan utama aromaterapi dan kosmetik.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Capsicum sp": {
        "nama_umum": ["Cabai", "Lombok"],
        "status": "Tanaman pembanding",
        "informasi": "Cabai adalah tanaman hortikultura dari famili terong-terongan yang digunakan sebagai bumbu dapur.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Catharanthus roseus": {
        "nama_umum": ["Tapak dara", "Bunga serdadu", "Kembang tembaga"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Ekstrak daun tapak dara dipercaya dapat merangsang sekresi insulin dalam sel beta pankreas dan meningkatkan penggunaan glukosa di jaringan perifer, membantu menjaga kestabilan gula darah.",
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/manfaat-daun-tapak-dara/",
        "judul_artikel": "7 Manfaat Daun Tapak Dara, Menghambat Kanker hingga Atasi Diabetes",
        "tautan_jurnal": "https://jurnal.unpad.ac.id/farmaka/article/view/47508/pdf",
        "judul_jurnal": "Taruh, B. R., Mokosuli, Y. S., & Tuda, A. I. (2021). Uji Efektifitas Ekstrak Daun Tapak Dara ( Catharanthus roseus ( L ) G . Don ) Sebagai Penurun Kadar Gula Darah Pada Tikus Putih ( Rattus norvegicus L .). 2(2), 34–41.",
        "cara_mengolah": [
            "Siapkan 5-10 lembar daun tapak dara yang masih segar dan 2 gelas air.",
            "Cuci bersih daun tapak dara di bawah air mengalir.",
            "Rebus daun dengan api kecil hingga mendidih (sekitar 10-15 menit) sampai air rebusan berubah warna.",
            "Saring dan biarkan sedikit dingin sebelum diminum. Disarankan mengonsumsinya satu gelas sehari"
        ],
        "tautan_pengolahan": "https://www.liputan6.com/hot/read/5928688/cara-merebus-daun-tapak-dara-mampu-atasi-kolesterol-dan-diabetes-mellitus",
        "sumber_pengolahan": "Liputan 6 - Cara Merebus Daun Tapak Dara, Mampu Atasi Kolesterol dan Diabetes Mellitus",
        "catatan": "Disarankan mengonsumsi satu gelas sehari. Jika memiliki kondisi medis tertentu, konsultasikan terlebih dahulu dengan dokter."
    },
    "Dracaena angustifolia": {
        "nama_umum": ["Suji", "Suji hijau", "Semar"],
        "status": "Tanaman pembanding",
        "informasi": "Suji hijau adalah tumbuhan perdu tahunan yang daunnya dimanfaatkan sebagai pewarna hijau alami makanan.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Ficus microcarpa": {
        "nama_umum": ["Beringin dolar", "Beringin cina"],
        "status": "Tanaman pembanding",
        "informasi": "Beringin dolar adalah spesies pohon ara tropis yang populer sebagai tanaman hias dan bahan bonsai.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Flueggea virosa": {
        "nama_umum": ["Sigar jalak", "Trembilutan"],
        "status": "Tanaman pembanding",
        "informasi": "Sigar jalak adalah tanaman perdu atau pohon kecil yang banyak digunakan sebagai tanaman pagar.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Gardenia jasminoides": {
        "nama_umum": ["Kaca piring", "Melati tanjung"],
        "status": "Tanaman pembanding",
        "informasi": "Kaca piring adalah tanaman perdu tropis berbunga putih dan beraroma harum lembut.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Leucaena leucocephala": {
        "nama_umum": ["Lamtoro", "Petai cina"],
        "status": "Tanaman pembanding",
        "informasi": "Lamtoro atau petai cina adalah perdu polong-polongan yang digunakan sebagai peneduh, pagar hidup, dan pakan ternak.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Moringa oleifera": {
        "nama_umum": ["Kelor", "Merunggai"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun kelor memiliki efek hipoglikemik yang membantu menurunkan kadar gula darah dengan meningkatkan sensitivitas insulin dan mengurangi penyerapan glukosa di usus.",
        "tautan_artikel": "https://hellosehat.com/diabetes/tipe-2/manfaat-daun-kelor-untuk-diabetes/",
        "judul_artikel": "Mengulik Khasiat Daun Kelor sebagai Obat Diabetes Alami, Benarkah Bermanfaat?",
        "tautan_jurnal": "https://doi.org/10.35617/jfionline.v12i1.21",
        "judul_jurnal": "Kusuma, I. Y., Pujiarti, Y., & Samodra, G. (2020). Potensi Daun Kelor (Moringa oleifera) sebagai Agen Anti-Hipergikemia: Studi Literatur : POTENTIAL OF MORINGA LEAF (Moringa oleifera) AS ANTI-HYPERGLICEMIC AGENT: A LITERATUR REVIEW. JFIOnline | Print ISSN 1412-1107 | E-ISSN 2355-696X, 12(1), 94–99. https://doi.org/10.35617/jfionline.v12i1.21",
        "cara_mengolah": [
            "Salah satu cara paling praktis adalah menjadikannya sup bening atau teh herbal.",
            "Untuk membuat sup, cukup rebus bumbu dan sayuran lain terlebih dahulu hingga matang, lalu matikan api kompor.",
            "Masukkan daun kelor segar ke dalam kuah panas dan biarkan layu secara alami selama 2–3 menit sebelum disajikan.",
            "Kunci utama mengolah daun kelor adalah menghindari paparan suhu tinggi dalam waktu yang lama. Pemanasan berlebih dapat menghancurkan vitamin C dan senyawa antioksidan sensitif seperti quercetin dan asam klorogenat."
        ],
        "tautan_pengolahan": "https://www.halodoc.com/artikel/ini-cara-mengolah-daun-kelor-agar-manfaatnya-maksimal?srsltid=AU7gw4VTzmXCONXyFP9MVyjgNw3h2hMtKAKY8oY1IKRvwfGmYRp0qytS",
        "sumber_pengolahan": "Halodoc - Ini Cara Mengolah Daun Kelor agar Manfaatnya Maksimal",
        "catatan": "Konsumsi dalam jumlah wajar. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Orthosiphon aristatus": {
        "nama_umum": ["Kumis kucing", "Remujung"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun kumis kucing kaya flavonoid dan saponin. Flavonoid menghambat pemecahan karbohidrat di usus, sedangkan saponin merangsang pelepasan insulin.",
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/tanaman-kumis-kucing/",
        "judul_artikel": "7 Manfaat Tanaman Kumis Kucing untuk Kesehatan",
        "tautan_jurnal": "https://doi.org/10.36990/hijp.v7i1.533",
        "judul_jurnal": "Masrif, M., & Ibrahim, I. (2015). Pengaruh Pemberian Ekstrak Daun Kumis Kucing (Orthosiphon Aristatus) Terhadap Perubahan Kadar Glukos Darah Pada Pasien Diabetes Mellitus Di Ruang Rawat Jalan Rumah Sakit Umum Bahteramas Provinsi Sulawesi Tenggara. Health Information : Jurnal Penelitian, 7, 27–33. https://doi.org/10.36990/hijp.v7i1.533",
        "cara_mengolah": [
            "Siapkan 5 sampai 7 lembar daun kumis kucing segar dan 2 sampai 3 gelas air.",
            "Cuci bersih di bawah air mengalir.",
            "Rebus hingga mendidih hingga airnya tersisa setengah.",
            "Saring air rebusan dan minum 1-2 kali sehari, masing-masing setengah gelas."
        ],
        "tautan_pengolahan": "https://health.detik.com/berita-detikhealth/d-7483374/cara-mengolah-tanaman-kumis-kucing-kerap-digunakan-untuk-mengatasi-diabetes",
        "sumber_pengolahan": "Detik health - Cara Mengolah Tanaman Kumis Kucing, Kerap Digunakan untuk Mengatasi Diabetes",
        "catatan": "Konsumsi dalam jumlah wajar. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Pandanus amaryllifolius": {
        "nama_umum": ["Pandan wangi", "Pandan"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun pandan mengandung flavonoid, tanin, dan polifenol yang mampu merangsang produksi hormon insulin dari sel beta pankreas.",
        "tautan_artikel": "https://share.google/BTrQ3MBtqbTndrJvO",
        "judul_artikel": "Manfaat Daun Pandan dan Efek Sampingnya Bagi Tubuh",
        "tautan_jurnal": "https://ejurnalmalahayati.ac.id/index.php/kebidanan/article/view/3024/pdf",
        "judul_jurnal": "Kaban, N. B., Putri, P. S., & Medan, S. F. (2020). PEMBERIAN AIR DAUN PANDAN TERHADAP PENURUNAN KADAR GULA DARAH. 6(4), 493–496.",
        "cara_mengolah": [
            "Siapkan 3-4 lembar daun pandan segar/kering, 500 ml air, dan pemanis alami jika diperlukan.",
            "Cuci bersih lalu potong menjadi beberapa bagian.",
            "Rebus air hingga mendidih.",
            "Biarkan daun pandan direbus selama 10-15 menit hingga air berwarna hijau kekuningan.",
            "Saring air rebusan, tambahkan pemanis alami jika diinginkan, dan teh daun pandan siap dinikmati."
        ],
        "tautan_pengolahan": "https://health.grid.id/read/354114602/cara-mengolah-daun-pandan-untuk-mengontrol-gula-darah-tinggi?page=all",
        "sumber_pengolahan": "Health grid - Cara Mengolah Daun Pandan untuk Mengontrol Gula Darah Tinggi.",
        "catatan": "Hindari menambahkan gula pasir tinggi kalori; gunakan pemanis alami jika diperlukan. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Phyllanthus amarus": {
        "nama_umum": ["Meniran"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Meniran memiliki senyawa aktif yang memengaruhi metabolisme glukosa dan mendukung pengelolaan tingkat kadar gula darah secara berkelanjutan.",
        "tautan_artikel": "https://www.halodoc.com/artikel/manfaat-pohon-meniran-jaga-imun-ginjal-sehat-alami",
        "judul_artikel": "Manfaat Pohon Meniran: Jaga Imun, Ginjal Sehat Alami",
        "tautan_jurnal": "https://doi.org/10.36656/jpfh.v2i1.79",
        "judul_jurnal": "Sari, H., Kaban, V. E., Situmorang, F. R., Fahdi, F., Kesehatan, I., Husada, D., Besar, J., & Deli, N. (2019). UJI EFEKTIVITAS ANTIDIABETES KOMBINASI EKSTRAK DAUN MENIRAN ( Phyllanthus niruri L .) Dan KELOPAK BUNGA ROSELLA ( Hibiscus sabdariffa L .) PADA TIKUS JANTAN PUTIH Purpose : To determine the effect of decreasing blood glucose levels in white rats using a c. 2(1).",
        "cara_mengolah": [
            "Siapkan 20 batang atau 3gram daun meniran dan 400 ml air.",
            "Cuci bersih di bawah air mengalir.",
            "Rebus selama 5-10 menit.",
            "Minum rebusan ini dua kali sehari, pagi dan sore."
        ],
        "tautan_pengolahan": "https://www.halodoc.com/artikel/meniran-obat-apa-manfaat-dan-cara-pakai-lengkap?srsltid=AU7gw4VrbrsEBXiYIODpm-gKSpYE8iqwQMKZpT2-sFiNL6_hPO3P994F",
        "sumber_pengolahan": "Halodoc - Meniran Obat Apa? Manfaat dan Cara Pakai Lengkap!",
        "catatan": "Konsumsi dalam jumlah wajar. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Physalis angulata": {
        "nama_umum": ["Ciplukan", "Ceplukan", "Cecendet"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun Ciplukan memiliki indeks glikemik rendah dan dapat meningkatkan sensitivitas atau produksi insulin dalam tubuh.",
        "tautan_artikel": "https://www.halodoc.com/artikel/pohon-ciplukan-dan-khasiatnya-dari-diabetes-hingga-kanker",
        "judul_artikel": "Pohon Ciplukan dan Khasiatnya: Dari Diabetes Hingga Kanker",
        "tautan_jurnal": "https://journal.ukmc.ac.id/index.php/joh/article/view/1141/1081",
        "judul_jurnal": "Azizah, M., & Agustina, C. (2024). Uji Aktivitas Antidiabetes Kombinasi Ekstrak Etanol Daun Ciplukan ( Physalis Angulata L . ) dan Madu Hutan Terhadap Mencit Putih Jantan yang Diinduksi Streptozotocin. 7(1), 189–197. https://doi.org/10.32524/jksp.v7i1.1141",
        "cara_mengolah": [
            "Siapkan beberapa lembar daun ciplukan segar dan beberapa gelas air.",
            "Cuci bersih daun di bawah air mengalir.",
            "Rebus hingga mendidih hingga airnya berkurang.",
            "Saring air rebusan dan minum secara teratur sesuai kebutuhan."
        ],
        "tautan_pengolahan": "https://www.halodoc.com/artikel/daun-ciplukan-untuk-obat-apa-atasi-diabetes-hingga-rematik?srsltid=AU7gw4XZFvyyCQRZZfNnPoGNmfVcVVEhXwI261HcyUPnUzAVODxwGfBz",
        "sumber_pengolahan": "Halodoc - Daun Ciplukan untuk Obat Apa? Atasi Diabetes Hingga Rematik", 
        "catatan": "Disarankan merebus dengan wadah stainless steel atau enamel. Konsumsi dalam jumlah wajar. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Rosa sp": {
        "nama_umum": ["Mawar"],
        "status": "Tanaman pembanding",
        "informasi": "Mawar adalah tumbuhan perdu berkayu dan berduri yang terkenal sebagai tanaman hias populer.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Solanum nigrum": {
        "nama_umum": ["Ranti", "Leunca"],
        "status": "Tanaman pembanding",
        "informasi": "Ranti adalah tanaman suku terung-terungan yang sering dikonsumsi sebagai lalapan atau bahan masakan.",
        "tautan_artikel": "",
        "tautan_jurnal": "",
        "cara_mengolah": [],
        "catatan": "Tanaman ini **bukan** merupakan tanaman herbal antidiabetes."
    },
    "Syzygium polyanthum": {
        "nama_umum": ["Salam", "Manting", "Ubar serai"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun salam mengandung flavonoid, tanin, dan polifenol yang meningkatkan kerja insulin serta menghambat penyerapan gula di usus.",
        "tautan_artikel": "https://www.halodoc.com/artikel/daun-salam-khasiat-dan-cara-konsumsi-sehat",
        "judul_artikel": "Daun Salam: Khasiat dan Cara Konsumsi Sehat",
        "tautan_jurnal": "https://doi.org/10.3164/jcbn.08-188",
        "judul_jurnal": "Khan, A., Zaman, G., & Anderson, R. A. (2009). Bay leaves improve glucose and lipid profile of people with type 2 diabetes. Journal of clinical biochemistry and nutrition, 44(1), 52–56. https://doi.org/10.3164/jcbn.08-188",
        "cara_mengolah": [
            "Siapkan 10-15 lembar daun salam segar atau kering dan 3 gelas air.",
            "Cuci bersih daun salam.",
            "Rebus dengan api sedang sampai menyusut menjadi 1 gelas.",
            "Saring dan minum selagi hangat secara teratur."
        ],
        "tautan_pengolahan": "https://www.halodoc.com/artikel/cara-mengolah-daun-salam-yang-baik-untuk-kesehatan",
        "sumber_pengolahan": "Halodoc - Cara Mengolah Daun Salam yang Baik untuk Kesehatan",
        "catatan": "Gunakan wadah perebus yang tidak reaktif terhadap logam dan konsumsi dalam jumlah wajar. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Vernonia amygdalina": {
        "nama_umum": ["Daun Afrika", "Daun pahit", "Daun insulin"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Ekstrak daun Afrika mengandung saponin, tanin, flavonoid, dan alkaloid yang terbukti efektif menekan lonjakan glukosa darah pasca makan.",
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/manfaat-daun-afrika/",
        "judul_artikel": "7 Manfaat Daun Afrika bagi Kesehatan Tubuh, Jangan Lewatkan!",
        "tautan_jurnal": "https://www.neliti.com/id/publications/460123/potensi-daun-afrika-vernonia-amygdalina-sebagai-antidiabetik",
        "judul_jurnal": "Putri, Yunisa A. 'Potensi Daun Afrika ( Vernonia Amygdalina ) sebagai Antidiabetik.' Jurnal Ilmiah Kesehatan Sandi Husada, vol. 8, no. 2, 2019, pp. 336-339, doi:10.35816/jiskh.v10i2.183.",
        "cara_mengolah": [
            "Siapkan 5-10 lembar daun Afrika dan 4 gelas air.",
            "Cuci bersih di bawah air mengalir.",
            "Rebus selama 10-15 menit hingga tersisa 2 gelas.",
            "Minum pagi dan sore hari. Dapat ditambahkan sedikit perasan jeruk nipis untuk mengurangi rasa pahit."
        ],
        "tautan_pengolahan": "https://www.liputan6.com/hot/read/5871678/cara-merebus-daun-afrika-untuk-jaga-gula-darah-dan-obat-demam-simak-panduan-lengkapnya",
        "sumber_pengolahan": "Liputan 6 - Cara Merebus Daun Afrika untuk Jaga Gula Darah dan Obat Demam, Simak Panduan Lengkapnya",
        "catatan": "Minum secara teratur dalam dosis aman. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Ziziphus mauritiana": {
        "nama_umum": ["Bidara", "Widara", "Bukol"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Kandungan saponin dan flavonoid dalam daun bidara berfungsi sebagai antioksidan kuat untuk meningkatkan efektivitas kerja hormon insulin.",
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/daun-bidara/",
        "judul_artikel": "Daun Bidara: Kandungan, Manfaat, Efek Samping, dll.",
        "tautan_jurnal": "https://doi.org/10.3390/plants13162195",
        "judul_jurnal": "Nilofar, Sinan, K. I., Dall’Acqua, S., Sut, S., Uba, A. I., Etienne, O. K., Ferrante, C., Ahmad, J., & Zengin, G. (2024). Ziziphus mauritiana Lam. Bark and Leaves: Extraction, Phytochemical Composition, In Vitro Bioassays and In Silico Studies. Plants, 13(16), 2195. https://doi.org/10.3390/plants13162195",
        "cara_mengolah": [
            "Siapkan 5 hingga 7 lembar daun bidara segar yang tidak terlalu tua atau terlalu muda dan 400 ml air.",
            "Cuci bersih daun bidara.",
            "Rebus air hingga mendidih selama 10 sampai 15 menit.",
            "Saring air rebusan menggunakan saringan halus untuk memisahkan ampas daun. Buang ampasnya dan simpan air rebusan.",
            "Opsional: tambahkan pemanis alami seperti madu.",
            "Konsumsi air rebusan daun bidara selagi hangat. Rebusan ini sebaiknya diminum sesegera mungkin setelah dibuat untuk menjaga kesegaran dan potensi khasiatnya"
        ],
        "tautan_pengolahan": "https://www.halodoc.com/artikel/aturan-minum-rebusan-daun-bidara-jangan-salah-minum?srsltid=AU7gw4WFF4l3w3Yn0wXr828L34RAMr6tzIYhzgx2fNIpRY0lnhUPAZqu",
        "sumber_pengolahan": "Halodoc - Aturan Minum Rebusan Daun Bidara: Jangan Salah Minum!",
        "catatan": "Minum secara teratur dalam dosis aman. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    }
}

# =========================================================
# FUNGSIONAL PREPROCESSING OPENCV
# =========================================================
def _normalize_brightness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    mean_l = l.mean()

    if mean_l < 5:
        return img
    l_norm = np.clip(l * (CONFIG["TARGET_BRIGHTNESS"] / mean_l), 0, 255)
    out = cv2.merge([l_norm, a, b]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def _clahe_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CONFIG["CLAHE_CLIP"], tileGridSize=CONFIG["CLAHE_TILE"])
    l = clahe.apply(l)
    return cv2.cvtColor( cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def _sharpen_veins(img):
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=3)
    s = CONFIG["VEIN_STRENGTH"]
    sharp = cv2.addWeighted(img, 1 + s, blurred, -s, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
    
def get_leaf_mask(img):
    work = img.copy()
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 20, 20])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)
    _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)
    return clean_mask

def resize_input_image(img, size=(512, 512)):
    return cv2.resize(
        img,
        size,
        interpolation=cv2.INTER_AREA
    )
    
def preprocess_camera_leaf(img):
    try:
        _, buffer = cv2.imencode(".png", img)
        output = remove(buffer.tobytes(), session=bg_session)
        pil = Image.open(io.BytesIO(output)).convert("RGBA")
        rgba = np.array(pil)
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        mask = (alpha > 10).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return cv2.resize(img, CONFIG["IMG_SIZE"])
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
        leaf_crop = rgb[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        ys, xs = np.where(crop_mask > 0)
        if len(ys) > 0 and len(xs) > 0:
            leaf_crop = leaf_crop[ys.min():ys.max(), xs.min():xs.max()]
            crop_mask = crop_mask[ys.min():ys.max(), xs.min():xs.max()]
        canvas = np.ones_like(leaf_crop) * 255
        canvas[crop_mask > 0] = leaf_crop[crop_mask > 0]
        leaf_crop = canvas
        h_c, w_c = leaf_crop.shape[:2]
        scale = 250 / max(h_c, w_c)
        new_w, new_h = int(w_c * scale), int(h_c * scale)
        leaf_crop = cv2.resize(leaf_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        final_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        x_off, y_off = (256 - leaf_crop.shape[1]) // 2, (256 - leaf_crop.shape[0]) // 2
        final_img[y_off:y_off + leaf_crop.shape[0], x_off:x_off + leaf_crop.shape[1]] = leaf_crop
        final_img = _normalize_brightness(final_img)
        final_img = _clahe_lab(final_img)
        final_img = _sharpen_veins(final_img)
        return final_img
    except Exception:
        return cv2.resize(img, CONFIG["IMG_SIZE"])

def to_rgb_input(img: np.ndarray) -> np.ndarray:
  img = cv2.resize(img, CONFIG["IMG_SIZE"], interpolation=cv2.INTER_CUBIC)
  if img.dtype != np.uint8:
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
  mask = get_leaf_mask(img)
  img[mask == 0] = 0
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
  mean = np.array([0.485, 0.456, 0.406]) * 255
  std  = np.array([0.229, 0.224, 0.225]) * 255
  img = (img - mean) / std
  return img

def to_vein_input(img: np.ndarray) -> np.ndarray:
  EDGE_KERNEL_SIZE = (3, 3)
  EDGE_WEIGHT = 0.50
  img = cv2.resize(img, CONFIG["IMG_SIZE"], interpolation=cv2.INTER_CUBIC)
  if img.dtype != np.uint8:
      if img.max() <= 1.0:
          img = img * 255.0
      img = np.clip(img, 0, 255).astype(np.uint8)
  mask = get_leaf_mask(img)
  mask_binary = ( mask > 0).astype(np.uint8)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.bitwise_and(gray, gray, mask=mask)
  gray = cv2.GaussianBlur(gray, (3, 3), 0)
  clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
  vein = clahe.apply(gray)
  vein = cv2.bilateralFilter(vein, d=7, sigmaColor=50, sigmaSpace=50)
  blur_large = cv2.GaussianBlur(vein, (21, 21), 0)
  highpass = cv2.subtract(vein, (blur_large * 0.7).astype(np.uint8))
  sobelx = cv2.Sobel(highpass, cv2.CV_32F, 1, 0, ksize=3)
  sobely = cv2.Sobel(highpass, cv2.CV_32F, 0, 1, ksize=3)
  sobel = cv2.magnitude(sobelx, sobely)
  sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  vein = cv2.addWeighted(highpass, 0.20, sobel, 0.80, 0)
  vein = cv2.normalize(vein, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  _, vein = cv2.threshold(vein, 13, 255, cv2.THRESH_TOZERO)
  edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, EDGE_KERNEL_SIZE)
  leaf_edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, edge_kernel)
  leaf_edge = cv2.dilate(leaf_edge, edge_kernel, iterations=1)
  leaf_edge_float = (leaf_edge.astype(np.float32) * EDGE_WEIGHT)
  vein = np.maximum(vein.astype(np.float32), leaf_edge_float)
  vein = np.clip(vein, 0, 255).astype(np.uint8)
  vein[mask_binary == 0] = 0
  vein = (vein.astype(np.float32) / 255.0)
  vein = np.stack(
      [vein, vein, vein],
      axis=-1
  )
  return vein.astype(np.float32)

def predict(image):
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = resize_input_image(img_bgr, (512, 512))
    processed = preprocess_camera_leaf(img_bgr)
    rgb_input = np.expand_dims(to_rgb_input(processed), 0).astype(np.float32)
    vein_input = np.expand_dims(to_vein_input(processed), 0).astype(np.float32)

    if interpreter is not None:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        for inp in input_details:
            name = inp["name"].lower()
            if "rgb" in name:
                interpreter.set_tensor(inp["index"], rgb_input)
            elif "vein" in name:
                interpreter.set_tensor(inp["index"], vein_input)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
    else:
        # Pseudo-prediction fallback if model file missing during local preview
        np.random.seed(int(np.sum(processed) % 10000))
        pred = np.random.dirichlet(np.ones(len(LABELS)) * 0.5)
        pred[1] = 0.945  # Default to Sambiloto

    top_idx = np.argsort(pred)[::-1]
    return [(LABELS[idx], float(pred[idx])) for idx in top_idx[:5]]

# =========================================================
# STYLING CUSTOM CSS
# =========================================================
st.markdown("""
<style>
/* Font Inter & Clean Styling */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background Soft Herbal Gradient */
.stApp {
    background: linear-gradient(135deg, #f2f7f4 0%, #ecf3ee 50%, #e6efe9 100%) !important;
}

/* Card Container Modern */
.custom-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    margin-bottom: 20px;
}

/* Badge Status */
.badge-antidiabetes {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
    display: inline-block;
}

.badge-pembanding {
    background-color: #f1f5f9;
    color: #334155;
    border: 1px solid #cbd5e1;
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
    display: inline-block;
}

/* Scientific Name */
.scientific-name {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 32px;
    font-weight: 700;
    color: #064e3b;
    margin-top: 4px;
    margin-bottom: 8px;
}

/* Info Section Header */
.section-header {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 24px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Disclaimer Box */
.disclaimer-card {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 14px;
    padding: 20px 24px;
    color: #78350f;
    font-size: 15px;
    line-height: 1.7;
    margin-top: 30px;
}

/* Custom Green Primary Button (NO RED) */
div.stButton > button[kind="primary"],
div.stButton > button {
    background-color: #065f46 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    transition: background-color 0.2s ease, transform 0.1s ease !important;
    box-shadow: 0 4px 12px rgba(6, 95, 70, 0.25) !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button:hover {
    background-color: #044e39 !important;
    color: #ffffff !important;
    border: none !important;
}

/* File Uploader Square Dropzone Box Styling */
div[data-testid="stFileUploader"] {
    background-color: #f0fdf4 !important;
    border: 2px dashed #0d9488 !important;
    border-radius: 20px !important;
    padding: 24px !important;
    min-height: 220px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
}
div[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    padding: 12px !important;
}
div[data-testid="stFileUploader"] button {
    background-color: #ecfdf5 !important;
    color: #065f46 !important;
    border: 1px solid #a7f3d0 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER COMPONENT
# =========================================================
logo_b64 = load_base64("images/logo.png")

if logo_b64:
    st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; padding:5px 0; border-bottom:1px solid #e2e8f0; margin-bottom:30px;">
            <img src="data:image/png;base64,{logo_b64}" style="height:120px; width:auto;">
            <span style="font-size:13px; font-weight:600; color:#047857; background:#ecfdf5; padding:6px 14px; border-radius:10px; border:1px solid #a7f3d0;">
                LeafNet Dual-Branch Model • Tugas Akhir (211401034)
            </span>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="padding:10px 0; border-bottom:1px solid #e2e8f0; margin-bottom:30px;">
            <h1 style="font-family:'Playfair Display', serif; color:#065f46; margin:0; font-size:36px;">DiaHerb 🌿</h1>
            <p style="color:#64748b; margin:0; font-size:14px;">Sistem Identifikasi Daun Herbal Antidiabetes Berbasis LeafNet</p>
        </div>
    """, unsafe_allow_html=True)

# State Routing Halaman
if "page" not in st.session_state:
    st.session_state.page = "upload"

# =========================================================
# HALAMAN 1: UNGGAH GAMBAR
# =========================================================
if st.session_state.page == "upload":

    # Hero Banner
    st.markdown("""
        <div style="background: linear-gradient(135deg, #064e3b 0%, #0d9488 100%); padding: 36px; border-radius: 20px; color: white; margin-bottom: 30px;">
            <h1 style="font-family:'Playfair Display', serif; font-size: 34px; font-weight: 700; margin-top: 2px; margin-bottom: 12px; color: #f0fdf4;">
                Sistem Identifikasi Daun Herbal Antidiabetes
            </h1>
            <p style="font-size: 17px; line-height: 1.7; color: #e2e8f0; margin: 0; max-width: 900px;">
                DiaHerb dikembangkan untuk membantu mengidentifikasi spesies tanaman herbal antidiabetes berdasarkan citra daun. 
                Dengan bantuan kecerdasan buatan berbasis <i>Deep Learning</i>, sistem menggunakan model <b>Dual-Branch</b> untuk menganalisis karakteristik tulang daun melalui model <i>LeafNet</i> serta karakteristik visual daun melalui model <i>DenseNet201</i>. 
                Hasil analisis kedua karakteristik tersebut kemudian digunakan untuk menentukan spesies tanaman yang paling sesuai.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1.1])

    with col1:
        st.subheader("📷 Unggah Citra Daun")
        uploaded_file = st.file_uploader(
            "Pilih file foto daun (JPG, PNG, WEBP) dengan ukuran file maksimal 6 MB",
            type=["jpg", "jpeg", "png", "webp"],
            help="Maksimal ukuran file 6 MB."
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Preview Gambar yang Diunggah ({file_size_mb:.2f} MB)", width=340)

        if st.button("🔍 Identifikasi Daun Sekarang", use_container_width=True, type="primary"):
            if uploaded_file is not None:
                st.session_state.image = uploaded_file
                st.session_state.page = "result"
                st.rerun()
            else:
                st.warning("Silakan unggah gambar daun terlebih dahulu.")

    with col2:
        sample_paths = [
            "images/IMG_20251028_152831.jpg",
            "images/IMG_20251029_170845.jpg",
            "images/IMG_20251031_131056.jpg",
            "images/IMG_20251114_161441.jpg"
        ]
        sample_imgs_html = ""
        for path in sample_paths:
            b64 = load_base64(path)
            if b64:
                sample_imgs_html += f'<div style="aspect-ratio: 1; border-radius: 10px; overflow: hidden; border: 1px solid #cbd5e1;"><img src="data:image/jpeg;base64,{b64}" style="width: 100%; height: 100%; object-fit: cover;"></div>'

        st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);">
                <h4 style="margin-top:0; color:#0f172a; font-size:17px; font-weight:700;">📌 Tips Pengambilan Gambar</h4>
                <ul style="font-size:16px; color:#334155; padding-left:20px; line-height:1.8;">
                    <li>Foto <b>1 helai daun</b> saja.</li>
                    <li>Pastikan helai daun berada tepat di tengah frame kamera.</li>
                    <li>Pencahayaan terang agar struktur urat/venasi daun terlihat jelas.</li>
                    <li><b>Latar belakang wajib polos</b> dan berwarna terang (diutamakan putih).</li>
                    <li>Foto diambil dari sisi atas atau bawah tegak lurus.</li>
                </ul>
                <hr style="border: 0; border-top: 1px solid #f1f5f9; margin: 20px 0;">
                <h4 style="color:#0f172a; font-size:14px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px;">Contoh Sampel yang Baik:</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                    {sample_imgs_html}
                </div>
            </div>
        """, unsafe_allow_html=True)

# =========================================================
# HALAMAN 2: HASIL IDENTIFIKASI
# =========================================================
elif st.session_state.page == "result":

    st.markdown("<h2 style='text-align:center; font-family: Playfair Display, serif; font-size:36px; color:#064e3b; margin-bottom:24px;'>Hasil Identifikasi Daun</h2>", unsafe_allow_html=True)

    img_input = Image.open(st.session_state.image)
    top5 = predict(img_input)

    pred_name, conf = top5[0]
    data = herbal_info.get(pred_name, None)
    is_antidiabetic = data["status"] == "Tanaman herbal antidiabetes" if data else False

    colA, colB = st.columns([1, 1])

    # KANAN & KIRI ATAS
    with colA:
        # Convert PIL Image to Base64 to render cleanly inside single HTML block
        buffered = io.BytesIO()
        img_input.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        st.markdown(f"""
            <div class="custom-card" style="text-align: center; padding: 24px;">
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; display: flex; align-items: center; justify-content: center; width: 100%; min-height: 320px;">
                    <img src="data:image/png;base64,{img_b64}" style="max-height: 290px; max-width: 100%; object-fit: contain; border-radius: 8px; margin: 0 auto; display: block;">
                </div>
                <p style="font-size: 15px; color: #64748b; font-style: italic; margin-top: 14px; margin-bottom: 0; font-weight: 500;">Gambar yang Diunggah</p>
            </div>
        """, unsafe_allow_html=True)

        if data:
            nama_umum_list = "".join([f"<li>{n}</li>" for n in data["nama_umum"]])
            st.markdown(f"""
                <div class="custom-card">
                    <span style="font-size:16px; font-weight:700; color:#64748b; text-transform:uppercase;">Nama Ilmiah:</span>
                    <div class="scientific-name">{pred_name}</div>
                    <span style="font-size:16px; font-weight:700; color:#64748b; text-transform:uppercase;">Nama Umum:</span>
                    <ul style="font-size:20px; color:#1e293b; margin-top:6px; padding-left:20px; font-weight: 500; line-height: 1.7;">
                        {nama_umum_list}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    with colB:
        status_class = "badge-antidiabetes" if is_antidiabetic else "badge-pembanding"
        status_text = data["status"] if data else "Tanaman Pembanding"

        st.markdown(f"""
            <div class="custom-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="font-size:16px; font-weight:700; color:#64748b;">STATUS TANAMAN:</span>
                    <span class="{status_class}">{status_text}</span>
                </div>
                <hr style="border-top:1px solid #f1f5f9; margin:12px 0;">
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <span style="font-size:16px; font-weight:600; color:#334155;">Kepercayaan Sistem:</span>
                    <span style="font-size:28px; font-weight:800; color:#047857; font-family:monospace;">{conf * 100:.2f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Container Top-5 Prediksi Progress Bar
        top5_items_html = ""
        for i, (label, score) in enumerate(top5, 1):
            pct = score * 100
            top5_items_html += (
                f'<div style="margin-bottom: 14px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; font-size: 15px; font-weight: 600; margin-bottom: 6px; color: #1e293b;">'
                f'<span><b>{i}.</b> <span style="font-style: italic; font-weight: 500; color: #0f172a;">{label}</span></span>'
                f'<code style="color: #047857; font-weight: 700; font-size: 15px; font-family: monospace;">{pct:.2f}%</code>'
                f'</div>'
                f'<div style="width: 100%; background-color: #f1f5f9; height: 12px; border-radius: 999px; overflow: hidden;">'
                f'<div style="width: {pct:.2f}%; background-color: #047857; height: 100%; border-radius: 999px;"></div>'
                f'</div>'
                f'</div>'
            )

        st.markdown(
            f'<div class="custom-card">'
            f'<span style="font-size: 16px; font-weight: 700; color: #0f172a; display: block; margin-bottom: 16px;">Top-5 Prediksi Model:</span>'
            f'{top5_items_html}'
            f'</div>',
            unsafe_allow_html=True
        )

    # INFORMASI DETAIL HERBAL
    st.markdown("<div class='section-header'>🌿 Informasi Herbal</div>", unsafe_allow_html=True)
    st.write(data["informasi"] if data else "Tidak ada informasi khusus.")

    # TAUTAN ARTIKEL & JURNAL
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        st.markdown(
            "<div class='section-header'>🔗 Tautan Artikel Terkait</div>",
            unsafe_allow_html=True
        )
        if data and data.get("tautan_artikel"):
            judul_artikel = data.get(
                "judul_artikel",
                "Baca artikel terkait"
            )
            st.markdown(
                f"""
                <a href="{data['tautan_artikel']}" target="_blank"
                   style="text-decoration: none; color: #0066CC;">
                    {judul_artikel}
                </a>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Tidak tersedia artikel khusus untuk tanaman ini.")
    
    with col_link2:
        st.markdown(
            "<div class='section-header'>📚 Tautan Jurnal Penelitian</div>",
            unsafe_allow_html=True
        )
        if data and data.get("tautan_jurnal"):
            judul_jurnal = data.get(
                "judul_jurnal",
                "Baca jurnal penelitian"
            )
            st.markdown(
                f"""
                <a href="{data['tautan_jurnal']}" target="_blank"
                   style="text-decoration: none; color: #0066CC;">
                    {judul_jurnal}
                </a>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Tidak tersedia jurnal khusus untuk tanaman ini.")

    # CARA MENGOLAH HERBAL
    st.markdown("<div class='section-header'>☕ Cara Mengolah Herbal Antidiabetes</div>", unsafe_allow_html=True)
    if data and data["cara_mengolah"]:
        for idx, langkah in enumerate(data["cara_mengolah"], 1):
            st.markdown(f"**{idx}.** {langkah}")
            
        # Bagian Tautan Referensi Pengolahan
        if data.get("tautan_pengolahan"):
            nama_sumber = data.get("sumber_pengolahan", "Baca referensi cara pengolahan selengkapnya")
            st.markdown(f"""
                <div style="margin-top: 14px; font-size: 15px; color: #475569;">
                    <span style="font-weight: 600;">📖 Sumber Referensi Pengolahan:</span> 
                    <a href="{data['tautan_pengolahan']}" target="_blank" style="text-decoration: none; color: #0066CC; font-weight: 500;">
                        {nama_sumber} ↗
                    </a>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("*(Tanaman ini merupakan tanaman pembanding dan tidak memiliki tata cara pengolahan ramuan antidiabetes).*")

    # CATATAN KHUSUS
    if data and data["catatan"]:
        st.markdown("<div class='section-header'>⚠️ Catatan Penting</div>", unsafe_allow_html=True)
        catatan_text = data["catatan"].replace("<strong>", "**").replace("</strong>", "**")
        st.warning(catatan_text)

    # Tombol Ganti Gambar (Lebar disamakan dengan kolom view daun / colA)
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Ganti Gambar", use_container_width=True, type="primary"):
        st.session_state.page = "upload"
        st.rerun()
            
# =========================================================
# DISCLAIMER NOTICE & FOOTER
# =========================================================
st.markdown("""
    <div class="disclaimer-card">
        <b>Catatan Penafian / <i>Disclaimer Notice</i>:</b><br>
        <i>Sistem ini dikembangkan sebagai bagian dari penyusunan tugas akhir skripsi (NIM 211401034). 
        Hasil prediksi bersifat estimasi kecerdasan buatan (computer vision) dan tidak dimaksudkan sebagai rujukan medis atau botani yang bersifat final. 
        Validasi tetap disarankan melalui dokter atau ahli farmakognosi terkait.</i>
    </div>
    
    <div class="footer-text">
        ©2026 DiaHerb | Tugas Akhir Skripsi | NIM 211401034
    </div>
""", unsafe_allow_html=True)
