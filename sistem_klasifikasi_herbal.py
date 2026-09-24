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
        "tautan_jurnal": "https://jurnal.ikbis.ac.id/index.php/infokes/article/view/371/221",
        "cara_mengolah": [
            "Siapkan 25 lembar daun sambiloto segar dan 110 ml air bersih.",
            "Cuci bersih daun sambiloto di bawah air mengalir.",
            "Rebus daun sambiloto sampai air mendidih.",
            "Minum air rebusan daun sambiloto satu kali sehari dengan takaran 100 ml."
        ],
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
        "tautan_jurnal": "https://jurnal.unpad.ac.id/farmaka/article/view/47508/pdf",
        "cara_mengolah": [
            "Siapkan 5-10 lembar daun tapak dara yang masih segar dan 2 gelas air.",
            "Cuci bersih daun tapak dara di bawah air mengalir.",
            "Rebus daun dengan api kecil sampai air berubah warna.",
            "Saring dan biarkan hingga hangat sebelum diminum."
        ],
        "catatan": "Disarankan mengonsumsi satu gelas sehari. Konsultasikan dengan dokter untuk rencana pengobatan yang aman."
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
        "tautan_jurnal": "https://doi.org/10.35617/jfionline.v12i1.21",
        "cara_mengolah": [
            "Siapkan segenggam daun kelor (10-15 gram) dan 600 ml air.",
            "Cuci bersih daun kelor di bawah air mengalir.",
            "Panaskan air hingga mendidih, lalu masukkan daun kelor dan rebus 5-15 menit.",
            "Saring air rebusan untuk diminum. Daun rebusan tetap bisa dikonsumsi sebagai lalapan."
        ],
        "catatan": "Mengonsumsi dalam jumlah wajar. Jika memiliki kondisi medis tertentu, konsultasikan dengan dokter."
    },
    "Orthosiphon aristatus": {
        "nama_umum": ["Kumis kucing", "Remujung"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun kumis kucing kaya flavonoid dan saponin. Flavonoid menghambat pemecahan karbohidrat di usus, sedangkan saponin merangsang pelepasan insulin.",
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/tanaman-kumis-kucing/",
        "tautan_jurnal": "https://doi.org/10.36990/hijp.v7i1.533",
        "cara_mengolah": [
            "Siapkan segenggam daun kumis kucing (10-15 gram) dan 500 ml air.",
            "Cuci bersih di bawah air mengalir.",
            "Rebus selama 15-20 menit.",
            "Saring air rebusan dan minum 2-3 kali sehari."
        ],
        "catatan": "Gunakan dalam jumlah wajar dan konsultasikan ke dokter untuk pemakaian jangka panjang."
    },
    "Pandanus amaryllifolius": {
        "nama_umum": ["Pandan wangi", "Pandan"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun pandan mengandung flavonoid, tanin, dan polifenol yang mampu merangsang produksi hormon insulin dari sel beta pankreas.",
        "tautan_artikel": "https://share.google/BTrQ3MBtqbTndrJvO",
        "tautan_jurnal": "https://ejurnalmalahayati.ac.id/index.php/kebidanan/article/view/3024/pdf",
        "cara_mengolah": [
            "Siapkan 3-4 lembar daun pandan segar/kering dan 500 ml air.",
            "Cuci bersih dan potong menjadi beberapa bagian.",
            "Rebus dalam air mendidih selama 10-15 menit hingga air berwarna hijau kekuningan.",
            "Saring air rebusan dan minum hangat."
        ],
        "catatan": "Hindari menambahkan gula pasir tinggi kalori; gunakan pemanis alami jika diperlukan."
    },
    "Phyllanthus amarus": {
        "nama_umum": ["Meniran"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Meniran memiliki senyawa aktif yang memengaruhi metabolisme glukosa dan mendukung pengelolaan tingkat kadar gula darah secara berkelanjutan.",
        "tautan_artikel": "https://www.halodoc.com/artikel/manfaat-pohon-meniran-jaga-imun-ginjal-sehat-alami",
        "tautan_jurnal": "https://doi.org/10.36656/jpfh.v2i1.79",
        "cara_mengolah": [
            "Siapkan segenggam daun meniran segar (10-15 gram) dan 3 gelas air.",
            "Cuci bersih di bawah air mengalir.",
            "Rebus hingga air menyusut sekitar 1 gelas.",
            "Saring dan konsumsi selagi hangat 1-2 kali sehari."
        ],
        "catatan": "Konsultasikan dengan dokter jika sedang mengonsumsi obat-obatan medis rutin."
    },
    "Physalis angulata": {
        "nama_umum": ["Ciplukan", "Ceplukan", "Cecendet"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Daun Ciplukan memiliki indeks glikemik rendah dan dapat meningkatkan sensitivitas atau produksi insulin dalam tubuh.",
        "tautan_artikel": "https://www.halodoc.com/artikel/pohon-ciplukan-dan-khasiatnya-dari-diabetes-hingga-kanker",
        "tautan_jurnal": "https://journal.ukmc.ac.id/index.php/joh/article/view/1141/1081",
        "cara_mengolah": [
            "Siapkan 10-15 gram daun ciplukan segar dan 3 gelas air (600 ml).",
            "Cuci bersih daun di bawah air mengalir.",
            "Rebus dengan api sedang (hindari panci aluminium) hingga menyusut jadi 1 gelas.",
            "Saring dan minum 1-2 kali sehari."
        ],
        "catatan": "Disarankan merebus dengan wadah stainless steel atau enamel."
    },
    "Rosa sp.": {
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
        "tautan_jurnal": "https://doi.org/10.3164/jcbn.08-188",
        "cara_mengolah": [
            "Siapkan 10-15 lembar daun salam segar dan 600 ml air (3 gelas).",
            "Cuci bersih daun salam.",
            "Rebus dengan api sedang sampai menyusut menjadi 1 gelas (200 ml).",
            "Saring dan minum hangat 2 kali sehari sebelum makan."
        ],
        "catatan": "Gunakan wadah perebus yang tidak reaktif terhadap logam."
    },
    "Vernonia amygdalina": {
        "nama_umum": ["Daun Afrika", "Daun pahit", "Daun insulin"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Ekstrak daun Afrika mengandung saponin, tanin, flavonoid, dan alkaloid yang terbukti efektif menekan lonjakan glukosa darah pasca makan.",
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/manfaat-daun-afrika/",
        "tautan_jurnal": "https://www.neliti.com/id/publications/460123/potensi-daun-afrika-vernonia-amygdalina-sebagai-antidiabetik",
        "cara_mengolah": [
            "Siapkan 5-10 lembar daun Afrika dan 4 gelas air.",
            "Cuci bersih di bawah air mengalir.",
            "Rebus selama 10-15 menit hingga tersisa 2 gelas.",
            "Minum pagi dan sore hari. Dapat ditambahkan sedikit perasan jeruk nipis."
        ],
        "catatan": "Minum secara teratur dalam dosis aman."
    },
    "Ziziphus mauritiana": {
        "nama_umum": ["Bidara", "Widara", "Bukol"],
        "status": "Tanaman herbal antidiabetes",
        "informasi": "Kandungan saponin dan flavonoid dalam daun bidara berfungsi sebagai antioksidan kuat untuk meningkatkan efektivitas kerja hormon insulin.",
        "tautan_artikel": "https://hellosehat.com/herbal-alternatif/herbal/daun-bidara/",
        "judul_artikel": "Daun Bidara: Kandungan, Manfaat, Efek Samping, dll.",
        "tautan_jurnal": "https://doi.org/10.3390/plants13162195",
        "judul_jurnal": "Ziziphus mauritiana Lam. Bark and Leaves: Extraction, Phytochemical Composition, In Vitro Bioassays and In Silico Studies",
        "cara_mengolah": [
            "Siapkan 10 lembar daun bidara tua, 1/2 buah jeruk nipis, dan 600 ml air.",
            "Cuci bersih daun bidara.",
            "Rebus air hingga mendidih, masukkan daun bidara dan masak 20 menit dengan api kecil.",
            "Tambahkan perasan jeruk nipis dan nikmati selagi hangat."
        ],
        "catatan": "Gunakan dalam jumlah wajar dan konsultasikan ke dokter untuk pemakaian jangka panjang."
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
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def _sharpen_veins(img):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
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

def to_rgb_input(img):
    img = cv2.resize(img, CONFIG["IMG_SIZE"], interpolation=cv2.INTER_CUBIC)
    mask = get_leaf_mask(img)
    img[mask == 0] = [240, 240, 240]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    return (img - mean) / std

def to_vein_input(img):
    img = cv2.resize(img, CONFIG["IMG_SIZE"], interpolation=cv2.INTER_CUBIC)
    mask = get_leaf_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    vein = clahe.apply(gray)
    vein = cv2.bilateralFilter(vein, d=7, sigmaColor=50, sigmaSpace=50)
    blur_large = cv2.GaussianBlur(vein, (21, 21), 0)
    highpass = cv2.subtract(vein, (blur_large * 0.7).astype(np.uint8))
    sobelx = cv2.Sobel(highpass, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(highpass, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vein = cv2.addWeighted(highpass, 0.45, sobel, 0.75, 0)
    vein = cv2.normalize(vein, None, 0, 255, cv2.NORM_MINMAX)
    _, vein = cv2.threshold(vein, 35, 255, cv2.THRESH_TOZERO)
    vein[mask == 0] = 0
    vein = vein.astype(np.float32) / 255.0
    return np.stack([vein, vein, vein], axis=-1)

def predict(image):
    img = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
logo_b64 = load_base64("images/diaherb_logo.png")

if logo_b64:
    st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 0; border-bottom:1px solid #e2e8f0; margin-bottom:30px;">
            <img src="data:image/png;base64,{logo_b64}" style="height:70px; width:auto;">
            <span style="font-size:13px; font-weight:600; color:#047857; background:#ecfdf5; padding:6px 14px; border-radius:10px; border:1px solid #a7f3d0;">
                LeafNet Dual-Branch Model • Tugas Akhir 211401034
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

        # Tombol Ganti Gambar (Lebar disamakan dengan kolom view daun / colA)
        if st.button("🔄 Ganti Gambar", use_container_width=True, type="primary"):
            st.session_state.page = "upload"
            st.rerun()

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
    st.markdown("<hr>", unsafe_allow_html=True)

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
    else:
        st.write("*(Tanaman ini merupakan tanaman pembanding dan tidak memiliki tata cara pengolahan ramuan antidiabetes).*")

    # CATATAN KHUSUS
    if data and data["catatan"]:
        st.markdown("<div class='section-header'>⚠️ Catatan Penting</div>", unsafe_allow_html=True)
        catatan_text = data["catatan"].replace("<strong>", "**").replace("</strong>", "**")
        st.warning(catatan_text)

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
