# Fast RAG Image Retrieval & Description
CLIP + FAISS + BLIP + FLAN-T5

## Overview
Repository ini berisi implementasi Retrieval-Augmented Generation (RAG) berbasis citra
untuk klasifikasi dan deskripsi ras kucing dan anjing. Sistem menggabungkan image
retrieval berbasis kemiripan visual dan model generatif untuk menghasilkan deskripsi
yang faktual tanpa hallucination ras.

---

## System Architecture (RAG)

Pipeline sistem terdiri dari dua tahap utama:

### 1. Image Retrieval
- Model: CLIP ViT-L/14
- Embedding: cosine-normalized image embeddings
- Indexing: FAISS (IndexFlatIP)
- Output:
  - Top-K gambar paling mirip
  - Ras dominan (majority voting)
  - Confidence score

### 2. Generative Description
- BLIP digunakan untuk menghasilkan caption visual dari gambar query
- FLAN-T5 digunakan untuk memperhalus deskripsi teks
- Aturan utama:
  - Model generatif tidak menentukan ras
  - Ras hanya berasal dari hasil retrieval
  - Tidak ada penambahan informasi eksternal
  - Tidak terjadi hallucination

---

## Repository Structure

```text
.
├── Sistem_Retrieval_Augmented_Generation_(RAG).ipynb
│
├── app.py
│
├── models/
│   ├── embeddings.npy
│   ├── labels_clean.npy
│   ├── image_paths.npy
│   └── faiss.index
│
├── archive/
│   └── images/
│
└── README.md
```

---

## Models Used

### Retrieval
- Image Encoder: openai/clip-vit-large-patch14
- Similarity Search: FAISS (Inner Product / Cosine Similarity)

### Generative
- Visual Captioning: Salesforce/blip-image-captioning-base
- Text Refinement: google/flan-t5-base

Catatan:
Model besar seperti BLIP-2 dan FLAN-T5 Large digunakan pada notebook Colab
untuk eksperimen dan validasi arsitektur. Aplikasi Streamlit menggunakan model
base untuk efisiensi dan stabilitas saat deployment.

---

## Colab Notebook
Notebook digunakan untuk:
- Membangun embedding CLIP
- Membersihkan dan menormalkan label dataset
- Membuat FAISS index
- Menyimpan artifact model ke folder models/

Fokus utama notebook adalah eksperimen dan validasi pipeline.

---

## Streamlit Application

Aplikasi Streamlit menyediakan:
- Upload gambar kucing atau anjing
- Top-K image retrieval
- Prediksi ras dan confidence
- Deskripsi teks berbasis RAG
- Visualisasi hasil retrieval

---

## How to Use (Step-by-Step)

### Step 1: Clone Repository
Clone repository ini ke komputer lokal.

```bash
git clone https://github.com/MasaraPaya/Sistem-Retrieval-Augmented-Generation-RAG.git
cd Sistem-Retrieval-Augmented-Generation-RAG

```
---

### Step 2: Prepare Environment
Disarankan menggunakan virtual environment.

python -m venv venv
source venv/bin/activate        # Linux / MacOS
venv\Scripts\activate           # Windows

Install dependensi.

pip install -r requirements.txt

---

### Step 3: Prepare Models and Index
Pastikan folder berikut tersedia sebelum menjalankan aplikasi:

models/
- embeddings.npy
- labels_clean.npy
- image_paths.npy
- faiss.index

File-file ini dihasilkan dari notebook Colab.

---

### Step 4: Prepare Dataset
Pastikan dataset gambar berada di path berikut:

archive/images/

Nama file gambar harus konsisten dengan image_paths.npy.

---

### Step 5: Run Streamlit Application

streamlit run app.py

Aplikasi akan berjalan secara lokal dan dapat diakses melalui browser.

---

### Step 6: Using the Application
1. Upload gambar kucing atau anjing.
2. Sistem melakukan image retrieval untuk mencari gambar paling mirip.
3. Ras dominan ditentukan berdasarkan majority voting Top-K retrieval.
4. Confidence score ditampilkan.
5. BLIP menghasilkan caption visual.
6. FLAN-T5 memperhalus deskripsi berdasarkan hasil retrieval.
7. Hasil akhir ditampilkan beserta visualisasi gambar retrieval.

---

## Non-Hallucination Design

Sistem dirancang untuk memastikan:
- Ras hanya berasal dari hasil retrieval
- Model generatif tidak membuat klaim ras baru
- Confidence ditampilkan secara eksplisit
- Deskripsi tetap jujur saat confidence rendah

Pendekatan ini mendukung prinsip trustworthy dan explainable AI.

---

## Conclusion

Pipeline RAG yang diimplementasikan bersifat:
- Valid secara metodologis
- Stabil untuk deployment
- Bebas hallucination ras
- Sesuai dengan instruksi akademik RAG multimodal
