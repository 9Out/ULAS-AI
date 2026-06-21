import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
from google_play_scraper import Sort, reviews, app
from urllib.parse import urlparse, parse_qs
import plotly.express as px
from wordcloud import WordCloud
from preprocess import preprocess_playstore
from collections import Counter
from nltk.util import ngrams

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="ULAS-AI", layout="wide")

# --- FUNGSI LOAD PAYLOAD MODEL (Updated) ---
@st.cache_resource
def load_model_payload():
    try:
        # Load file joblib yang baru (v8) yang berisi dictionary
        return joblib.load('model/playstore_sentiment_pipeline_v8.joblib')
    except Exception as e:
        st.error(f"Gagal memuat model! Pastikan path benar dan library scikit-learn terinstall. Error: {e}")
        st.stop()

# Panggil payload-nya
payload = load_model_payload()
model = payload['pipeline']
metadata = payload['metadata']
env_info = payload['env']

# --- FUNGSI EKSTRAK ID APLIKASI ---def get_app_id(input_user):
def get_app_info(input_user):
    """
    Mendukung:
    1. URL Play Store
    2. App ID langsung
    """

    if not input_user:
        return None

    input_user = input_user.strip()

    app_id = None
    # ==========================
    # 1. URL Play Store
    # ==========================
    if "play.google.com" in input_user:

        try:
            parsed = urlparse(input_user)
            params = parse_qs(parsed.query)

            app_id = params.get("id", [None])[0]

            if app_id:
                st.toast(f"🔗 URL terdeteksi: {app_id}")

        except Exception:
            pass

    # ==========================
    # 2. App ID langsung
    # ==========================
    if not app_id:

        match = re.search(r'id=([a-zA-Z0-9._]+)', input_user)

        if match:
            app_id = match.group(1)
            st.toast(f"🔎 App ID terdeteksi: {app_id}")
    
    if not app_id:

        app_id_pattern = r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)+$"

        if re.match(app_id_pattern, input_user):
            app_id = input_user
            st.toast(f"📦 App ID terdeteksi: {app_id}")

    if not app_id:
        return None
    
    # ==========================
    # 3. Validasi ke Play Store
    # ==========================
    try:

        info = app(
            app_id,
            lang="id",
            country="id"
        )

        return info

    except Exception:
        return None


# --- UI UTAMA ---
col_head, col_spacer = st.columns([4, 1])
with col_head:
    st.title("📊 Analisis Sentimen Aplikasi Play Store")
    st.write("Masukkan Link URL play store atau ID aplikasi untuk melihat analisis sentimen dari ulasan terbaru penggunanya.")

# Input App ID dan Jumlah Data
col1, col2 = st.columns([4, 1])

with col1:
    app_input = st.text_input(
        "Link URL / ID Aplikasi Play Store (contoh: com.duolingo)",
        ""
    )

with col2:
    jumlah_data = st.selectbox(
        "Jumlah Data",
        [500, 1000, 2000, 5000, 10000],
        index=0  # default 500
    )
    
# kolom bantuan
with st.expander("❓ Bantuan Pengguna"):
    tab_cara, tab_format = st.tabs(["📖 Cara Mencari", "📋 Format Input"])
    
    with tab_cara:
        st.link_button(
            "🔍 Buka Google Play Store",
            "https://play.google.com/store"
        )
        
        st.markdown("""
        **Cara mendapatkannya:**
        1. Cari aplikasi di Google Play Store.
        2. Buka halaman aplikasi.
        3. Salin URL halaman aplikasi.
        4. Jika menggunakan aplikasi Play Store di HP, pilih **Bagikan (Share)** lalu **Salin Tautan (Copy Link)**.
        5. Jika hanya ingin menggunakan ID aplikasi, ambil teks setelah `id=`.

        Contoh:
        ```
        https://play.google.com/store/apps/details?id=com.duolingo
        ```
        ID aplikasinya adalah:
        ```
        com.duolingo
        ```
        """)
    
    with tab_format:
        st.markdown("""           
        Anda dapat memasukkan:

        **1. Link Play Store**
        - https://play.google.com/store/apps/details?id=com.duolingo
        - https://play.google.com/store/apps/details?id=com.naver.linewebtoon

        **2. ID aplikasi**
        - com.duolingo
        - id=com.mobile.legends

        Sistem akan mendeteksi format input secara otomatis.
        """)
    
mulai_btn = st.button("Mulai Analisis", type="primary")

# --- SIDEBAR: INFORMASI MODEL (MENAMPILKAN METADATA) ---
with st.sidebar:
    st.header("🤖 Informasi Model ML")
    st.info(f"**Versi:** {metadata['version']}")
    
    # Tampilkan Metrics
    st.subheader("Performa Model")
    st.metric(label="Akurasi Eksperimen", value= "97,4%")
    # st.metric(label="Akurasi Eksperimen", value=f"{metadata['metrics']['accuracy'] * 100}%")
    st.caption(f"Mean CV Score: {metadata['metrics']['cv_f1_mean']:.4f}")
    
    # Detail Tambahan
    st.subheader("Spesifikasi")
    st.write(f"**Algoritma:** {metadata['model']}")
    st.write(f"**Ekstraksi Fitur:** {metadata['feature_extraction']}")
    st.write(f"**Confussion Matrix:** {metadata['confusion_matrix']}")
    
    # Info Environment
    with st.expander("Detail Library Environment"):
        st.code(f"Python: {env_info['python']}\nSklearn: {env_info['sklearn']}\nJoblib: {env_info['joblib']}")

# --- LOGIKA ANALISIS ---
if mulai_btn and app_input:
    if not app_input.strip():
        st.warning("Masukkan Link/URL Play Store yang berisi ID aplikasi, atau App ID terlebih dahulu.")
        st.stop()

    info_app = get_app_info(app_input)

    if not info_app:
        st.error(
            "Aplikasi tidak ditemukan di Google Play Store atau App ID tidak valid."
        )
        st.error(
            "Coba cek kembali ID aplikasi atau masukkan URL Play Store yang berisi ID aplikasi."
        )
        st.stop()
        
    app_id = info_app["appId"]
    judul = info_app["title"]
    icon = info_app["icon"]
    genre = info_app["genre"]
    core = info_app.get('score', '-')
    rating_text = f"{score:.2f}" if score is not None else "Belum memiliki rating"
    total_reviews = info_app["reviews"]
    
    # PENGENALAN APLIKASI
    st.markdown("### 📱 Informasi Aplikasi")

    col1, col2 = st.columns([1, 4])

    with col1:
        st.image(icon, width=160)

    with col2:
        st.markdown(
            f"#### {judul}"
        )
        st.markdown(
            f"**Genre:** {genre}"
        )
        st.markdown(
            f"**Rating:** ⭐ {rating_text}"
        )
        installs = info_app.get("installs", "-")

        st.markdown(
            f"**Jumlah Instalasi:** {installs}"
        )
            
    deskripsi = info_app.get("description", "")
    
    if len(deskripsi) > 1000:
        deskripsi = deskripsi[:600] + "..."
    st.info(deskripsi)
    
    # check dulu apakah ada review
    if not total_reviews:
        st.warning("Aplikasi ini belum memiliki ulasan.")
        st.stop()
    
    # Animasi Loading
    with st.spinner(f'Sedang mengambil {jumlah_data} ulasan data dari {judul}, membersihkan teks alay, dan menganalisis sentimen... 🚀'):
        try:
            # 1. Scraping Data Play Store
            result, _ = reviews(
                app_id,
                lang='id', # Tarik ulasan berbahasa Indonesia
                country='id',
                sort=Sort.NEWEST,
                count=jumlah_data # banyak ulasan dipilih user
            )
            
            if not result:
                st.warning("Ulasan tidak ditemukan atau ID aplikasi salah.")
                st.stop()

            # 2. Preprocessing Data ke DataFrame
            df = pd.DataFrame(result)
            df = df[['content', 'score', 'reviewCreatedVersion', 'at']]
            
            # A. Buat kolom baru berisi Tahun & Bulan (contoh: 2023-10)
            df['YearMonth'] = pd.to_datetime(df['at']).dt.to_period('M')
            
            # B. Cari versi aplikasi terbanyak (modus) di setiap bulannya
            # Kita abaikan data yang kosong (None/NaN) saat mencari modus
            mode_per_month = df.dropna(subset=['reviewCreatedVersion']) \
                               .groupby('YearMonth')['reviewCreatedVersion'] \
                               .apply(lambda x: x.mode()[0] if not x.mode().empty else None)
            
            # C. Fungsi untuk menambal data yang hilang
            def impute_version(row):
                # Jika versinya kosong/None
                if pd.isna(row['reviewCreatedVersion']):
                    # Ambil versi terbanyak di bulan yang sama
                    imputed = mode_per_month.get(row['YearMonth'])
                    # Jika di bulan itu ternyata semua ulasan versinya kosong, baru pakai 'Unknown Version'
                    return imputed if pd.notna(imputed) else 'Unknown Version'
                return row['reviewCreatedVersion']

            # D. Terapkan fungsi ke dataframe
            df['reviewCreatedVersion'] = df.apply(impute_version, axis=1)
            
            # E. Bersihkan kolom bantuan agar dataframe tetap rapi
            df = df.drop(columns=['YearMonth', 'at'])
            
            # Menerapkan fungsi preprocess_playstore ke kolom 'content'
            df['cleaned_content'] = df['content'].apply(preprocess_playstore)
            
            # 3. Prediksi Sentimen menggunakan Model
            # Prediksi sentimen numerik (0 atau 1)
            predicted_sentiments = model.predict(df['cleaned_content'])
            # Konversi label numerik menjadi label teks (Negatif/Positif) menggunakan metadata
            predicted_labels = [metadata['labels'][pred] for pred in predicted_sentiments]
            df['sentiment'] = predicted_labels
            
            # Cek apakah model mendukung probabilitas (untuk Bar Chart)
            # Karena model kita sekarang sudah diekstraksi dari dictionary, 
            # predict_proba() sekarang berjalan di level model pipeline
            try:
                proba = model.predict_proba(df['cleaned_content'])
                df['confidence'] = proba.max(axis=1) # Ambil probabilitas tertinggi
            except:
                # Fallback jika model tidak punya predict_proba (misal SVM tanpa flag probability)
                # Dalam kasus logistic regression ini harusnya ada
                df['confidence'] = 1.0

            st.success("Analisis Selesai! Berikut hasilnya:")

            # --- VISUALISASI HASIL ---
            total_ulasan = len(df)
            sentiment_percent = (
                            df['sentiment']
                            .value_counts(normalize=True)
                            .mul(100)
                        )
            negatif_pct = sentiment_percent['Negatif']
            positif_pct = sentiment_percent['Positif']

            kata_neg = " ".join(
                df[df['sentiment']=='Negatif']['cleaned_content']
            ).split()

            # top_pos = Counter(kata_pos).most_common(10)
            top_neg = Counter(kata_neg).most_common(10)

            # kata_pos_dominan = top_pos[0][5] if top_pos else "-"
            kata_neg_dominan = top_neg[:5] if top_neg else "-"
                        
            
            # Baris 1: Pie Chart & Bar Chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Persentase Sentimen")
                pie_fig = px.pie(df, names='sentiment', color='sentiment', 
                                 color_discrete_map={'Positif':'#2ECB71', 'Negatif':'#E74C3C'},
                                 hole=0.4)
                st.plotly_chart(pie_fig, width="stretch")
                
                positif_pct = sentiment_percent['Positif']
                negatif_pct = sentiment_percent['Negatif']

                st.info(f"""
                #### **Interpretasi Distribusi Sentimen**

                Dari seluruh ulasan yang dianalisis:

                - Sentimen Positif: {positif_pct:.1f}%
                - Sentimen Negatif: {negatif_pct:.1f}%

                Mayoritas pengguna memberikan tanggapan
                {'positif' if positif_pct > negatif_pct else 'negatif'}
                terhadap aplikasi.
                """)
                
            with col2:
                st.subheader("Distribusi Probabilitas / Skor")
                # Box plot untuk distribusi confidence
                bar_fig = px.box(df, x='sentiment', y='confidence', color='sentiment',
                                 color_discrete_map={'Positif':'#2ECB71', 'Negatif':'#E74C3C'},
                                 labels={'sentiment':'Sentimen Prediksi', 'confidence':'Skor Keyakinan'},
                                 title="Tingkat Keyakinan Model per Sentimen")
                st.plotly_chart(bar_fig, width="stretch")
                
                # Statistik confidence positif
                pos_conf = df[df['sentiment']=='Positif']['confidence']

                avg_pos = (pos_conf.mean() * 100)
                med_pos = (pos_conf.median() * 100)
                min_pos = (pos_conf.min() * 100)
                max_pos = (pos_conf.max() * 100)

                # Statistik confidence negatif
                neg_conf = df[df['sentiment']=='Negatif']['confidence']

                avg_neg = (neg_conf.mean() * 100)
                med_neg = (neg_conf.median() * 100)
                min_neg = (neg_conf.min() * 100)
                max_neg = (neg_conf.max() * 100)

                # Confidence tinggi
                high_conf_pct = (
                    (df['confidence'] >= 0.8).sum()
                    / len(df)
                ) * 100
                
                # Untuk membuat Kesimpulan
                if high_conf_pct >= 90:
                    kualitas_model = "sangat yakin terhadap hampir seluruh hasil klasifikasi"
                elif high_conf_pct >= 75:
                    kualitas_model = "cukup konsisten dan yakin dalam melakukan klasifikasi"
                elif high_conf_pct >= 60:
                    kualitas_model = "memiliki tingkat keyakinan yang cukup baik, meskipun masih terdapat sejumlah prediksi dengan keyakinan rendah"
                else:
                    kualitas_model = "menunjukkan variasi keyakinan yang cukup besar sehingga beberapa prediksi perlu ditinjau lebih lanjut"
                
                # kesimpulan model
                if avg_pos > avg_neg + 0.1:
                    perbandingan = "Model cenderung lebih yakin saat mengidentifikasi ulasan positif dibandingkan ulasan negatif."
                elif avg_neg > avg_pos + 0.1:
                    perbandingan = "Model cenderung lebih yakin saat mengidentifikasi ulasan negatif dibandingkan ulasan positif."
                else:
                    perbandingan = "Tingkat keyakinan model pada sentimen positif dan negatif relatif seimbang."
                
                st.info(f"""
                    #### Interpretasi Tingkat Keyakinan Model

                    Model menunjukkan tingkat keyakinan yang cukup tinggi terhadap hasil klasifikasi.

                    📈 **Sentimen Positif**
                    - Rata-rata skor keyakinan: **{avg_pos:.1f}%**
                    - Median: **{med_pos:.1f}%**
                    - Rentang skor: **{min_pos:.1f}% – {max_pos:.1f}%**

                    📉 **Sentimen Negatif**
                    - Rata-rata skor keyakinan: **{avg_neg:.1f}%**
                    - Median: **{med_neg:.1f}%**
                    - Rentang skor: **{min_neg:.1f}% – {max_neg:.1f}%**

                    🎯 Berdasarkan data diatas, rata-rata keyakinan model terhadap prediksi sentimen positif sebesar **{avg_pos:.1f}%** dan sentimen negatif sebesar **{avg_neg:.1f}%**. Secara keseluruhan,
                    sebanyak **{high_conf_pct:.1f}%** prediksi memiliki skor keyakinan di atas **0.80**, yang menunjukkan bahwa model **{kualitas_model}**. {perbandingan}.
                    """)
            
            st.subheader("Persentase Keluhan Berdasarkan Versi Aplikasi")
            
            df_neg = df[df['sentiment'] == 'Negatif']
            
            if not df_neg.empty:
                # 1. Dapatkan total semua keluhan (sebagai pembagi / 100%)
                total_keluhan = len(df_neg)
                
                # 2. Hitung jumlah keluhan di masing-masing versi
                keluhan_versi = df_neg['reviewCreatedVersion'].value_counts().reset_index()
                keluhan_versi.columns = ['Versi', 'Jumlah Keluhan']
                
                # 3. RUMUS PERSENTASE: (Jumlah Keluhan per Versi / Total Keluhan) * 100
                keluhan_versi['Persentase (%)'] = (keluhan_versi['Jumlah Keluhan'] / total_keluhan) * 100
                
                # 4. Bulatkan 1 angka di belakang koma (contoh: 45.678 jadi 45.7)
                keluhan_versi['Persentase (%)'] = keluhan_versi['Persentase (%)'].round(1)
                
                # 5. Buat label teks khusus untuk ditampilkan di atas batang chart
                keluhan_versi['Label'] = keluhan_versi['Persentase (%)'].astype(str) + '%'
                
                # Visualisasi Bar Chart
                keluhan_fig = px.bar(
                    keluhan_versi.head(10), # Ambil Top 10 versi terbanyak
                    x='Versi', 
                    y='Persentase (%)', 
                    text='Label', # Munculkan teks persentase
                    labels={'Persentase (%)': 'Persentase Keluhan (%)'},
                    color_discrete_sequence=['#E74C3C']
                )
                st.plotly_chart(keluhan_fig, width="stretch")
            else:
                st.info("Wah, hebat! Tidak ada keluhan (sentimen negatif) yang ditemukan pada sampel ini.")
            
            versi_tertinggi = keluhan_versi.iloc[0]

            st.info(f"""
            #### **Interpretasi Grafik Keluhan Berdasarkan Versi**

            Grafik menunjukkan persentase ulasan negatif pada setiap versi aplikasi.

            Versi **{versi_tertinggi['Versi']}**
            memiliki tingkat keluhan tertinggi sebesar
            **{versi_tertinggi['Persentase (%)']:.1f}%** atau **{versi_tertinggi['Jumlah Keluhan']}** keluhan.
            
            Ulasan negatif yang dominan pada versi ini sering mengandung kata **"{kata_neg_dominan}"**, 

            Hal ini mengindikasikan bahwa versi tersebut kemungkinan
            mengandung bug, perubahan fitur, atau masalah performa
            yang memicu banyak keluhan pengguna.
            """)

            # # Baris 3: Wordcloud
            st.subheader("Kata yang Paling Sering Muncul (Wordcloud)")
            col3, col4 = st.columns(2)
            
            def generate_wordcloud(text_data, title, colormap):

                all_bigrams = []

                for review in text_data.dropna():
                    words = review.split()

                    all_bigrams.extend(
                        [" ".join(bg) for bg in ngrams(words, 2)]
                    )

                freq = Counter(all_bigrams)
                

                # buat threshold dinamis menyesuaikan review yang ada
                for threshold in [3, 2, 1]:
                    filtered_freq = {
                        k:v for k,v in freq.items()
                        if v >= threshold
                    }

                    if len(filtered_freq) >= 5:
                        freq = filtered_freq
                        break

                if len(freq) > 0:

                    wc = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap=colormap
                    ).generate_from_frequencies(freq)

                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(title)

                    return fig

                return None
            
            with col3:
                fig_pos = generate_wordcloud(df[df['sentiment'] == 'Positif']['cleaned_content'], 'Bigram Positif', 'Greens')
                if fig_pos: st.pyplot(fig_pos)
                
            with col4:
                fig_neg = generate_wordcloud(df_neg['cleaned_content'], 'Bigram Negatif', 'Reds')
                if fig_neg: st.pyplot(fig_neg)
                
                
            # Positif
            bigram_pos = []

            for text in df[df['sentiment']=='Positif']['cleaned_content'].dropna():
                words = text.split()
                bigram_pos.extend(
                    [" ".join(bg) for bg in ngrams(words, 1)]
                )

            top_poso = Counter(bigram_pos).most_common(10)

            # Negatif
            bigram_neg = []

            for text in df[df['sentiment']=='Negatif']['cleaned_content'].dropna():
                words = text.split()
                bigram_neg.extend(
                    [" ".join(bg) for bg in ngrams(words, 1)]
                )

            top_nega = Counter(bigram_neg).most_common(10)

            st.markdown("---")
            st.subheader("Top 10 Kata Dominan")

            col5, col6 = st.columns(2)

            with col5:
                st.write("### Sentimen Positif")

                df_top_pos = pd.DataFrame(
                    top_poso,
                    columns=["Kata", "Frekuensi"]
                )

                st.dataframe(
                    df_top_pos,
                    width="stretch"
                )

            with col6:
                st.write("### Sentimen Negatif")

                df_top_neg = pd.DataFrame(
                    top_nega,
                    columns=["Kata", "Frekuensi"]
                )
                
                st.dataframe(
                    df_top_neg,
                    width="stretch"
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

elif mulai_btn and not app_input:
    st.warning("Jangan lupa masukkan Link atau ID Aplikasinya ya!")