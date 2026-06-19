import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
from google_play_scraper import Sort, reviews
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

# --- FUNGSI EKSTRAK ID APLIKASI ---
def get_app_id(url_or_id):
    """Mengekstrak App ID dari URL (misal: https://play.google.com/store/apps/details?id=com.duolingo)"""
    match = re.search(r'id=([a-zA-Z0-9._]+)', url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()

# --- UI UTAMA ---
col_head, col_spacer = st.columns([4, 1])
with col_head:
    st.title("📊 Analisis Sentimen Aplikasi Play Store")
    st.write("Masukkan Link atau ID aplikasi untuk melihat analisis sentimen dari ulasan terbaru penggunanya.")

# Input App ID dan Jumlah Data
col1, col2 = st.columns([4, 1])

with col1:
    app_input = st.text_input(
        "Link / ID Aplikasi Play Store (contoh: com.duolingo)",
        ""
    )

with col2:
    jumlah_data = st.selectbox(
        "Jumlah Data",
        [500, 1000, 2000, 5000, 10000],
        index=0  # default 500
    )
    
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
    app_id = get_app_id(app_input)
    
    # Animasi Loading
    with st.spinner(f'Sedang mengambil {jumlah_data} ulasan data dari {app_id}, membersihkan teks alay, dan menganalisis sentimen... 🚀'):
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
                **Interpretasi Distribusi Sentimen**

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

                avg_pos = pos_conf.mean()
                med_pos = pos_conf.median()
                min_pos = pos_conf.min()
                max_pos = pos_conf.max()

                # Statistik confidence negatif
                neg_conf = df[df['sentiment']=='Negatif']['confidence']

                avg_neg = neg_conf.mean()
                med_neg = neg_conf.median()
                min_neg = neg_conf.min()
                max_neg = neg_conf.max()

                # Confidence tinggi
                high_conf_pct = (
                    (df['confidence'] >= 0.8).sum()
                    / len(df)
                ) * 100
                
                if high_conf_pct >= 90:
                    kualitas_model = "sangat tinggi"
                elif high_conf_pct >= 75:
                    kualitas_model = "tinggi"
                elif high_conf_pct >= 60:
                    kualitas_model = "cukup baik"
                else:
                    kualitas_model = "perlu dievaluasi lebih lanjut"
                
                st.info(f"""
                    ### Interpretasi Tingkat Keyakinan Model

                    Model menunjukkan tingkat keyakinan yang cukup tinggi terhadap hasil klasifikasi.

                    📈 **Sentimen Positif**
                    - Rata-rata skor keyakinan: **{avg_pos:.3f}**
                    - Median: **{med_pos:.3f}**
                    - Rentang skor: **{min_pos:.3f} – {max_pos:.3f}**

                    📉 **Sentimen Negatif**
                    - Rata-rata skor keyakinan: **{avg_neg:.3f}**
                    - Median: **{med_neg:.3f}**
                    - Rentang skor: **{min_neg:.3f} – {max_neg:.3f}**

                    🎯 Berdasarkan data diatas, rata-rata keyakinan model terhadap prediksi sentimen positif sebesar **{avg_pos:.3f}** dan sentimen negatif sebesar **{avg_neg:.3f}**.
                    Sebanyak **{high_conf_pct:.1f}%** prediksi memiliki skor keyakinan di atas **0.80**, yang menunjukkan bahwa model **{kualitas_model}** dalam melakukan klasifikasi sentimen pada ulasan aplikasi ini.
                    """)

            # # Baris 2: Chart Keluhan per Versi
            # st.subheader("Persentase Keluhan Berdasarkan Versi Aplikasi")
            # # --- PERBAIKAN 4: GUNAKAN LABEL TEKS YANG BENAR ---
            # # Kita filter dataframe hanya untuk sentimen Negatif
            # df_neg = df[df['sentiment'] == 'Negatif']
            
            # # --- PERBAIKAN 5: CEK KEBERADAAN DATA NEGATIF DENGAN BENAR ---
            # # Kita gunakan .empty, dan ini akan konsisten dengan pie chart
            # if not df_neg.empty:
            #     keluhan_versi = df_neg['reviewCreatedVersion'].value_counts().reset_index()
            #     keluhan_versi.columns = ['Versi', 'Jumlah Keluhan']
            #     # Tampilkan top 10 versi yang paling banyak dikeluhkan
            #     keluhan_fig = px.bar(keluhan_versi.head(10), x='Versi', y='Jumlah Keluhan', 
            #                          text='Jumlah Keluhan', labels={'Jumlah Keluhan':'Jumlah Keluhan'},
            #                          color_discrete_sequence=['#E74C3C'])
            #     st.plotly_chart(keluhan_fig, width="stretch")
            # else:
            #     st.info("Wah, hebat! Tidak ada keluhan (sentimen negatif) yang ditemukan pada sampel ini.")
            
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
            **Interpretasi Grafik Keluhan Berdasarkan Versi**

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
                

                # hanya ambil bigram yang muncul minimal 3 kali
                freq = {
                    k:v
                    for k,v in freq.items()
                    if v >= 3
                }

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
                    [" ".join(bg) for bg in ngrams(words, 4)]
                )

            top_poso = Counter(bigram_pos).most_common(10)

            # Negatif
            bigram_neg = []

            for text in df[df['sentiment']=='Negatif']['cleaned_content'].dropna():
                words = text.split()
                bigram_neg.extend(
                    [" ".join(bg) for bg in ngrams(words, 4)]
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