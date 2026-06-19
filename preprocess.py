import re
import unicodedata
from typing import List

_RE_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_RE_APOS = re.compile(r"[\u2018\u2019\u02BC]")
_RE_ZW = re.compile(r"[\u200B-\u200F\uFEFF\u2060-\u2063]")
_RE_URLS = re.compile(r'https?://\S+|www\.\S+', re.I)
_RE_MENTIONS = re.compile(r'(?<![A-Za-z0-9_])@[a-zA-Z0-9_-]+')
_RE_BRACKETS = re.compile(r'[\[\]\{\}\(\)【［〔｢\]\}\)】］〕｣]')
_RE_AT_INFIX = re.compile(r'(?i)(?<=[a-z])@(?=[a-z])')

HARD_SEPARATORS = r"/|\\:;~_.,\-!()\[\]{}<>=+\"'"

# 1. MAPPING ANGKA KE HURUF (Alay level standar)
LEET_MAP_TABLE = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '@': 'a'
}

# 2. KAMUS ALAY KHUSUS DATASET PLAYSTORE (Berdasarkan Ekstraksi Otomatis)
SUBS = {
    # 1. Negasi / Penolakan (Penting untuk Sentimen)
    'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
    'ngga': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'tak': 'tidak',
    'gabisa': 'tidak bisa', 'gajelas': 'tidak jelas', 'nga': 'tidak',
    'nggk': 'tidak', 'ngak': 'tidak', 'jgn': 'jangan', 'jngan': 'jangan',
    'ngk': 'tidak',

    # 2. Kata Ganti Orang
    'sy': 'saya', 'gw': 'saya', 'gua': 'saya', 'gue': 'saya', 'ku': 'aku', 'mu': 'kamu',
    'orng': 'orang', 'sya': 'saya', 'user': 'pengguna', 'driver': 'pengemudi',
    'CS': 'customer service','ojol': 'ojek online', 'ortu': 'orang tua', 'abg': 'abang',


    # 3. Kata Penghubung & Preposisi
    'yg': 'yang', 'kalo': 'kalau', 'klo': 'kalau', 'karna': 'karena',
    'krn': 'karena', 'krna': 'karena', 'tp': 'tapi', 'tpi': 'tapi',
    'dgn': 'dengan', 'utk': 'untuk', 'pdhl': 'padahal', 'jd': 'jadi',
    'jdi': 'jadi', 'jg': 'juga', 'lg': 'lagi', 'lgi': 'lagi', 'd': 'di', 'k': 'ke',
    'ad': 'ada', 'dr': 'dari', 'msh': 'masih', 'msih': 'masih', 'sm': 'sama',
    'sma': 'sama', 'tb': 'tiba', 'tba': 'tiba', 'kl': 'kalau', 'klw': 'kalau',
    'klu':'kalau', 'knp': 'kenapa', '&': 'dan', 'emg': 'memang', 'kdng': 'kadang',
    'bs': 'bisa', 'lbh': 'lebih', 'lbih': 'lebih', 'sllu': 'selalu', 'hrs': 'harus',

    # 4. Keterangan Waktu & Keadaan
    'udh': 'sudah', 'sdh': 'sudah', 'dah': 'sudah', 'blm': 'belum', 'skrng': 'sekarang',
    'skrg': 'sekarang', 'trs': 'terus', 'trus': 'terus', 'sampe': 'sampai', 'smpe': 'sampai',
    'bgt': 'banget', 'aja': 'saja', 'aj': 'saja', 'pas': 'saat', 'dulu': 'dahulu', 'bngt': 'banget',
    'knp': 'kenapa', 'gini': 'begini', 'gitu': 'begitu', 'gimana': 'bagaimana', 'thn': 'tahun',
    'kyk': 'seperti', 'nyampe': 'sampai', 'nyampek': 'sampai', 'getun': 'sebal', 'dlu': 'dulu',
    'nnti': 'nanti', 'kmren': 'kemarin', 'kemaren': 'kemarin', 'knpa': 'kenapa', 'blum': 'belum',

    # 5. Kata Kerja & Aktivitas (Sering Typo/Singkatan)
    'pake': 'pakai', 'bikin': 'buat', 'dapet': 'dapat', 'pesen': 'pesan',
    'nunggu': 'tunggu', 'liat': 'lihat', 'benerin': 'perbaiki',
    'donlod': 'unduh', 'updet': 'perbarui', 'tf': 'transfer', 'nyari': 'cari',
    'tlpon': 'telepon', 'tlp': 'telepon', 'tlphn': 'telepon', 'tlpnh': 'telepon',
    'tlpn': 'telepon', 'cht': 'pesan', 'chat': 'pesan', 'vid': 'video', 'ai': 'kecerdasan buatan',
    'ongkir': 'ongkos kirim', 'ongkirnya': 'ongkos kirimnya', 'onkir': 'ongkos kirim', 'notif': 'notifikasi',
    'verif': 'verifikasi', 'ferif': 'verifikasi', 'wjh': 'wajah', 'wjah': 'wajah', 'konfirm': 'konfirmasi',

    # 6. Istilah Aplikasi & Domain
    'apk': 'aplikasi', 'app': 'aplikasi', 'apps': 'aplikasi', 'apl': 'aplikasi',
    'eror': 'error', 'ngelag': 'lag', 'lemot': 'lambat', 'lelet': 'lambat',
    'muter': 'loading', 'cs': 'customer service', 'min': 'admin', 'rb': 'ribu',

    # 7. Kata Sifat & Ekspresi
    'bener': 'benar', 'kaya': 'seperti', 'kayak': 'seperti', 'kek': 'seperti', 'tlg': 'tolong',
    'cuman': 'cuma', 'cm': 'cuma', 'cma': 'cuma',  'doang': 'saja', 'kesel': 'kesal', 'ancur': 'hancur',
    'jelek': 'buruk', 'kren': 'keren', 'males': 'malas', 'bgs': 'bagus', 'bgus': 'bagus',
    'b': 'biasa', 'ilang': 'hilang', 'sapa': 'siapa', 'bnyk': 'banyak', 'makasih': 'terima kasih',

    #  8. typo
    'skolah': 'sekolah', 'ngasih': 'memberikan', 'msen': 'pesan', 'nganter': 'mengantar', 'smping': 'samping',
    'gaenak': 'tidak enak', 'tmn': 'teman', 'nmr': 'nomor', 'nomer': 'nomor', 'thx': 'terima kasih', 'makasih': 'terima kasih',
    'mkasih': 'terima kasih', 'hp': 'ponsel', 'tuk': 'untuk', 'no': 'nomor', 'norek': 'nomor rekening', 'n': 'dan', 'lbih': 'lebih',
    'nyoba': 'coba', 'dg': 'dengan', 'sperti': 'seperti', 'drpd': 'daripada', 'apknya': 'aplikasi nya', 'tsb': 'tersebut', 'tgl': 'tanggal',
    'msh': 'masih', 'gtu': 'gitu', 'ttap': 'tetap', 'pke': 'pakai', 'pake': 'pakai', 'y': 'ya', 'cape': 'lelah', 'capek': 'lelah', 'sllu': 'selalu',
    'hrga': 'harga', 'udah': 'sudah', 'kya': 'seperti', 'aga': 'agak', 'gaada': 'tidak ada', 'gmbr': 'gambar','ngebug': 'bug', 'bugnya': 'bug', 'lemot': 'lambat',
    'ngelag': 'lag', 'lagnya': 'lag', 'dsbt': 'dan sebagainya', 'dsb': 'dan sebagainya', 'tmbh': 'tambah', 'beberp': 'beberapa', 'bbrp': 'beberapa',
    'kntong': 'kantong', 'tq': 'terima kasih', 'dtg': 'datang', 'brg': 'barang', 'cmn': 'cuman', 'yng': 'yang', 'mesan': 'pesan', 'slomo': 'slow motion',
    'mo': 'mau', 'appnya': 'aplikasi nya', 'dri': 'dari', 'naro': 'taruh', 'gada': 'tidak ada', 'vol': 'volume', 'bkn': 'bukan', 'bgian': 'bagian',
    'jumlh': 'jumlah', 'bsa': 'bisa', 'bru': 'baru', 'ngelek': 'nge lag', 'mabar': 'main bareng', 'temen': 'teman', 'ok': 'oke', 'makasi': 'terima kasih',
    'malem': 'malam', 'serem': 'seram', 'pkek': 'pakai', 'bngett': 'banget', 'jwbannya': 'jawaban nya', 'kumplit': 'komplet', 'mlh': 'malah', 'tokped': 'tokopedia',
    'dll': 'dan lain-lain', 'dpat': 'dapat', 'lngsng': 'langsung', 'dpat': 'dapat', 'lngsng': 'langsung', 'moga': 'semoga', 'gj': 'tidak jelas', 'gaje': 'tidak jelas',
    'mkan': 'makan', 'makn': 'makan', 'pdhal': 'padahal', 'ndak': 'tidak', 'bbrpa': 'beberapa', 'bgtt': 'bangett', 'ama': 'sama', 'dpt': 'dapat', 'sndiri': 'sendiri',
    'byk': 'banyak', 'ttp': 'tetap', 'sukak': 'suka', 'dech': 'dehh', 'aplgi': 'apalagi', 'tmbhn': 'tambahan', 'indo': 'indonesia', 'ttg': 'tentang', 'gtw': 'tidak tahu',
    'ntar': 'nanti', 'masi': 'masih', 'ampe': 'sampai', 'gjls': 'tidak jelas', 'dikit': 'sedikit', 'ngirim': 'kirim', 'poto': 'foto', 'kdang': 'kadang', 'dkit': 'sedikit',
    'vitur': 'fitur', 'makin': 'semakin',

}

def normalize_chars(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

def strip_symbol_chars(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(('S', 'C')))

def handle_intraword_symbols(text: str) -> str:
    text = re.sub(rf"[{HARD_SEPARATORS}]+", " ", text)
    text = re.sub(r"(?<=\w)[^\w\s]+(?=\w)", "", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    return text.strip()

def squeeze_repeats(token: str, max_repeat: int = 2) -> str:
    """Mengurangi karakter berulang: 'baguuuus' jadi 'baguus'"""
    out, cnt, prev = [], 0, ''
    for ch in token:
        if ch == prev:
            cnt += 1
            if cnt <= max_repeat or ch.isdigit():
                out.append(ch)
        else:
            prev = ch
            cnt = 1
            out.append(ch)
    return ''.join(out)

def normalize_plesetan(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if t.isdigit():
            out.append(t)
            continue
        # Cek kamus pertama
        if t in SUBS:
            out.append(SUBS[t])
            continue

        # Map leet speak (misal: b4gus -> bagus)
        t = ''.join(LEET_MAP_TABLE.get(ch, ch) for ch in _RE_AT_INFIX.sub('a', t))
        t = squeeze_repeats(t)
        # Cek kamus lagi setelah leet dibersihkan
        t = SUBS.get(t, t)

        out.append(t)
    return out

def preprocess_playstore(text: str) -> str:
    """
    Pipeline Utama: Pembersihan -> Lowercase -> Kamus Alay (Tanpa Stopword/Stemming)
    """
    if not isinstance(text, str) or not text:
        return ""
    
    text = unicodedata.normalize("NFKD", text)

    # Mengubah lagu2, lagu², lagu", lagu' menjadi lagu-lagu
    text = re.sub(r'\b([a-zA-Z]+)(?:2|²|"|\')', r'\1-\1', text)
    text = re.sub(r'\b([0-9]+)(?:x|X)', r'\1 kali', text)

    # 1. Cleaning Tanda Baca, URL, Emoji
    text = _RE_URLS.sub(" ", text)
    text = _RE_MENTIONS.sub(" ", text)
    text = _RE_BRACKETS.sub("", text)
    text = _RE_DASHES.sub("-", text)
    text = _RE_APOS.sub("'", text)
    text = _RE_ZW.sub("", text)
    
    # Membuang aksen/Mn jika ada yang tersisa
    text = normalize_chars(text)
    text = handle_intraword_symbols(text)
    text = strip_symbol_chars(text)

    # 2. Case Folding
    text = text.lower()

    # 3. Sisakan Huruf & Angka, lalu rapihkan spasi
    text = re.sub(r"[^0-9a-z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Normalisasi Plesetan / Kamus Alay
    tokens = text.split()
    tokens = normalize_plesetan(tokens)

    return " ".join(tokens)
