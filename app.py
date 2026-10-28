import os
import base64
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

st.set_page_config(page_title="LAKSAMANA | Analisis Pangan & Iklim", page_icon=(os.path.join(os.path.dirname(__file__), "logo.png") if os.path.exists(os.path.join(os.path.dirname(__file__), "logo.png")) else None), layout="wide")


# ---- Tampilan (hanya gaya visual, tidak memengaruhi data atau model) ----
NAVY, BLUE, GREEN = "#0F3D5E", "#1D5C8D", "#2E9E83"
_LOGO_PNG = os.path.join(os.path.dirname(__file__), "logo.png")
_LOGO_SVG = (
    '<svg viewBox="0 0 64 64" width="60" height="60" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="32" cy="32" r="30" fill="none" stroke="#FFFFFF" stroke-width="2"/>'
    '<circle cx="32" cy="15" r="4" fill="none" stroke="#FFFFFF" stroke-width="3"/>'
    '<path d="M32 19V49M23 26H41M15 37c1 9 8 14 17 14s16-5 17-14M12 41l3-5 5 3M52 41l-3-5-5 3" fill="none" '
    'stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M8 57c6-4 10 4 16 0s10 4 16 0 10 4 16 0" fill="none" stroke="#2E9E83" stroke-width="3" stroke-linecap="round"/>'
    '</svg>'
)
if os.path.exists(_LOGO_PNG):
    with open(_LOGO_PNG, "rb") as _f:
        _LOGO_HTML = '<img src="data:image/png;base64,' + base64.b64encode(_f.read()).decode() + '" style="height:60px">'
else:
    _LOGO_HTML = _LOGO_SVG

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; max-width: 1180px; }
.lk-hero { background: #0F3D5E; border-bottom: 4px solid #2E9E83; border-radius: 4px; padding: 20px 26px; margin-bottom: 1.4rem;
           display: flex; align-items: center; gap: 20px; }
.lk-hero .lk-title { font-size: 2rem; font-weight: 700; letter-spacing: .06em; color: #FFFFFF; line-height: 1.1; }
.lk-hero .lk-sub { font-size: 1.05rem; color: #BFE3D8; margin-top: 2px; }
.lk-hero .lk-desc { font-size: .9rem; color: #D5E3EC; margin-top: 8px; max-width: 900px; line-height: 1.45; }
h2, h3 { font-size: 1.3rem; font-weight: 600; color: #0F3D5E; border-bottom: 2px solid #2E9E83; padding-bottom: .3rem; }
h4 { font-size: 1.05rem; font-weight: 600; color: #1D5C8D; margin-top: 1.4rem; }
h5 { font-size: .95rem; font-weight: 600; color: #3E4853; }
[data-testid="stMetric"] { border: 1px solid #D5E0E8; border-left: 4px solid #2E9E83; border-radius: 4px; padding: 12px 14px; background: #FFFFFF; }
[data-testid="stMetricLabel"] { color: #5B6670; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; color: #0F3D5E; }
section[data-testid="stSidebar"] { background: #EAF1F6; border-right: 1px solid #D0DDE7; }
section[data-testid="stSidebar"] label p { color: #0F3D5E; }
[data-testid="stAlert"] { border-radius: 4px; }
[data-testid="stDataFrame"] { border: 1px solid #D5E0E8; border-radius: 4px; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: #5B7488; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; counter-reset: nav; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { counter-increment: nav; width: 100%; margin: 0; padding: 10px 12px; border-radius: 4px;
    border-left: 4px solid transparent; cursor: pointer; align-items: center; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label::before { content: counter(nav, decimal-leading-zero); font-size: .75rem; font-weight: 600; color: #7C93A5; margin-right: 12px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #DCE8F0; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background: #FFFFFF; border-left-color: #2E9E83; box-shadow: 0 1px 3px rgba(15,61,94,.15); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: #0F3D5E; font-weight: 600; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::before { color: #2E9E83; }
.lk-brand { display: flex; align-items: center; gap: 12px; padding-bottom: 14px; margin-bottom: 14px; border-bottom: 1px solid #C9D9E4; }
.lk-brand .t { font-weight: 700; letter-spacing: .08em; color: #0F3D5E; line-height: 1.1; }
.lk-brand .s { font-size: .78rem; color: #5B7488; }
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] { display: flex; flex-direction: column; gap: 4px; counter-reset: nav; }
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div { counter-increment: nav; width: 100%; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] { width: 100%; box-sizing: border-box; display: flex; align-items: center; padding: 10px 12px;
    border-radius: 4px; border-left: 4px solid transparent; cursor: pointer; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display: none; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"]::before { content: counter(nav, decimal-leading-zero); font-size: .75rem; font-weight: 600; color: #7C93A5; margin-right: 12px; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: #DCE8F0; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] { background: #FFFFFF; border-left-color: #2E9E83; box-shadow: 0 1px 3px rgba(15,61,94,.15); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { color: #0F3D5E; font-weight: 600; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before { color: #2E9E83; }
hr { margin: 1.2rem 0; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.figsize": (6, 3.6), "figure.dpi": 110, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9AA3AD",
    "axes.labelcolor": "#3E4853", "xtick.color": "#5B6670", "ytick.color": "#5B6670",
    "axes.grid": True, "grid.color": "#E3E8EC", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=[BLUE, GREEN, "#C0643F", "#8A94A0"]),
})

DATA_FILE = os.path.join(os.path.dirname(__file__), "Dataset_Pangan_Iklim_Cleaned_Normalized.xlsx")

st.markdown(
    '<div class="lk-hero">' + _LOGO_HTML + '<div>'
    '<div class="lk-title">LAKSAMANA</div>'
    '<div class="lk-sub">Sistem Analisis Hubungan Iklim dan Ketersediaan Pangan</div>'
    '<div class="lk-desc">Aplikasi prototipe yang mempelajari pola dari dataset pangan dan iklim Tanjungpinang, lalu mengestimasi surplus/defisit berdasarkan komoditas, periode, dan indikator iklim.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

@st.cache_data
def load_data(path):
    xl = pd.ExcelFile(path)
    food = pd.read_excel(path, sheet_name="2_Cleaning_Pangan")
    climate = pd.read_excel(path, sheet_name="2_Cleaning_Iklim")

    food["periode_iklim"] = food["periode_iklim"].astype(str).str.strip()
    climate["periode"] = climate["periode"].astype(str).str.strip()

    for c in ["pasokan_total", "kebutuhan", "surplus_hitung"]:
        food[c] = pd.to_numeric(food[c], errors="coerce")
    for c in ["curah_hujan_mm", "kecepatan_angin_maks_ms", "kecepatan_angin_rata_ms"]:
        climate[c] = pd.to_numeric(climate[c], errors="coerce")

    climate_for_merge = climate.rename(columns={"periode": "periode_iklim"})
    merged = food.merge(
        climate_for_merge[["periode_iklim", "curah_hujan_mm", "kecepatan_angin_maks_ms", "kecepatan_angin_rata_ms"]],
        on="periode_iklim", how="left"
    )
    merged["tanggal_mulai"] = pd.to_datetime(merged["tanggal_mulai"], errors="coerce")
    merged["tahun_bulan"] = merged["periode_iklim"]
    merged["bulan_angka"] = pd.to_datetime(merged["periode_iklim"] + "-01", errors="coerce").dt.month

    # Aggregate weekly rows to commodity-month to reduce repeated climate rows.
    monthly = merged.dropna(subset=["periode_iklim", "komoditas", "surplus_hitung"]).groupby(
        ["periode_iklim", "komoditas"], as_index=False
    ).agg(
        surplus_bulanan=("surplus_hitung", "sum"),
        pasokan_bulanan=("pasokan_total", "sum"),
        kebutuhan_bulanan=("kebutuhan", "sum"),
        curah_hujan_mm=("curah_hujan_mm", "first"),
        kecepatan_angin_maks_ms=("kecepatan_angin_maks_ms", "first"),
        kecepatan_angin_rata_ms=("kecepatan_angin_rata_ms", "first"),
        bulan_angka=("bulan_angka", "first"),
        jumlah_minggu=("surplus_hitung", "count"),
    )
    monthly = monthly.dropna(subset=["curah_hujan_mm", "kecepatan_angin_maks_ms", "kecepatan_angin_rata_ms"])
    return food, climate, merged, monthly

try:
    food, climate, merged, monthly = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Dataset tidak dapat dibaca: {e}")
    st.stop()

if monthly.empty:
    st.error("Tidak ditemukan baris yang memiliki pasangan data pangan dan iklim.")
    st.stop()

# Sidebar navigation
_LOGO_SIDE = _LOGO_HTML.replace('height:60px', 'height:38px') if os.path.exists(_LOGO_PNG) else (
    _LOGO_SVG.replace('width="60" height="60"', 'width="38" height="38"').replace('#FFFFFF', '#0F3D5E')
)
st.sidebar.markdown(
    '<div class="lk-brand">' + _LOGO_SIDE + '<div><div class="t">LAKSAMANA</div><div class="s">Analisis pangan dan iklim</div></div></div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Menu", ["Ringkasan Data", "Analisis Hubungan Iklim", "Prediksi & Risiko ML", "Ringkasan Hasil", "Catatan Metodologi"])

if page == "Ringkasan Data":
    st.markdown("### Ringkasan dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris data pangan", f"{len(food):,}")
    c2.metric("Bulan iklim", f"{climate['periode'].nunique():,}")
    c3.metric("Komoditas", f"{food['komoditas'].nunique():,}")
    c4.metric("Data komoditas-bulan berpasangan", f"{len(monthly):,}")
    st.info("Aplikasi memakai data hasil cleaning sebagai sumber pembelajaran. Data mentah tetap disimpan di file Excel untuk pelacakan dan pemeriksaan.")
    st.markdown("#### Cakupan data yang digunakan")
    st.write(f"Periode iklim: **{climate['periode'].min()} sampai {climate['periode'].max()}**.")
    st.write(f"Periode yang dapat digunakan untuk model: **{monthly['periode_iklim'].min()} sampai {monthly['periode_iklim'].max()}**.")
    st.markdown("#### Jumlah baris per komoditas")
    counts = monthly.groupby("komoditas").size().sort_values(ascending=False).rename("Jumlah periode")
    st.dataframe(counts.reset_index(), width="stretch", hide_index=True)
    st.markdown("#### Pratinjau data hasil penggabungan")
    st.dataframe(monthly.head(30), width="stretch", hide_index=True)

elif page == "Ringkasan Hasil":
    st.markdown("### Ringkasan hasil analisis")
    st.write(
        "Bagian ini merangkum hasil yang dihitung langsung dari dataset dan menyediakan "
        "ringkasan yang dapat diunduh untuk bahan laporan."
    )

    climate_cols = [
        "curah_hujan_mm",
        "kecepatan_angin_maks_ms",
        "kecepatan_angin_rata_ms",
    ]
    target_cols = ["surplus_bulanan", "pasokan_bulanan", "kebutuhan_bulanan"]
    climate_labels = {
        "curah_hujan_mm": "Curah hujan",
        "kecepatan_angin_maks_ms": "Angin maksimum",
        "kecepatan_angin_rata_ms": "Angin rata-rata",
    }
    target_labels = {
        "surplus_bulanan": "Surplus bulanan",
        "pasokan_bulanan": "Pasokan bulanan",
        "kebutuhan_bulanan": "Kebutuhan bulanan",
    }
    climate_month_count = climate["periode"].nunique()
    correlation_matrix = monthly[climate_cols + target_cols].corr(method="spearman")
    overall_correlations = correlation_matrix.loc[climate_cols, target_cols].rename(
        index=climate_labels, columns=target_labels
    )
    surplus_correlations = overall_correlations["Surplus bulanan"]
    strongest_overall_indicator, strongest_overall_value = max(
        surplus_correlations.items(), key=lambda item: abs(item[1])
    )
    if surplus_correlations.abs().max() < 0.1:
        overall_conclusion = (
            "Ketiga korelasi gabungan antara indikator iklim dan surplus berada dekat nol. "
            "Dalam dataset ini belum terlihat hubungan gabungan yang kuat dan konsisten."
        )
    else:
        overall_conclusion = (
            f"Korelasi gabungan terbesar antara indikator iklim dan surplus adalah "
            f"{strongest_overall_indicator.lower()} (Spearman {strongest_overall_value:+.3f}). "
            "Angka korelasi saja tidak membuktikan hubungan sebab-akibat."
        )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Baris komoditas-bulan", f"{len(monthly):,}")
    summary_cols[1].metric("Komoditas", f"{monthly['komoditas'].nunique():,}")
    summary_cols[2].metric("Bulan pengamatan", f"{monthly['periode_iklim'].nunique():,}")
    deficit_count = int((monthly["surplus_bulanan"] < 0).sum())
    summary_cols[3].metric(
        "Baris komoditas-bulan defisit",
        f"{deficit_count:,} ({deficit_count / len(monthly):.1%})",
    )
    st.caption(
        f"Periode yang dianalisis: {monthly['periode_iklim'].min()} sampai "
        f"{monthly['periode_iklim'].max()}. Data pangan mingguan telah dijumlahkan "
        "menjadi data komoditas-bulan."
    )

    st.markdown("#### Hasil hubungan iklim dengan pangan")
    st.write(
        "Korelasi Spearman menunjukkan apakah dua hal cenderung bergerak searah atau "
        "berlawanan. Angka dekat 0 berarti pola hubungan yang terlihat lemah."
    )
    st.dataframe(
        overall_correlations.round(3),
        width="stretch",
    )

    commodity_rows = []
    for name, group in monthly.groupby("komoditas"):
        if len(group) >= 5:
            commodity_rows.append({
                "Komoditas": name,
                "Jumlah bulan": len(group),
                "Hujan dengan surplus": group["curah_hujan_mm"].corr(
                    group["surplus_bulanan"], method="spearman"
                ),
                "Angin maksimum dengan surplus": group["kecepatan_angin_maks_ms"].corr(
                    group["surplus_bulanan"], method="spearman"
                ),
            })
    commodity_correlations = pd.DataFrame(commodity_rows)
    correlation_candidates = [
        (row["Komoditas"], label, row[column])
        for _, row in commodity_correlations.iterrows()
        for column, label in [
            ("Hujan dengan surplus", "curah hujan"),
            ("Angin maksimum dengan surplus", "angin maksimum"),
        ]
        if pd.notna(row[column])
    ]
    if correlation_candidates:
        strongest_name, strongest_indicator, strongest_value = max(
            correlation_candidates, key=lambda item: abs(item[2])
        )
        st.info(
            f"Hubungan terbesar secara angka pada tabel per komoditas adalah "
            f"{strongest_indicator} dengan surplus {strongest_name} "
            f"(Spearman {strongest_value:+.3f}). Ini hanya pola pada data yang tersedia, "
            "bukan bukti bahwa indikator tersebut menyebabkan surplus berubah."
        )
    st.dataframe(
        commodity_correlations.round(3),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Indikator iklim yang sama dipakai untuk beberapa komoditas pada bulan yang sama. "
        f"Jadi {len(monthly)} baris komoditas-bulan tidak berarti ada {len(monthly)} "
        "kondisi iklim yang berbeda; "
        f"data iklim hanya mencakup {climate_month_count} bulan."
    )

    st.markdown("#### Hasil pengujian model")
    st.write(
        "Tabel berikut memakai pembagian waktu 80% periode awal untuk latihan dan 20% "
        "periode terbaru untuk pengujian."
    )
    st.caption(
        "KNN mencari contoh historis yang kondisinya mirip untuk mengelompokkan defisit "
        "atau surplus. Regresi Linier memakai pola historis untuk memperkirakan angka "
        "surplus dan kebutuhan."
    )
    features = [
        "komoditas",
        "bulan_angka",
        "curah_hujan_mm",
        "kecepatan_angin_maks_ms",
        "kecepatan_angin_rata_ms",
    ]
    regression_targets = ["surplus_bulanan", "kebutuhan_bulanan"]
    model_data = monthly.dropna(
        subset=features + regression_targets
    ).sort_values("periode_iklim").copy()
    model_periods = sorted(model_data["periode_iklim"].unique())
    model_results = []
    knn_summary = None
    test_periods = set()

    if len(model_periods) >= 6:
        split_idx = min(len(model_periods) - 1, max(1, int(len(model_periods) * 0.8)))
        train_periods = set(model_periods[:split_idx])
        test_periods = set(model_periods[split_idx:])
        train = model_data[model_data["periode_iklim"].isin(train_periods)].copy()
        test = model_data[model_data["periode_iklim"].isin(test_periods)].copy()

        def make_report_preprocessor():
            return ColumnTransformer([
                ("komoditas", OneHotEncoder(handle_unknown="ignore"), ["komoditas"]),
                ("angka", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), features[1:]),
            ])

        risk_train = train.dropna(subset=["surplus_bulanan"]).copy()
        risk_test = test.dropna(subset=["surplus_bulanan"]).copy()
        if not risk_train.empty and not risk_test.empty:
            risk_train["risiko_defisit"] = risk_train["surplus_bulanan"] < 0
            risk_test["risiko_defisit"] = risk_test["surplus_bulanan"] < 0
            report_risk_model = Pipeline([
                ("preprocess", make_report_preprocessor()),
                ("model", KNeighborsClassifier(
                    n_neighbors=min(5, len(risk_train)),
                    weights="distance",
                )),
            ])
            report_risk_model.fit(risk_train[features], risk_train["risiko_defisit"])
            risk_predictions = report_risk_model.predict(risk_test[features])
            knn_summary = {
                "balanced_accuracy": balanced_accuracy_score(
                    risk_test["risiko_defisit"], risk_predictions
                ),
                "f1": f1_score(
                    risk_test["risiko_defisit"], risk_predictions, zero_division=0
                ),
                "test_rows": len(risk_test),
                "test_deficits": int(risk_test["risiko_defisit"].sum()),
            }

        for target in regression_targets:
            target_train = train.dropna(subset=[target])
            target_test = test.dropna(subset=[target])
            if target_train.empty or target_test.empty:
                continue
            report_regressor = Pipeline([
                ("preprocess", make_report_preprocessor()),
                ("model", LinearRegression()),
            ])
            report_regressor.fit(target_train[features], target_train[target])
            predictions = report_regressor.predict(target_test[features])
            baseline_predictions = [target_train[target].mean()] * len(target_test)
            for model_name, values in [
                ("Regresi Linier", predictions),
                ("Tebakan rata-rata data latihan", baseline_predictions),
            ]:
                model_results.append({
                    "Target": target_labels[target],
                    "Metode": model_name,
                    "MAE (lebih kecil lebih baik)": mean_absolute_error(
                        target_test[target], values
                    ),
                    "RMSE (lebih kecil lebih baik)": mean_squared_error(
                        target_test[target], values
                    ) ** 0.5,
                    "R² (lebih besar umumnya lebih baik)": (
                        r2_score(target_test[target], values)
                        if len(target_test) > 1
                        else float("nan")
                    ),
                })

        model_results_df = pd.DataFrame(model_results)
        if not model_results_df.empty:
            st.dataframe(
                model_results_df.round(3),
                width="stretch",
                hide_index=True,
            )
        if knn_summary:
            knn_cols = st.columns(2)
            knn_cols[0].metric(
                "Ketepatan seimbang KNN",
                f"{knn_summary['balanced_accuracy']:.1%}",
            )
            knn_cols[1].metric("F1 untuk kelas defisit", f"{knn_summary['f1']:.1%}")
        st.caption(
            f"Data latihan: {len(train_periods)} bulan ({len(train)} baris); data uji: "
            f"{len(test_periods)} bulan ({len(test)} baris), dari {min(test_periods)} "
            f"sampai {max(test_periods)}. MAE dan RMSE adalah besar kesalahan dalam "
            "satuan target; bandingkan dengan metode rata-rata sebagai patokan sederhana."
        )
        if knn_summary:
            st.caption(
                f"Pengujian KNN memakai {knn_summary['test_rows']} baris uji dan "
                f"{knn_summary['test_deficits']} di antaranya berlabel defisit. "
                "Skor tinggi pada periode uji yang singkat tidak menjamin hasil akan sama "
                "pada periode lain."
            )
    else:
        st.warning(
            "Periode data belum cukup untuk menampilkan evaluasi model berbasis waktu."
        )

    st.markdown("#### Kesimpulan yang dapat ditulis")
    st.info(
        f"{overall_conclusion} Ketika dilihat per komoditas, polanya berbeda. "
        f"Karena data iklim hanya tersedia untuk {climate_month_count} bulan dan hubungan statistik tidak "
        "membuktikan sebab-akibat, hasil ini belum cukup untuk menyimpulkan bahwa iklim "
        "memengaruhi atau menyebabkan perubahan ketersediaan komoditas."
    )
    st.markdown("#### Batasan hasil")
    st.markdown(f"""
- Data iklim hanya mencakup {climate_month_count} bulan; jumlah kondisi iklim yang diamati terbatas.
- KNN mempelajari label surplus/defisit dari catatan historis, bukan kejadian gangguan pelayaran.
- Periode uji terdiri dari {len(test_periods)} bulan pada pembagian 80/20; nilai evaluasi dapat berubah saat periode data bertambah.
- Nilai MAE/RMSE gabungan perlu dibaca hati-hati jika satuan antar-komoditas berbeda.
- Dataset tidak berisi catatan gangguan logistik atau jumlah stok cadangan aktual.
- Korelasi dan model menunjukkan pola pada dataset, bukan bukti sebab-akibat atau jaminan prediksi masa depan.
""")

    def report_table(dataframe):
        headers = [str(column) for column in dataframe.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for values in dataframe.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in values) + " |")
        return "\n".join(lines)

    report_lines = [
        "# Ringkasan hasil analisis LAKSAMANA",
        "",
        "## Tujuan",
        "Menganalisis hubungan antara indikator iklim dan surplus pangan, serta mengevaluasi "
        "model yang memperkirakan surplus/kebutuhan dan mengelompokkan defisit historis.",
        "",
        "## Data",
        "- Sumber: Dataset_Pangan_Iklim_Cleaned_Normalized.xlsx, sheet 2_Cleaning_Pangan dan 2_Cleaning_Iklim.",
        f"- Baris komoditas-bulan: {len(monthly)}",
        f"- Komoditas: {monthly['komoditas'].nunique()}",
        f"- Periode pangan: {monthly['periode_iklim'].min()} sampai {monthly['periode_iklim'].max()} "
        f"({monthly['periode_iklim'].nunique()} bulan); data iklim mencakup {climate_month_count} bulan",
        f"- Baris berlabel defisit (surplus < 0): {deficit_count} "
        f"({deficit_count / len(monthly):.1%})",
        "- Data mingguan diringkas menjadi total komoditas-bulan, lalu dipasangkan dengan data iklim.",
        "- Fitur model: komoditas, bulan, curah hujan, kecepatan angin maksimum, dan kecepatan angin rata-rata.",
        "- KNN menganggap surplus < 0 sebagai defisit; Regresi Linier memperkirakan surplus dan kebutuhan.",
        "",
        "## Metode",
        "- Korelasi Spearman membandingkan pola naik-turun indikator iklim dan pangan; korelasi tidak membuktikan sebab-akibat.",
        "- Data dibagi berdasarkan urutan waktu: 80% periode awal untuk latihan dan 20% periode terbaru untuk pengujian.",
        "- KNN mengelompokkan kondisi berdasarkan contoh historis terdekat; Regresi Linier memperkirakan nilai target dari pola data latihan.",
        "- Hasil Regresi Linier dibandingkan dengan patokan sederhana yang selalu menebak rata-rata data latihan.",
        "",
        "## Korelasi Spearman gabungan",
        "| Indikator iklim | Surplus bulanan | Pasokan bulanan | Kebutuhan bulanan |",
        "|---|---:|---:|---:|",
    ]
    for climate_key in climate_cols:
        values = overall_correlations.loc[climate_labels[climate_key]]
        report_lines.append(
            f"| {climate_labels[climate_key]} | {values[target_labels['surplus_bulanan']]:.3f} "
            f"| {values[target_labels['pasokan_bulanan']]:.3f} "
            f"| {values[target_labels['kebutuhan_bulanan']]:.3f} |"
        )
    report_lines.extend([
        "",
        "## Korelasi surplus per komoditas",
        report_table(commodity_correlations.round(3)),
        "",
        "## Evaluasi model",
    ])
    if len(model_periods) >= 6:
        report_lines.append(
            f"Pembagian kronologis 80/20: latihan {len(train_periods)} bulan "
            f"({len(train)} baris), uji {len(test_periods)} bulan "
            f"({len(test)} baris; {min(test_periods)} sampai {max(test_periods)})."
        )
        if knn_summary:
            report_lines.append(
                f"- KNN: ketepatan seimbang {knn_summary['balanced_accuracy']:.1%}; "
                f"F1 defisit {knn_summary['f1']:.1%}; data uji "
                f"{knn_summary['test_rows']} baris dengan "
                f"{knn_summary['test_deficits']} label defisit."
            )
        if model_results:
            report_lines.extend([
                "",
                report_table(pd.DataFrame(model_results).round(3)),
            ])
    report_lines.extend([
        "",
        "## Kesimpulan",
        f"{overall_conclusion} Pola tiap komoditas berbeda. "
        "Hasil ini menunjukkan hubungan/pola dalam data, tetapi tidak membuktikan bahwa iklim "
        "menyebabkan perubahan surplus atau ketersediaan pangan.",
        "",
        "## Keterbatasan",
        f"- Data iklim hanya mencakup {climate_month_count} bulan dan indikator iklim berulang pada komoditas di bulan yang sama.",
        "- Evaluasi model memakai periode uji yang singkat; skor bukan jaminan performa masa depan.",
        "- Nilai MAE/RMSE gabungan perlu dibaca hati-hati jika satuan antar-komoditas berbeda.",
        "- Dataset tidak berisi catatan gangguan logistik atau stok cadangan aktual.",
        "- Simulasi buffer adalah perkiraan sederhana dan bukan standar operasional resmi.",
        "",
    ])
    st.download_button(
        "Unduh ringkasan hasil (.md)",
        data="\n".join(report_lines),
        file_name="Ringkasan_Hasil_LAKSAMANA.md",
        mime="text/markdown",
    )

elif page == "Analisis Hubungan Iklim":
    st.markdown("### Apakah indikator iklim berhubungan dengan surplus/defisit pangan?")
    st.write("Analisis dilakukan pada data agregat komoditas-bulan. Korelasi menunjukkan hubungan, bukan bukti sebab-akibat.")
    commodity_options = ["Semua komoditas"] + sorted(monthly["komoditas"].dropna().unique().tolist())
    selected = st.selectbox("Pilih komoditas", commodity_options)
    d = monthly.copy() if selected == "Semua komoditas" else monthly[monthly["komoditas"] == selected].copy()

    climate_cols = ["curah_hujan_mm", "kecepatan_angin_maks_ms", "kecepatan_angin_rata_ms"]
    target_cols = ["surplus_bulanan", "pasokan_bulanan", "kebutuhan_bulanan"]
    corr_data = d[climate_cols + target_cols].corr(method="spearman")
    st.markdown("#### Korelasi Spearman")
    st.caption("Nilai mendekati +1 = hubungan searah kuat; mendekati -1 = berlawanan kuat; mendekati 0 = hubungan monotonik lemah.")
    st.dataframe(corr_data.loc[climate_cols, target_cols].round(3), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Curah hujan vs surplus/defisit")
        fig, ax = plt.subplots()
        ax.scatter(d["curah_hujan_mm"], d["surplus_bulanan"], alpha=0.7)
        ax.set_xlabel("Curah hujan (mm)")
        ax.set_ylabel("Surplus/defisit bulanan (satuan mengikuti sumber)")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown("#### Kecepatan angin maksimum vs surplus/defisit")
        fig, ax = plt.subplots()
        ax.scatter(d["kecepatan_angin_maks_ms"], d["surplus_bulanan"], alpha=0.7)
        ax.set_xlabel("Kecepatan angin maksimum (m/s)")
        ax.set_ylabel("Surplus/defisit bulanan (satuan mengikuti sumber)")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Pola pangan dan iklim dari waktu ke waktu")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("##### Rata-rata surplus per komoditas")
        commodity_surplus = (
            monthly.groupby("komoditas")["surplus_bulanan"]
            .mean()
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(8, max(4, len(commodity_surplus) * 0.32)))
        commodity_surplus.plot.barh(ax=ax, color=BLUE)
        ax.set_xlabel("Rata-rata surplus bulanan (satuan sumber)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Grafik ini membandingkan rata-rata historis per komoditas. Pastikan satuan tiap komoditas sebanding sebelum membandingkan besar angkanya.")
    with chart_right:
        st.markdown("##### Tren indikator iklim bulanan")
        climate_trend = climate.copy()
        climate_trend["periode"] = climate_trend["periode"].astype(str)
        climate_trend = climate_trend.sort_values("periode")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(climate_trend["periode"], climate_trend["curah_hujan_mm"], marker="o", label="Curah hujan (mm)")
        ax.set_ylabel("Curah hujan (mm)")
        ax.set_xlabel("Periode")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(climate_trend["periode"], climate_trend["kecepatan_angin_maks_ms"], marker="o", label="Angin maksimum (m/s)", color="#C0643F")
        ax.plot(climate_trend["periode"], climate_trend["kecepatan_angin_rata_ms"], marker="o", label="Angin rata-rata (m/s)", color=GREEN)
        ax.set_ylabel("Kecepatan angin (m/s)")
        ax.set_xlabel("Periode")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    if selected != "Semua komoditas":
        st.markdown(f"#### Tren surplus bulanan: {selected}")
        commodity_trend = monthly[monthly["komoditas"] == selected].sort_values("periode_iklim")
        st.line_chart(
            commodity_trend.set_index("periode_iklim")["surplus_bulanan"],
            x_label="Periode",
            y_label="Surplus bulanan (satuan sumber)",
        )

    st.markdown("#### Korelasi per komoditas")
    rows = []
    for name, group in monthly.groupby("komoditas"):
        if len(group) >= 5:
            rows.append({
                "Komoditas": name,
                "Jumlah periode": len(group),
                "Korelasi hujan-surplus": group["curah_hujan_mm"].corr(group["surplus_bulanan"], method="spearman"),
                "Korelasi angin maks-surplus": group["kecepatan_angin_maks_ms"].corr(group["surplus_bulanan"], method="spearman"),
            })
    corr_by_com = pd.DataFrame(rows)
    if not corr_by_com.empty:
        st.dataframe(corr_by_com.round(3).sort_values("Korelasi hujan-surplus", ascending=False), width="stretch", hide_index=True)
    st.warning("Data iklim hanya mencakup sekitar 24 bulan. Nilai korelasi perlu dibaca hati-hati dan tidak cukup untuk menyatakan bahwa iklim menyebabkan perubahan pangan.")

elif page == "Prediksi & Risiko ML":
    st.markdown("### Klasifikasi risiko dan estimasi berbasis machine learning")
    st.write("KNN dan Regresi Linier dilatih dari data historis komoditas-bulan yang telah dipasangkan dengan data iklim.")
    st.warning("Dataset belum mencatat gangguan pelayaran atau stok cadangan aktual. Risiko di bawah ini berarti risiko defisit berdasarkan pola historis, bukan prediksi kejadian gangguan maritim.")

    features = [
        "komoditas", "bulan_angka", "curah_hujan_mm",
        "kecepatan_angin_maks_ms", "kecepatan_angin_rata_ms",
    ]
    regression_targets = ["surplus_bulanan", "kebutuhan_bulanan"]
    d = monthly.dropna(subset=features + regression_targets).sort_values("periode_iklim").copy()
    periods = sorted(d["periode_iklim"].unique())
    if len(periods) < 6:
        st.error("Periode data terlalu sedikit untuk evaluasi model berbasis waktu.")
        st.stop()

    train_percent = st.slider(
        "Persentase data latih (urut waktu)",
        min_value=50,
        max_value=85,
        value=80,
        step=5,
    )
    split_idx = min(len(periods) - 1, max(1, int(len(periods) * train_percent / 100)))
    train_periods = set(periods[:split_idx])
    test_periods = set(periods[split_idx:])
    train = d[d["periode_iklim"].isin(train_periods)].copy()
    test = d[d["periode_iklim"].isin(test_periods)].copy()

    def make_preprocessor():
        return ColumnTransformer([
            ("komoditas", OneHotEncoder(handle_unknown="ignore"), ["komoditas"]),
            ("angka", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), features[1:]),
        ])

    risk_train = train.dropna(subset=["surplus_bulanan"])
    risk_test = test.dropna(subset=["surplus_bulanan"])
    risk_train = risk_train.copy()
    risk_test = risk_test.copy()
    risk_train["risiko_defisit"] = risk_train["surplus_bulanan"] < 0
    risk_test["risiko_defisit"] = risk_test["surplus_bulanan"] < 0
    risk_model = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", KNeighborsClassifier(
            n_neighbors=min(5, len(risk_train)),
            weights="distance",
        )),
    ])
    risk_model.fit(risk_train[features], risk_train["risiko_defisit"])

    st.markdown("#### Evaluasi pada periode terbaru yang tidak dipakai saat pelatihan")
    if len(risk_test):
        risk_test_predictions = risk_model.predict(risk_test[features])
        metric_cols = st.columns(2)
        metric_cols[0].metric(
            "Balanced accuracy KNN",
            f"{balanced_accuracy_score(risk_test['risiko_defisit'], risk_test_predictions):.1%}",
        )
        metric_cols[1].metric(
            "F1-score kelas defisit",
            f"{f1_score(risk_test['risiko_defisit'], risk_test_predictions, zero_division=0):.1%}",
        )
        st.caption(
            f"Evaluasi KNN mencakup {len(risk_test)} observasi komoditas-bulan; "
            f"{int(risk_test['risiko_defisit'].sum())} di antaranya berlabel defisit, "
            f"dari {len(test_periods)} bulan iklim yang ditahan. "
            "Kondisi iklim yang sama dipakai oleh beberapa komoditas, sehingga skor tinggi pada sampel kecil ini belum membuktikan performa umum."
        )

    regressors = {}
    surplus_eval_model = None
    regression_scores = []
    test_predictions = {}
    for target in regression_targets:
        target_train = train.dropna(subset=[target])
        target_test = test.dropna(subset=[target])
        model = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LinearRegression()),
        ])
        model.fit(target_train[features], target_train[target])
        regressors[target] = model
        if target == "surplus_bulanan":
            surplus_eval_model = model
        if len(target_test):
            prediction = model.predict(target_test[features])
            baseline_prediction = [target_train[target].mean()] * len(target_test)
            for model_name, values in [
                ("Regresi Linier", prediction),
                ("Baseline (rata-rata latih)", baseline_prediction),
            ]:
                regression_scores.append({
                    "Target": "Surplus bulanan" if target == "surplus_bulanan" else "Kebutuhan bulanan",
                    "Model": model_name,
                    "MAE": mean_absolute_error(target_test[target], values),
                    "RMSE": mean_squared_error(target_test[target], values) ** 0.5,
                    "R²": r2_score(target_test[target], values) if len(target_test) > 1 else float("nan"),
                })
            if target == "surplus_bulanan":
                test_predictions = {
                    "actual": target_test[target].to_numpy(),
                    "predicted": prediction,
                }

    risk_model.fit(d[features], d["surplus_bulanan"] < 0)
    for target in regression_targets:
        model = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", LinearRegression()),
        ])
        model.fit(d[features], d[target])
        regressors[target] = model

    st.markdown("#### Perbandingan performa regresi")
    st.dataframe(
        pd.DataFrame(regression_scores).round(3),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Data latih: {len(train_periods)} bulan ({len(train)} observasi komoditas-bulan) | "
        f"Data uji: {len(test_periods)} bulan ({len(test)} observasi). "
        f"Uji memakai periode {min(test_periods)} sampai {max(test_periods)}; "
        "semua observasi pada periode tersebut dikeluarkan dari pelatihan. "
        "Skor adalah evaluasi awal pada data terbatas, bukan jaminan performa masa depan."
    )

    if test_predictions:
        st.markdown("#### Aktual vs prediksi surplus (data uji)")
        fig, ax = plt.subplots()
        ax.scatter(test_predictions["actual"], test_predictions["predicted"], alpha=0.75)
        bounds = [
            min(test_predictions["actual"].min(), test_predictions["predicted"].min()),
            max(test_predictions["actual"].max(), test_predictions["predicted"].max()),
        ]
        ax.plot(bounds, bounds, linestyle="--", color="#C0643F", label="Prediksi ideal")
        ax.set_xlabel("Surplus aktual (satuan sumber)")
        ax.set_ylabel("Surplus prediksi (satuan sumber)")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    if surplus_eval_model is not None:
        st.markdown("#### Koefisien Regresi Linier surplus")
        coefficient_names = surplus_eval_model.named_steps["preprocess"].get_feature_names_out()
        coefficients = pd.DataFrame({
            "Fitur": coefficient_names,
            "Koefisien": surplus_eval_model.named_steps["model"].coef_,
        }).sort_values("Koefisien", key=lambda values: values.abs(), ascending=False)
        st.dataframe(coefficients.round(3), width="stretch", hide_index=True)
        st.caption("Koefisien menunjukkan asosiasi dalam model pada data latih, bukan bukti sebab-akibat. Fitur komoditas dikodekan one-hot.")

    st.markdown("#### Simulasikan kondisi komoditas dan gangguan")
    commodities = sorted(d["komoditas"].unique().tolist())
    periods_for_month = sorted(d["periode_iklim"].unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        commodity = st.selectbox("Komoditas", commodities)
        period = st.selectbox("Periode acuan iklim (YYYY-MM)", periods_for_month, index=len(periods_for_month) - 1)
        disruption_days = st.number_input(
            "Lama gangguan logistik (hari)", min_value=1, max_value=365, value=7, step=1,
        )
    with c2:
        climate_period = climate[climate["periode"] == period]
        rainfall_default = float(climate_period["curah_hujan_mm"].iloc[0])
        windmax_default = float(climate_period["kecepatan_angin_maks_ms"].iloc[0])
        windavg_default = float(climate_period["kecepatan_angin_rata_ms"].iloc[0])
        rainfall = st.number_input("Curah hujan (mm)", min_value=0.0, value=rainfall_default, step=10.0)
        windmax = st.number_input("Kecepatan angin maksimum (m/s)", min_value=0.0, value=windmax_default, step=0.1)
        windavg = st.number_input("Kecepatan angin rata-rata (m/s)", min_value=0.0, value=windavg_default, step=0.1)
        available_stock = st.number_input(
            "Stok layak pakai saat ini (satuan mengikuti dataset)",
            min_value=0.0, value=0.0, step=1.0,
        )

    if st.button("Analisis risiko dan estimasi buffer", type="primary"):
        month_num = int(pd.to_datetime(period + "-01").month)
        scenario = pd.DataFrame([{
            "komoditas": commodity,
            "bulan_angka": month_num,
            "curah_hujan_mm": rainfall,
            "kecepatan_angin_maks_ms": windmax,
            "kecepatan_angin_rata_ms": windavg,
        }])

        risk_prediction = bool(risk_model.predict(scenario[features])[0])
        risk_probabilities = dict(zip(
            risk_model.classes_,
            risk_model.predict_proba(scenario[features])[0],
        ))
        risk_probability = float(risk_probabilities.get(True, 0.0))
        predicted_surplus = float(regressors["surplus_bulanan"].predict(scenario[features])[0])
        predicted_monthly_need = max(
            0.0, float(regressors["kebutuhan_bulanan"].predict(scenario[features])[0])
        )
        estimated_deficit = max(0.0, -predicted_surplus)
        estimated_need_during_disruption = predicted_monthly_need / 30 * disruption_days
        recommended_buffer = max(0.0, estimated_need_during_disruption - available_stock)

        st.markdown("### Hasil estimasi model")
        if risk_prediction:
            st.error(f"KNN mengklasifikasikan risiko defisit historis ({risk_probability:.1%} probabilitas kelas defisit).")
        else:
            st.success(f"KNN mengklasifikasikan risiko surplus ({1 - risk_probability:.1%} probabilitas kelas surplus).")
        result_cols = st.columns(3)
        result_cols[0].metric("Estimasi surplus bulanan", f"{predicted_surplus:,.2f}")
        result_cols[1].metric("Estimasi defisit", f"{estimated_deficit:,.2f}")
        result_cols[2].metric("Estimasi buffer minimum", f"{recommended_buffer:,.2f}")
        st.caption(
            f"Simulasi buffer = kebutuhan bulanan terprediksi / 30 × {disruption_days} hari "
            f"− stok layak pakai ({available_stock:,.2f}), dibatasi minimum 0. "
            "Gunakan satuan yang konsisten dengan dataset. Ini estimasi sederhana, bukan standar stok resmi."
        )

elif page == "Catatan Metodologi":
    st.markdown("### Cara kerja aplikasi")
    st.markdown("""
1. Membaca sheet **2_Cleaning_Pangan** dan **2_Cleaning_Iklim** dari workbook.
2. Menggabungkan data berdasarkan periode iklim.
3. Mengagregasi data pangan mingguan menjadi total komoditas-bulan.
4. Menampilkan korelasi Spearman untuk eksplorasi hubungan iklim dan pangan.
5. Melatih KNN untuk mengklasifikasikan defisit/surplus dari label historis (surplus bulanan < 0 berarti defisit).
6. Melatih Regresi Linier untuk mengestimasi surplus bulanan dan kebutuhan bulanan dari data historis.
7. Menguji model pada periode terbaru yang tidak disertakan saat pelatihan, lalu melatih ulang pada seluruh data untuk simulasi.
8. Menghitung buffer simulasi dari kebutuhan bulanan terprediksi, lama gangguan yang dimasukkan, dan stok layak pakai yang dimasukkan.
""")
    st.markdown("### Batasan yang harus diketahui")
    st.markdown("""
- Data iklim hanya mencakup 24 bulan; model belum tentu dapat digeneralisasi ke tahun lain.
- KNN mempelajari label defisit/surplus historis, bukan label gangguan pelayaran. Dataset belum memiliki catatan gangguan logistik atau penutupan pelabuhan.
- Korelasi tidak membuktikan sebab-akibat.
- Nilai surplus bulanan merupakan penjumlahan catatan mingguan yang dipasangkan dengan bulan berdasarkan kolom periode iklim.
- Buffer yang ditampilkan adalah simulasi sederhana berdasarkan kebutuhan prediksi dan input stok/durasi, bukan buffer stock resmi atau rekomendasi operasional.
- Jika dataset diperbarui, lakukan pemeriksaan kualitas dan evaluasi ulang model.
""")
    st.markdown("### Saran pengembangan")
    st.write("Untuk versi penelitian final, pertimbangkan validasi silang berbasis waktu, pemeriksaan anomali per komoditas, penambahan data keterlambatan kapal/logistik, serta perbandingan dengan baseline sederhana.")

st.sidebar.markdown("---")
st.sidebar.caption("LAKSAMANA • Prototipe analisis pangan dan iklim")
