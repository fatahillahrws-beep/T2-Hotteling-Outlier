"""
app.py — Dashboard Streamlit: Diagram Kontrol Hotelling T² Robust (7 estimator × 3 batas kontrol)
================================================================================================
Jalankan:   streamlit run app.py
"""
from __future__ import annotations

import inspect
import os
import time
import traceback

import numpy as np
import pandas as pd
import streamlit as st

import charts as ch
import core
from core import (BATAS_SEMUA, DEFAULT_PRM, DIST_SEMUA, EST_SEMUA, KOLOM_EXTRA, LEBIH_BESAR_BAIK, WARNA_EST)

st.set_page_config(page_title="Diagram Kontrol T² Robust", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

pm = ch.pm

# ======================================================================================
# Kompatibilitas versi Streamlit (width="stretch" vs use_container_width)
# ======================================================================================
def _wide(fn):
    ps = inspect.signature(fn).parameters
    if "width" in ps:
        return {"width": "stretch"}
    if "use_container_width" in ps:
        return {"use_container_width": True}
    return {}


W_PLOT, W_DF = _wide(st.plotly_chart), _wide(st.dataframe)
W_BTN, W_DL = _wide(st.button), _wide(st.download_button)
PLOT_CFG = dict(displaylogo=False, toImageButtonOptions=dict(format="png", scale=2))


def plot(fig, key, select=False):
    kw = dict(config=PLOT_CFG, key=key, theme=None, **W_PLOT)
    if select:
        kw.update(on_select="rerun", selection_mode="points")
    return st.plotly_chart(fig, **kw)


def titik_terpilih(ev):
    """Ambil daftar titik yang dipilih dari event st.plotly_chart (aman untuk berbagai versi)."""
    try:
        sel = ev["selection"] if ev is not None else None
        return list(sel["points"]) if sel else []
    except Exception:
        return []


# ======================================================================================
# CSS
# ======================================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1450px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #f5f8fc; border-right: 1px solid #e3e9f2; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
h1, h2, h3 { letter-spacing: -0.01em; color: #0f172a; }
h3 { margin-top: 1.4rem; }

.hero { background: linear-gradient(115deg, #0b2350 0%, #16407f 52%, #2a78d6 100%); color: #fff;
        padding: 1.5rem 2rem 1.3rem 2rem; border-radius: 18px; margin-bottom: .9rem;
        box-shadow: 0 10px 28px rgba(22,64,127,.22); }
.hero .t { font-size: 1.75rem; font-weight: 800; line-height: 1.2; }
.hero .s { font-size: .98rem; opacity: .88; margin-top: .3rem; }
.chips { margin-top: .85rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.chip { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28); color: #fff;
        border-radius: 999px; padding: .15rem .7rem; font-size: .78rem; font-weight: 500; }
.status { display: inline-block; border-radius: 999px; padding: .2rem .8rem; font-size: .8rem; font-weight: 600;
          margin: .2rem 0 .8rem 0; }
.st-ok { background: #dcfce7; color: #166534; } .st-warn { background: #fef3c7; color: #92400e; }
.st-none { background: #e2e8f0; color: #334155; }

.kpi { background: #fff; border: 1px solid #e4eaf3; border-radius: 14px; padding: .8rem 1rem .75rem 1rem;
       box-shadow: 0 1px 2px rgba(15,23,42,.05); height: 100%; }
.kpi .l { font-size: .74rem; color: #64748b; font-weight: 600; }
.kpi .v { font-size: 1.5rem; font-weight: 800; color: #0f172a; line-height: 1.25; }
.kpi .s { font-size: .78rem; color: #64748b; }

.interp { border: 1px solid #d8e6f8; border-left: 4px solid #2a78d6; background: #f4f8ff;
          border-radius: 4px 12px 12px 4px; padding: .75rem 1.05rem .6rem 1.05rem; margin: .3rem 0 1.5rem 0; }
.interp .it { font-size: .78rem; font-weight: 700; color: #1d4f9c; margin-bottom: .1rem; }
.interp .ib { font-size: .92rem; color: #1f2937; line-height: 1.55; margin-bottom: .55rem; }

.card { background: #fff; border: 1px solid #e4eaf3; border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(15,23,42,.05); height: 100%; }
.card b.h { color: #16407f; }
.brand { font-weight: 800; font-size: 1.05rem; color: #0f2a5c; margin-bottom: .1rem; }
.brand-s { font-size: .78rem; color: #64748b; margin-bottom: .7rem; }
.sec-note { color: #64748b; font-size: .86rem; margin: -.3rem 0 .6rem 0; }
div[data-testid="stExpander"] { border-radius: 12px; background: #fff; }
button[kind="primary"] { font-weight: 700; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================================================================================
# Konfigurasi bawaan & preset
# ======================================================================================
NCPU = os.cpu_count() or 1
DEFAULTS = dict(
    dist="normal", n=1000, p=10, pers=20, rho=0.0, mu_out=3.0, alpha=0.00273, seed=2026,
    reps=100, n_ref=100, b_boot=100, n_jobs=NCPU,
    est_aktif=list(EST_SEMUA), batas_aktif=list(BATAS_SEMUA), batas_utama="KDE", positive=0,
    c_bacon=4, bacon_version=2, max_bacon_iter=100, n_iter_ogk=2, n_directions=500, sd_cutoff=3.0,
    h_frac=0.75, reg_rho=0.10, mrwcd_max_iter=100, irdet_q=0.975, irdet_max_iter=50,
    s_max_iter=100, s_tol=1e-7,
    g_p=[3, 10, 20], g_eps=[5, 10, 20, 30], g_reps=100,
    preset="Standar · 100 iterasi",
)
PRM_KEYS = ["c_bacon", "bacon_version", "max_bacon_iter", "n_iter_ogk", "n_directions", "sd_cutoff", "h_frac",
            "reg_rho", "mrwcd_max_iter", "irdet_q", "irdet_max_iter", "s_max_iter", "s_tol"]
PRESETS = {
    "Demo cepat · ±1 menit": dict(n=300, p=5, reps=20, n_ref=20, b_boot=100),
    "Standar · 100 iterasi": dict(n=1000, p=10, reps=100, n_ref=100, b_boot=100),
    "Sesuai notebook · 500 replikasi": dict(n=1000, p=10, reps=500, n_ref=100, b_boot=1000),
    "Presisi tinggi · 1000 iterasi": dict(n=1000, p=10, reps=1000, n_ref=1000, b_boot=1000),
}
LABEL_DIST = {"normal": "Normal multivariat", "gamma": "Gamma (2, 1)", "eksponensial": "Eksponensial (θ = 2)",
              "weibull": "Weibull (2, 1)"}
NAV = ["🏠 Beranda", "📊 Skenario Utama", "📈 Diagram Kontrol", "🔬 Analisis Lanjutan", "🧮 Simulasi Grid",
       "📥 Unduh Hasil"]
MET = {"accuracy": ("Hit rate", True), "fp_rate": ("FP rate", False), "fn_rate": ("FN rate", False),
       "f1": ("F1", True), "balanced_accuracy": ("Balanced accuracy", True), "auc": ("AUC", True),
       "waktu_detik": ("Waktu (dtk)", False)}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("nav", NAV[0])
for k in ["g_p", "g_eps", "g_reps"]:            # pertahankan state widget yang tidak selalu tampil
    st.session_state[k] = st.session_state[k]


def apply_preset():
    for k, v in PRESETS[st.session_state.preset].items():
        st.session_state[k] = v


def reset_semua():
    for k, v in DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v
    apply_preset()


def ambil_cfg():
    ss = st.session_state
    prm = dict(DEFAULT_PRM)
    prm.update({k: ss[k] for k in PRM_KEYS})
    prm["bacon_alpha"] = ss["alpha"]
    bs = [b for b in BATAS_SEMUA if b in ss["batas_aktif"]]
    es = [e for e in EST_SEMUA if e in ss["est_aktif"]]
    bu = ss["batas_utama"] if ss["batas_utama"] in bs else (bs[0] if bs else "KDE")
    cfg = {k: ss[k] for k in ["dist", "n", "p", "pers", "rho", "mu_out", "alpha", "seed", "reps", "n_ref",
                              "b_boot", "n_jobs", "positive"]}
    cfg.update(est_aktif=es, batas_aktif=bs, batas_utama=bu, prm=prm)
    if cfg["dist"] != "normal":
        cfg["rho"] = 0.0
    return cfg


def cfg_valid(cfg):
    if not cfg["est_aktif"]:
        return "Pilih minimal satu estimator."
    if not cfg["batas_aktif"]:
        return "Pilih minimal satu batas kontrol."
    if cfg["n"] < 5 * cfg["p"]:
        return f"Jumlah observasi n harus ≥ 5 × p (saat ini n = {cfg['n']}, p = {cfg['p']})."
    return None


def tabel_cfg(cfg):
    n2 = round(cfg["n"] * cfg["pers"] / 100)
    baris = [("Distribusi", LABEL_DIST[cfg["dist"]]), ("Jumlah observasi (n)", cfg["n"]),
             ("Jumlah variabel (p)", cfg["p"]), ("Proporsi outlier (ε)", f"{cfg['pers']}%  ({n2} observasi)"),
             ("Korelasi (ρ)", cfg["rho"]), ("Pergeseran outlier (μ)", cfg["mu_out"]),
             ("Laju alarm palsu (α)", cfg["alpha"]), ("Seed", cfg["seed"]),
             ("Replikasi Fase II", cfg["reps"]), ("Dataset Fase I (KDE & bootstrap)", cfg["n_ref"]),
             ("Sampel bootstrap (B)", cfg["b_boot"]), ("Core CPU", cfg["n_jobs"]),
             ("Kelas positif Sens/Spec/F1", "0 — Normal" if cfg["positive"] == 0 else "1 — Outlier"),
             ("Estimator", ", ".join(cfg["est_aktif"])), ("Batas kontrol", ", ".join(cfg["batas_aktif"])),
             ("Batas utama", cfg["batas_utama"])]
    baris += [(f"Param · {k}", v) for k, v in cfg["prm"].items() if k != "bacon_alpha"]
    return pd.DataFrame(baris, columns=["Parameter", "Nilai"]).astype({"Nilai": str})


# ======================================================================================
# Komponen UI kecil
# ======================================================================================
def kpi(label, value, sub="", warna="#2a78d6"):
    return (f'<div class="kpi" style="border-top:3px solid {warna}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def kpi_row(items):
    for c, it in zip(st.columns(len(items)), items):
        c.markdown(kpi(*it), unsafe_allow_html=True)


def callout(cara, hasil):
    st.markdown(f'<div class="interp"><div class="it">📖 Cara membaca grafik</div><div class="ib">{cara}</div>'
                f'<div class="it">💡 Interpretasi hasil</div><div class="ib">{hasil}</div></div>',
                unsafe_allow_html=True)


def seksi(judul, catatan=None):
    st.markdown(f"### {judul}")
    if catatan:
        st.markdown(f'<div class="sec-note">{catatan}</div>', unsafe_allow_html=True)


def nav_widget(options, key):
    if hasattr(st, "segmented_control"):
        return st.segmented_control("nav", options, key=key, label_visibility="collapsed")
    return st.radio("nav", options, key=key, horizontal=True, label_visibility="collapsed")


def pilih_batas(res, key, label="Batas kontrol"):
    bs = res["cfg"]["batas_aktif"]
    return st.selectbox(label, bs, index=bs.index(res["cfg"]["batas_utama"]), key=key)


# ======================================================================================
# Eksekusi simulasi
# ======================================================================================
class Throttle:
    def __init__(self, dt=0.12):
        self.t, self.dt = 0.0, dt

    def ok(self):
        now = time.time()
        if now - self.t >= self.dt:
            self.t = now
            return True
        return False


def jalankan_utama(cfg, slot):
    ests, bs = cfg["est_aktif"], cfg["batas_aktif"]
    E = len(ests)
    butuh = bool({"KDE", "Boot"} & set(bs))
    u1 = cfg["n_ref"] * E if butuh else 0
    tot = u1 + cfg["reps"] * E
    bar = slot.progress(0.0, text="Menyiapkan simulasi…")
    thr, t0 = Throttle(), time.time()

    def cb1(i, t, label):
        if thr.ok() or i == t:
            bar.progress(min(i / tot, 1.0), text=f"{label} — {i}/{t} dataset · {time.time() - t0:.0f} dtk")

    def cb2(i, t):
        if thr.ok() or i == t:
            bar.progress(min((u1 + i * E) / tot, 1.0),
                         text=f"Fase II · replikasi {i}/{t} · {time.time() - t0:.0f} dtk")

    cl, faseI = core.hitung_batas(cfg["n"], cfg["p"], cfg["rho"], cfg["alpha"], cfg["n_ref"], cfg["seed"] + 99,
                                  ests, bs, cfg["dist"], cfg["prm"], cfg["b_boot"], cfg["n_jobs"], cb1)
    df, (t2s, y, X) = core.simulasi(cfg["n"], cfg["p"], cfg["pers"], cfg["mu_out"], cfg["rho"], cfg["reps"],
                                    cfg["seed"], cl, cfg["dist"], ests, bs, cfg["prm"], cfg["positive"],
                                    cfg["n_jobs"], cb2)
    metode = [f"T2-{e}-{b}" for e in ests for b in bs]
    rata = df.groupby("metode")[KOLOM_EXTRA + ["waktu_detik"]].mean().reindex(metode)
    sd = df.groupby("metode")[KOLOM_EXTRA].std().reindex(metode)
    rata["CL"] = pd.Series(cl)
    rng = np.random.default_rng(0)
    faseI = {e: (rng.choice(v, 300_000, replace=False) if len(v) > 300_000 else v) for e, v in faseI.items()}
    bar.progress(1.0, text=f"Selesai dalam {time.time() - t0:.1f} detik")
    return dict(cfg=cfg, df=df, cl=cl, faseI=faseI, last=(t2s, y, X), rata=rata, sd=sd, metode=metode,
                waktu=time.time() - t0)


def jalankan_grid(cfg, gp, geps, greps, slot):
    ests, bs, E = cfg["est_aktif"], cfg["batas_aktif"], len(cfg["est_aktif"])
    P, EPS = sorted(gp), sorted(geps)
    butuh = bool({"KDE", "Boot"} & set(bs))
    u1 = cfg["n_ref"] * E if butuh else 0
    u_sc = greps * E
    tot = len(P) * (u1 + len(EPS) * u_sc)
    st_ = {"done": 0}
    bar = slot.progress(0.0, text="Menyiapkan grid…")
    thr, t0 = Throttle(), time.time()

    def upd(extra, teks):
        if thr.ok():
            bar.progress(min((st_["done"] + extra) / tot, 1.0), text=f"{teks} · {time.time() - t0:.0f} dtk")

    grid = pd.DataFrame()
    for pg in P:
        clg, _ = core.hitung_batas(cfg["n"], pg, cfg["rho"], cfg["alpha"], cfg["n_ref"], cfg["seed"] + 99 + pg,
                                   ests, bs, cfg["dist"], cfg["prm"], cfg["b_boot"], cfg["n_jobs"],
                                   lambda i, t, l, pg=pg: upd(i, f"p = {pg} · {l}"))
        st_["done"] += u1
        for eg in EPS:
            d, _ = core.simulasi(cfg["n"], pg, eg, cfg["mu_out"], cfg["rho"], greps,
                                 cfg["seed"] + 1000 * pg + 10 * eg, clg, cfg["dist"], ests, bs, cfg["prm"],
                                 cfg["positive"], cfg["n_jobs"],
                                 lambda i, t, pg=pg, eg=eg: upd(i * E, f"p = {pg}, ε = {eg}% · replikasi {i}/{t}"))
            a = d.groupby("metode")[KOLOM_EXTRA + ["waktu_detik"]].mean().reset_index()
            a.insert(0, "p", pg)
            a.insert(1, "pers", eg)
            a["CL"] = a.metode.map(clg)
            grid = pd.concat([grid, a], ignore_index=True)
            st_["done"] += u_sc
            bar.progress(min(st_["done"] / tot, 1.0), text=f"p = {pg}, ε = {eg}% selesai · {time.time() - t0:.0f} dtk")
    bar.progress(1.0, text=f"Grid selesai dalam {time.time() - t0:.1f} detik")
    return dict(df=grid, cfg=cfg, P=P, EPS=EPS, reps=greps, waktu=time.time() - t0)


# ======================================================================================
# Kalimat interpretasi otomatis (dinamis, dihitung dari hasil)
# ======================================================================================
def _daftar(items, maks=3):
    items = list(items)
    if len(items) <= maks:
        return ", ".join(f"<b>{i}</b>" for i in items)
    return ", ".join(f"<b>{i}</b>" for i in items[:maks]) + f" dan {len(items) - maks} lainnya"


def ip_ringkasan(res):
    cfg, rata = res["cfg"], res["rata"]
    base = 1 - round(cfg["n"] * cfg["pers"] / 100) / cfg["n"]
    r = rata.sort_values("accuracy", ascending=False)
    top = r.index[0]
    t = (f"<b>{pm(top)}</b> memiliki hit rate tertinggi (<b>{r.accuracy.iloc[0]:.4f}</b>) dengan FP rate "
         f"{r.fp_rate.iloc[0]:.4f} dan FN rate {r.fn_rate.iloc[0]:.4f}. ")
    ok = int((rata.fp_rate <= cfg["alpha"]).sum())
    t += (f"Sebanyak <b>{ok} dari {len(rata)}</b> kombinasi menjaga alarm palsu di bawah α = {cfg['alpha']:g}. ")
    lemah = [pm(m) for m in r.index if r.loc[m, "accuracy"] <= base + 0.02]
    if lemah:
        t += (f"Hit rate 'dasar' bila diagram tidak pernah memberi alarm adalah {base:.2f} (= 1 − ε); "
              f"kombinasi {_daftar(lemah)} berada di kisaran itu, artinya hampir semua outlier lolos (efek masking).")
    else:
        t += f"Semua kombinasi berada jauh di atas hit rate dasar {base:.2f} (= 1 − ε), sehingga tidak ada yang sepenuhnya gagal."
    return t


def ip_confusion(res, batas):
    df, ests = res["df"], res["cfg"]["est_aktif"]
    g = {e: df[df.metode == f"T2-{e}-{batas}"][["TN", "FP", "FN", "TP"]].mean() for e in ests}
    tp = max(ests, key=lambda e: g[e].TP)
    fnw = max(ests, key=lambda e: g[e].FN)
    fpw = max(ests, key=lambda e: g[e].FP)
    n_out = g[tp].TP + g[tp].FN
    return (f"Pada batas <b>{batas}</b>, <b>{tp}</b> menangkap paling banyak outlier: {g[tp].TP:.1f} dari {n_out:.0f} "
            f"({100 * g[tp].TP / n_out:.1f}%). Outlier paling sering lolos pada <b>{fnw}</b> "
            f"(rata-rata {g[fnw].FN:.1f} observasi/replikasi), sedangkan alarm palsu terbanyak terjadi pada "
            f"<b>{fpw}</b> ({g[fpw].FP:.2f} observasi normal salah ditandai). Diagonal yang pekat berarti keputusan benar; "
            f"sel di luar diagonal yang pekat menandakan kelemahan estimator tersebut.")


def ip_bars(res):
    cfg, rata = res["cfg"], res["rata"]
    ests, bs, a = cfg["est_aktif"], cfg["batas_aktif"], cfg["alpha"]
    teks = []
    for b in bs:
        ms = [f"T2-{e}-{b}" for e in ests]
        ok = int((rata.loc[ms, "fp_rate"] <= a).sum())
        teks.append(f"batas <b>{b}</b>: {ok}/{len(ms)} estimator memenuhi FP ≤ α, rata-rata FN rate "
                    f"{rata.loc[ms, 'fn_rate'].mean():.3f}")
    best = rata.sort_values(["accuracy"], ascending=False).index[0]
    return ("; ".join(teks) + f". Kombinasi dengan hit rate tertinggi adalah <b>{pm(best)}</b>. "
            "Bila FP rate suatu batang jauh di atas garis α, diagram terlalu sensitif (banyak alarm palsu); "
            "bila FN rate tinggi, diagram terlalu longgar atau estimatornya terkena masking.")


def ip_control(t2, y, cl, e, bs, bu):
    c = cl[f"T2-{e}-{bu}"]
    pred = t2 > c
    tp, fn, fp = int((pred & y).sum()), int((~pred & y).sum()), int((pred & ~y).sum())
    urut = sorted(bs, key=lambda b: cl[f"T2-{e}-{b}"])
    rentang = " &lt; ".join(f"{b} ({cl[f'T2-{e}-{b}']:.2f})" for b in urut)
    return (f"Dengan batas <b>{bu}</b> (CL = {c:.2f}), T²-{e} memberi <b>{int(pred.sum())} alarm</b>: {tp} dari {int(y.sum())} "
            f"outlier terdeteksi (merah), {fn} outlier lolos (oranye), dan {fp} alarm palsu (biru). "
            f"Urutan batas dari paling ketat ke paling longgar: {rentang}. Semakin rendah garis batas, semakin banyak alarm; "
            f"selisih antar-batas menunjukkan seberapa besar asumsi distribusi memengaruhi keputusan.")


def ip_hist(t2, cl, e, bs, alpha):
    bagian, buruk = [], []
    for b in bs:
        c = cl[f"T2-{e}-{b}"]
        emp = float((t2 > c).mean())
        bagian.append(f"{b}: CL = {c:.2f} (melampaui {100 * emp:.3f}% data Fase I)")
        if emp > 1.5 * alpha:
            buruk.append(f"{b} terlalu ketat")
        elif emp < 0.5 * alpha:
            buruk.append(f"{b} terlalu longgar")
    kes = ("Semua batas mendekati target α." if not buruk else "Ketidaksesuaian: " + ", ".join(buruk) + " terhadap α.")
    return (f"Target proporsi data in-control di atas batas adalah α = {100 * alpha:.3f}%. " + "; ".join(bagian) + ". " + kes +
            " Batas F berasumsi T² klasik berdistribusi F; estimator robust umumnya memiliki ekor berbeda sehingga batas F "
            "bisa meleset, sedangkan KDE/Bootstrap belajar langsung dari sebaran T² Fase I.")


def ip_dist(t2s, y, cl, ests, bu):
    rasio = {e: float(np.median(t2s[e][y]) / cl[f"T2-{e}-{bu}"]) for e in ests}
    best, worst = max(rasio, key=rasio.get), min(rasio, key=rasio.get)
    mask = [e for e in ests if rasio[e] < 1]
    t = (f"Rasio median T² outlier terhadap CL: terbesar pada <b>{best}</b> ({rasio[best]:.1f}×), terkecil pada "
         f"<b>{worst}</b> ({rasio[worst]:.2f}×). ")
    if mask:
        t += (f"Pada {_daftar(mask)} median outlier berada <b>di bawah</b> CL, artinya lebih dari separuh outlier lolos — "
              "kotak merah menumpuk dengan kotak abu-abu (masking). ")
    else:
        t += "Median outlier semua estimator berada di atas CL, sehingga sebagian besar outlier dapat dipisahkan. "
    return t + "Semakin jauh kotak merah dari kotak abu-abu dan semakin tepat CL berada di celahnya, semakin baik."


def ip_cl(cl, ests, bs):
    kal = []
    if "F" in bs:
        r = [(e, cl[f"T2-{e}-F"]) for e in ests][0]
        kal.append(f"Batas F bernilai sama ({r[1]:.2f}) untuk semua estimator karena hanya bergantung pada n, p, α")
    for b in [x for x in bs if x != "F"]:
        v = {e: cl[f"T2-{e}-{b}"] for e in ests}
        hi, lo = max(v, key=v.get), min(v, key=v.get)
        kal.append(f"batas {b} tertinggi pada <b>{hi}</b> ({v[hi]:.2f}) dan terendah pada <b>{lo}</b> ({v[lo]:.2f})")
    return ("; ".join(kal) + ". CL yang sangat tinggi menandakan estimator menghasilkan T² Fase I yang besar (kovarians dipersempit), "
            "sehingga diagramnya lebih konservatif; CL rendah membuat diagram lebih sensitif namun berisiko alarm palsu.")


def ip_roc(t2s, y, cl, ests, bu):
    auc = {e: core.auc_score(y, t2s[e]) for e in ests}
    op = {}
    for e in ests:
        pred = t2s[e] > cl[f"T2-{e}-{bu}"]
        op[e] = float((pred & y).sum() / max(y.sum(), 1))
    best, worst = max(auc, key=auc.get), min(auc, key=auc.get)
    salah = [e for e in ests if auc[e] >= 0.95 and op[e] < 0.8]
    t = (f"AUC tertinggi: <b>{best}</b> ({auc[best]:.4f}); terendah: <b>{worst}</b> ({auc[worst]:.4f}). AUC menilai kemampuan "
         f"estimator memisahkan outlier dari data normal <i>tanpa bergantung pada batas kontrol</i>. ")
    if salah:
        t += (f"Perhatikan {_daftar(salah)}: AUC tinggi tetapi titik kerja (◆) pada batas {bu} hanya menangkap &lt; 80% outlier — "
              "estimatornya baik, namun <b>batas kontrolnya</b> terlalu longgar.")
    else:
        t += f"Titik kerja (◆) pada batas {bu} berada dekat sudut kiri-atas untuk estimator yang berkinerja baik."
    return t


def ip_stab(df, ests, bs_, metric, label):
    s = df[df.batas == bs_].groupby("estimator")[metric].std()
    s = s.reindex(ests).dropna()
    if s.empty:
        return "Data tidak tersedia."
    st_, vr = s.idxmin(), s.idxmax()
    return (f"Estimator paling stabil untuk {label}: <b>{st_}</b> (simpangan baku {s[st_]:.4f}); paling berfluktuasi: "
            f"<b>{vr}</b> ({s[vr]:.4f}). Kotak yang pendek berarti hasilnya konsisten dari satu dataset ke dataset lain; kotak yang "
            "tinggi atau banyak titik pencilan menandakan kinerja bergantung pada keberuntungan sampel.")


def ip_radar(rata, ests, bs_):
    sk = {}
    lemah = {}
    nm = ["Hit rate", "Spesifisitas", "Deteksi outlier", "F1", "Balanced acc.", "AUC"]
    for e in ests:
        r = rata.loc[f"T2-{e}-{bs_}"]
        v = [r.accuracy, 1 - r.fp_rate, 1 - r.fn_rate, r.f1, r.balanced_accuracy, r.auc]
        sk[e] = float(np.nanmean(v))
        lemah[e] = nm[int(np.nanargmin(v))]
    best, worst = max(sk, key=sk.get), min(sk, key=sk.get)
    return (f"Rata-rata enam metrik tertinggi: <b>{best}</b> ({sk[best]:.4f}) — poligonnya paling luas dan merata. "
            f"Poligon terkecil: <b>{worst}</b> ({sk[worst]:.4f}) dengan sumbu terlemah '{lemah[worst]}'. "
            "Poligon yang penyok pada satu sumbu menunjukkan estimator yang unggul di satu aspek tetapi mengorbankan aspek lain.")


def ip_heat(rata, ests, bs):
    ms = [f"T2-{e}-{b}" for e in ests for b in bs]
    z = pd.DataFrame(index=ms)
    for c, (lab, naik) in MET.items():
        v = rata.loc[ms, c].astype(float)
        rg = v.max() - v.min()
        nz = pd.Series(0.5, index=ms) if rg <= 1e-12 else (v - v.min()) / rg
        z[c] = nz if naik else 1 - nz
    sk = z.mean(1).sort_values(ascending=False)
    return (f"Berdasarkan rata-rata skor ternormalisasi seluruh kolom, tiga kombinasi teratas: {_daftar([pm(m) for m in sk.index[:3]])} "
            f"dan terbawah <b>{pm(sk.index[-1])}</b>. Warna dinormalisasi <i>per kolom</i> (hijau = terbaik di antara kombinasi yang dijalankan), "
            "jadi baris yang dominan hijau berarti unggul di hampir semua kriteria, sedangkan kolom waktu sering berwarna berlawanan "
            "dengan kolom akurasi (trade-off).")


def ip_time(rata, ests, bs_, ymetric, ylabel):
    d = pd.DataFrame({e: rata.loc[f"T2-{e}-{bs_}", ["waktu_detik", ymetric]] for e in ests}).T.astype(float)
    cepat, akurat = d.waktu_detik.idxmin(), d[ymetric].idxmax()
    par, best = [], -np.inf
    for e in d.sort_values("waktu_detik").index:
        if d.loc[e, ymetric] > best:
            par.append(e)
            best = d.loc[e, ymetric]
    x = d.waktu_detik[akurat] / d.waktu_detik[cepat]
    return (f"Tercepat: <b>{cepat}</b> ({d.waktu_detik[cepat]:.4f} dtk); {ylabel} tertinggi: <b>{akurat}</b> ({d.loc[akurat, ymetric]:.4f}) "
            f"dengan waktu {x:.1f}× lebih lama dari yang tercepat. Estimator pada garis Pareto ({_daftar(par, 6)}) tidak dikalahkan oleh "
            "estimator lain yang sekaligus lebih cepat <i>dan</i> lebih akurat — pilihan rasional bergantung pada berapa banyak waktu yang rela dikorbankan.")


def ip_conv(df, ests, bs_, metric, label):
    piv = df[df.batas == bs_].pivot(index="replikasi", columns="estimator", values=metric).expanding().mean()
    m = len(piv)
    if m < 10:
        return f"Hanya {m} replikasi sehingga kurva belum informatif; naikkan jumlah replikasi (mis. ≥ 100)."
    ekor = piv.iloc[int(m * 0.7):]
    rg = (ekor.max() - ekor.min()).reindex(ests).dropna()
    mx = rg.idxmax()
    ket = "sudah stabil" if rg.max() < 0.002 else "masih bergerak"
    saran = "Jumlah replikasi cukup untuk kesimpulan yang andal." if rg.max() < 0.002 else \
        "Naikkan jumlah replikasi (hingga 500–1000) agar rata-rata lebih mantap."
    return (f"Pada 30% replikasi terakhir, rentang perubahan rata-rata {label} terbesar adalah {rg.max():.4f} (estimator <b>{mx}</b>) — kurva {ket}. {saran} "
            "Garis yang mendatar berarti tambahan replikasi tidak lagi mengubah kesimpulan (galat Monte Carlo kecil).")


def ip_pca(t2s, y, cl, ests, bs_):
    fn = {e: int((y & ~(t2s[e] > cl[f"T2-{e}-{bs_}"])).sum()) for e in ests}
    fp = {e: int((~y & (t2s[e] > cl[f"T2-{e}-{bs_}"])).sum()) for e in ests}
    b, w = min(ests, key=lambda e: fn[e] + fp[e]), max(ests, key=lambda e: fn[e] + fp[e])
    return (f"Pada replikasi ini, kesalahan terkecil ada pada <b>{b}</b> (FN {fn[b]}, FP {fp[b]}) dan terbesar pada <b>{w}</b> "
            f"(FN {fn[w]}, FP {fp[w]}). Tekan ▶ Putar untuk berganti estimator: titik oranye (◆) yang menumpuk di gugus outlier menunjukkan outlier "
            "yang tidak terdeteksi, dan titik biru (✕) di gugus normal menunjukkan alarm palsu.")


def ip_dd(stt, e, batas):
    return (f"Klasik hanya menandai <b>{stt['found_cl']}</b> dari {stt['n_out']} outlier (di kanan garis hijau), sedangkan jarak robust {e} menandai "
            f"<b>{stt['found_rb']}</b> (di atas garis {batas}). Sebanyak <b>{stt['masked']}</b> outlier 'tersamar' oleh metode klasik (jarak klasik di bawah batas), "
            f"dan {stt['masked_found']} di antaranya berhasil terungkap oleh estimator robust. Titik yang menjauh di atas diagonal adalah bukti bahwa "
            "outlier memengaruhi rata-rata & kovarians klasik sehingga jaraknya terlihat kecil.")


def ip_grid_heat(grid, metode, metric, label, naik):
    g = grid[grid.metode == metode]
    a = g.loc[g[metric].idxmax()] if naik else g.loc[g[metric].idxmin()]
    b = g.loc[g[metric].idxmin()] if naik else g.loc[g[metric].idxmax()]
    return (f"Kondisi terbaik untuk {pm(metode)}: p = {int(a.p)}, ε = {int(a.pers)}% ({label} {a[metric]:.4f}); kondisi terburuk: "
            f"p = {int(b.p)}, ε = {int(b.pers)}% ({label} {b[metric]:.4f}). Sel hijau menandakan kinerja baik, merah menandakan buruk. "
            "Perubahan warna ke arah kanan memperlihatkan dampak bertambahnya proporsi outlier, ke arah bawah dampak bertambahnya dimensi.")


def ip_grid_anim(grid, ests, bs_, metric, label, naik):
    g = grid[grid.metode.map(lambda m: m.split("-")[-1]) == bs_]
    emax = g.pers.max()
    a = g[g.pers == emax].groupby("estimator")[metric].mean()
    best = a.idxmax() if naik else a.idxmin()
    worst = a.idxmin() if naik else a.idxmax()
    return (f"Pada ε tertinggi ({int(emax)}%), estimator terbaik untuk {label} adalah <b>{best}</b> ({a[best]:.4f}) dan terburuk <b>{worst}</b> ({a[worst]:.4f}). "
            "Estimator dengan garis yang tetap mendatar tahan terhadap bertambahnya outlier; garis yang jatuh tajam menandakan titik keruntuhan "
            "(breakdown) di mana outlier mulai 'menguasai' estimasi. Geser slider p / ε untuk melihat perubahannya.")


def ip_grid_fp(grid, ests, bs, alpha, eps_max):
    g = grid[grid.pers <= eps_max]
    fp = g.groupby("metode").fp_rate.mean()
    ok = [pm(m) for m in fp.index if fp[m] <= alpha]
    return (f"{len(ok)} dari {len(fp)} kombinasi memiliki FP rate rata-rata ≤ α = {alpha:g}"
            + (f" (mis. {_daftar(ok, 3)})." if ok else ".") +
            " Batang yang melewati garis putus-putus berarti diagram cenderung terlalu sensitif dan menghasilkan alarm palsu lebih banyak dari yang diizinkan.")


def ip_rank(ring):
    r = ring.sort_values("Hit rate", ascending=False)
    return (f"Peringkat 1: <b>{pm(r.index[0])}</b> (hit rate rata-rata {r['Hit rate'].iloc[0]:.4f}), diikuti <b>{pm(r.index[1])}</b> "
            f"({r['Hit rate'].iloc[1]:.4f}) dan <b>{pm(r.index[2])}</b> ({r['Hit rate'].iloc[2]:.4f}). "
            "Perbedaan panjang batang mencerminkan selisih kinerja rata-rata seluruh skenario p × ε; kolom 'Skenario FP > α' pada tabel "
            "menunjukkan seberapa sering suatu kombinasi melanggar batas alarm palsu." if len(r) >= 3 else "Jalankan lebih banyak kombinasi untuk melihat peringkat.")


# ======================================================================================
# Sidebar
# ======================================================================================
def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.markdown('<div class="brand">📊 Diagram Kontrol T² Robust</div><div class="brand-s">Panel konfigurasi simulasi</div>',
                    unsafe_allow_html=True)
        galat = cfg_valid(ambil_cfg())
        run = st.button("▶  Jalankan Simulasi", type="primary", disabled=galat is not None, **W_BTN)
        if galat:
            st.error(galat)
        st.button("↺ Kembalikan ke bawaan", on_click=reset_semua, **W_BTN)
        st.selectbox("Preset iterasi", list(PRESETS), key="preset", on_change=apply_preset,
                     help="Cara cepat mengatur n, p, jumlah replikasi, dataset Fase I, dan sampel bootstrap. "
                          "Setelah dipilih, setiap nilai masih dapat diubah manual di bawah.")

        with st.expander("① Skenario data", expanded=True):
            st.selectbox("Distribusi data", DIST_SEMUA, key="dist", format_func=LABEL_DIST.get,
                         help="Distribusi data bersih. Outlier dibangkitkan dengan menggeser rata-rata.")
            st.slider("Jumlah observasi (n)", 100, 5000, step=50, key="n",
                      help="Ukuran sampel tiap replikasi. Syarat: n ≥ 5 × p.")
            st.slider("Jumlah variabel (p)", 2, 50, key="p")
            st.slider("Proporsi outlier ε (%)", 1, 49, key="pers",
                      help="Persentase observasi yang digeser menjadi outlier (skenario utama).")
            st.slider("Korelasi antar variabel (ρ)", 0.0, 0.95, step=0.05, key="rho",
                      disabled=ss["dist"] != "normal", help="Hanya berlaku untuk data normal.")
            st.slider("Pergeseran outlier (μ)", 0.5, 10.0, step=0.5, key="mu_out",
                      help="Besar pergeseran; untuk distribusi non-normal dalam satuan simpangan baku.")
            st.number_input("Laju alarm palsu (α)", min_value=0.0001, max_value=0.10, step=0.0001, format="%.5f",
                            key="alpha", help="Bawaan 0,00273 setara batas 3-sigma (ARL₀ ≈ 370).")
            st.number_input("Seed acak", min_value=0, max_value=10_000_000, step=1, key="seed",
                            help="Seed yang sama menghasilkan data yang persis sama (dapat direproduksi).")

        with st.expander("② Iterasi & replikasi", expanded=True):
            st.slider("Replikasi simulasi (Fase II)", 10, 1000, step=10, key="reps",
                      help="Jumlah dataset simulasi. Bawaan 100, maksimum 1000. Lebih banyak = hasil lebih stabil tetapi lebih lama.")
            st.slider("Dataset Fase I (KDE & bootstrap)", 10, 1000, step=10, key="n_ref",
                      help="Jumlah dataset in-control untuk menurunkan batas KDE/Bootstrap.")
            st.slider("Sampel bootstrap (B)", 10, 1000, step=10, key="b_boot",
                      help="Banyaknya resampling untuk batas Bootstrap. Bawaan notebook asli: 1000.")
            st.number_input("Core CPU yang dipakai", min_value=1, max_value=max(NCPU, 1), step=1, key="n_jobs",
                            help=f"Terdeteksi {NCPU} core pada mesin ini.")

        with st.expander("③ Estimator & batas kontrol", expanded=False):
            st.multiselect("Estimator", EST_SEMUA, key="est_aktif",
                           help="Hapus estimator yang tidak diperlukan untuk mempercepat simulasi.")
            st.multiselect("Batas kontrol", BATAS_SEMUA, key="batas_aktif",
                           help="F = klasik (asumsi normal) · KDE = nonparametrik · Boot = bootstrap persentil.")
            opsi = [b for b in BATAS_SEMUA if b in ss["batas_aktif"]] or ["KDE"]
            if ss["batas_utama"] not in opsi:
                ss["batas_utama"] = opsi[0]
            st.selectbox("Batas utama (untuk grafik & tabel ringkas)", opsi, key="batas_utama")
            st.radio("Kelas positif Sensitivity/Specificity/F1", [0, 1], key="positive", horizontal=True,
                     format_func=lambda v: "0 · Normal" if v == 0 else "1 · Outlier",
                     help="0 mengikuti caret/R (kelas 'normal' dianggap positif). 1 menjadikan outlier sebagai kelas positif. "
                          "Hit rate, FP rate, dan FN rate tidak terpengaruh.")

        with st.expander("④ Parameter lanjutan estimator", expanded=False):
            st.caption("**BACON**")
            st.slider("c (ukuran subset awal = c·p)", 2, 10, key="c_bacon")
            st.radio("Versi titik awal", [1, 2], key="bacon_version", horizontal=True,
                     format_func=lambda v: "V1 · Mahalanobis" if v == 1 else "V2 · median")
            st.slider("Iterasi maks. BACON", 10, 500, step=10, key="max_bacon_iter")
            st.caption("**OGK**")
            st.slider("Iterasi OGK", 1, 5, key="n_iter_ogk")
            st.caption("**Stahel–Donoho**")
            st.slider("Jumlah arah acak", 50, 2000, step=50, key="n_directions")
            st.slider("Cutoff outlyingness", 1.5, 6.0, step=0.5, key="sd_cutoff")
            st.caption("**MRWCD**")
            st.slider("Proporsi subset h", 0.5, 0.95, step=0.05, key="h_frac")
            st.slider("Regularisasi ρ", 0.0, 0.5, step=0.05, key="reg_rho")
            st.slider("Iterasi maks. MRWCD", 10, 500, step=10, key="mrwcd_max_iter")
            st.caption("**IRDetMCD**")
            st.slider("Kuantil reweighting q", 0.90, 0.995, step=0.005, key="irdet_q", format="%.3f")
            st.slider("Iterasi maks. reweighting", 5, 200, step=5, key="irdet_max_iter")
            st.caption("**S-estimator**")
            st.slider("Iterasi maks. S", 10, 300, step=10, key="s_max_iter")
            st.select_slider("Toleransi konvergensi", [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], key="s_tol",
                             format_func=lambda v: f"{v:.0e}")

        cfg = ambil_cfg()
        E, B = len(cfg["est_aktif"]), len(cfg["batas_aktif"])
        st.markdown(f'<div class="brand-s">Beban komputasi: Fase I ≈ {cfg["n_ref"] * E:,} penyesuaian estimator · '
                    f'Fase II ≈ {cfg["reps"] * E:,} · {E * B} kombinasi diagram.</div>', unsafe_allow_html=True)
    return run


# ======================================================================================
# Halaman
# ======================================================================================
METODE_INFO = pd.DataFrame([
    ("IRDetMCD", "Pengembangan kami", "DetMCD + reweighting berulang hingga himpunan observasi bersih stabil (Cerioli, 2010)"),
    ("DetMCD", "Kami", "MCD deterministik dengan 6 titik awal + C-steps (Hubert, Rousseeuw & Verdonck, 2012)"),
    ("S", "Kami", "S-estimator Tukey biweight, breakdown 50% (Rousseeuw & Yohai, 1984)"),
    ("BACON", "Teman", "Subset dasar tumbuh bertahap, versi 2, c = 4 (Billor, Hadi & Velleman, 2000)"),
    ("OGK", "Teman", "Orthogonalized Gnanadesikan–Kettenring, skala MAD (Maronna & Zamar, 2002)"),
    ("SD", "Teman", "Stahel–Donoho: outlyingness dari proyeksi acak (Stahel, 1981; Donoho, 1982)"),
    ("MRWCD", "Teman", "Minimum Regularized Weighted Covariance Determinant, h = 75%, bobot 1/(1+d²)"),
], columns=["Estimator", "Asal", "Ide utama & referensi"])


def header(res):
    cfg = res["cfg"] if res else ambil_cfg()
    n2 = round(cfg["n"] * cfg["pers"] / 100)
    chips = [f"{LABEL_DIST[cfg['dist']]}", f"n = {cfg['n']}", f"p = {cfg['p']}", f"ε = {cfg['pers']}% ({n2} outlier)",
             f"μ = {cfg['mu_out']:g}", f"α = {cfg['alpha']:g}", f"{cfg['reps']} replikasi"]
    st.markdown(
        '<div class="hero"><div class="t">Diagram Kontrol Hotelling T² Robust</div>'
        '<div class="s">Perbandingan 7 estimator robust × 3 batas kontrol (F, KDE, Bootstrap) untuk deteksi outlier multivariat — '
        'simulasi Monte Carlo interaktif</div><div class="chips">' +
        "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div></div>", unsafe_allow_html=True)
    if res is None:
        st.markdown('<span class="status st-none">● Belum ada hasil — atur konfigurasi di panel kiri lalu klik “Jalankan Simulasi”</span>',
                    unsafe_allow_html=True)
    elif res["cfg"] != ambil_cfg():
        st.markdown('<span class="status st-warn">● Konfigurasi di panel kiri berbeda dari hasil yang ditampilkan — jalankan ulang untuk memperbarui</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status st-ok">● Hasil sesuai konfigurasi · selesai dalam {res["waktu"]:.1f} detik</span>',
                    unsafe_allow_html=True)


def page_beranda(res):
    if res is not None:
        seksi("Temuan utama")
        rata, cfg = res["rata"], res["cfg"]
        r = rata.sort_values("accuracy", ascending=False)
        kpi_row([("Kombinasi terbaik (hit rate)", pm(r.index[0]), f"hit rate {r.accuracy.iloc[0]:.4f}"),
                 ("FP rate terendah", pm(rata.fp_rate.idxmin()), f"{rata.fp_rate.min():.4f} (α = {cfg['alpha']:g})", "#eb6834"),
                 ("FN rate terendah", pm(rata.fn_rate.idxmin()), f"{rata.fn_rate.min():.4f}", "#1baf7a"),
                 ("AUC tertinggi", pm(rata.auc.idxmax()), f"{rata.auc.max():.4f}", "#4a3aa7")])
        st.write("")
        callout("Ringkasan otomatis dari seluruh kombinasi pada skenario utama.", ip_ringkasan(res))
    c1, c2 = st.columns([1.15, 1])
    with c1:
        seksi("Cara menggunakan")
        st.markdown(
            "1. **Atur konfigurasi** di panel kiri — mulai dari *preset* (bawaan: 100 iterasi), lalu sesuaikan skenario data, jumlah iterasi, estimator, dan parameter lanjutan.\n"
            "2. Klik **▶ Jalankan Simulasi**. Progres tampil langsung; seluruh core CPU dipakai secara paralel.\n"
            "3. Jelajahi halaman **Skenario Utama**, **Diagram Kontrol**, dan **Analisis Lanjutan**. Setiap grafik interaktif: *hover* untuk detail, "
            "*klik legenda* untuk menyembunyikan seri, *klik titik/batang* untuk rincian, dan tekan **▶ Putar** pada grafik beranimasi.\n"
            "4. Buka **Simulasi Grid** untuk membandingkan kinerja pada banyak kombinasi p × ε, lalu **Unduh Hasil** untuk Excel/CSV.")
        st.info("Tips: mulai dengan preset **Demo cepat** untuk mencoba tampilan, lalu naikkan replikasi ke 100–1000 untuk hasil yang andal.", icon="💡")
    with c2:
        seksi("Batas kontrol")
        st.latex(r"UCL_F=\frac{p(n-1)(n+1)}{n(n-p)}\,F_{(1-\alpha;\,p,\,n-p)}")
        st.latex(r"CL_{KDE}=\hat F_h^{-1}(1-\alpha),\qquad CL_{Boot}=\tfrac1B\sum_{b=1}^{B}q^*_b(1-\alpha)")
        st.caption("F mengandaikan T² klasik berdistribusi F; KDE dan Bootstrap belajar dari sebaran T² data in-control (Fase I).")
    seksi("Tujuh estimator yang dibandingkan")
    st.dataframe(METODE_INFO, hide_index=True, **W_DF)
    with st.expander("📚 Glosarium metrik"):
        st.markdown(
            "- **Hit rate (accuracy)** — proporsi observasi yang diklasifikasikan benar (normal→normal, outlier→outlier).\n"
            "- **FP rate** — proporsi observasi normal yang salah ditandai outlier (*alarm palsu*); idealnya ≤ α.\n"
            "- **FN rate** — proporsi outlier yang lolos tak terdeteksi; makin kecil makin baik.\n"
            "- **F1 / Balanced accuracy** — ringkasan keseimbangan presisi–recall dan sensitivitas–spesifisitas.\n"
            "- **AUC** — peluang T² sebuah outlier lebih besar daripada T² observasi normal; 1 = pemisahan sempurna, 0,5 = acak. Tidak bergantung pada batas kontrol.\n"
            "- **CL** — nilai batas kontrol; observasi dengan T² > CL menghasilkan alarm.\n"
            "- **Masking** — outlier menarik rata-rata & kovarians estimasi sehingga jarak dirinya sendiri terlihat kecil.")
    if res is not None:
        with st.expander("⚙️ Konfigurasi yang dipakai pada hasil ini"):
            st.dataframe(tabel_cfg(res["cfg"]), hide_index=True, **W_DF)


def page_utama(res):
    cfg, df, rata, sd = res["cfg"], res["df"], res["rata"], res["sd"]
    ests, bs, bu = cfg["est_aktif"], cfg["batas_aktif"], cfg["batas_utama"]
    if int(df["accuracy"].isna().sum()):
        st.warning(f"{int(df['accuracy'].isna().sum())} hasil kosong (estimator gagal pada beberapa dataset); rata-rata dihitung dari replikasi yang berhasil.")
    r = rata.sort_values("accuracy", ascending=False)
    ok = int((rata.fp_rate <= cfg["alpha"]).sum())
    kpi_row([("Hit rate tertinggi", pm(r.index[0]), f"{r.accuracy.iloc[0]:.4f}"),
             ("FP rate terendah", pm(rata.fp_rate.idxmin()), f"{rata.fp_rate.min():.4f}", "#eb6834"),
             ("FN rate terendah", pm(rata.fn_rate.idxmin()), f"{rata.fn_rate.min():.4f}", "#1baf7a"),
             ("Terpenuhi FP ≤ α", f"{ok} / {len(rata)}", f"α = {cfg['alpha']:g}", "#4a3aa7"),
             ("Tercepat", pm(rata.waktu_detik.idxmin()), f"{rata.waktu_detik.min():.4f} dtk/replikasi", "#eda100")])

    seksi("Ringkasan seluruh kombinasi", f"Rata-rata {cfg['reps']} replikasi · huruf tebal hijau = terbaik pada kolom tersebut")
    tab = r.copy()
    tab["FP ≤ α"] = np.where(tab.fp_rate <= cfg["alpha"], "✓", "✗")
    ren = {"accuracy": "Hit rate", "fp_rate": "FP rate", "fn_rate": "FN rate", "sensitivity": "Sensitivity",
           "specificity": "Specificity", "f1": "F1", "balanced_accuracy": "Balanced acc.", "auc": "AUC",
           "waktu_detik": "Waktu (dtk)"}
    tab = tab.rename(columns=ren)
    tab.index = [pm(m) for m in tab.index]
    baik = {ren[c] for c in LEBIH_BESAR_BAIK if c in ren}
    kol_num = [v for v in ren.values()] + ["CL"]

    def tebal(col):
        if col.name not in ren.values() or col.isna().all():
            return [""] * len(col)
        t = col.max() if col.name in baik else col.min()
        return ["font-weight:700;color:#15803d;background:#ecfdf3" if np.isclose(v, t) else "" for v in col]

    st.dataframe(tab.style.apply(tebal).format({c: "{:.4f}" for c in kol_num}), height=min(38 * len(tab) + 40, 800), **W_DF)
    callout("Baris = satu kombinasi estimator–batas (urut dari hit rate tertinggi). <b>Hit rate</b> makin besar makin baik; <b>FP rate</b> harus ≤ α; "
            "<b>FN rate</b> makin kecil makin baik; <b>AUC</b> mengukur daya pisah estimator tanpa memandang batas.", ip_ringkasan(res))

    with st.expander(f"Ringkas per estimator (batas {bu})"):
        rk = rata[[m.split("-")[-1] == bu for m in rata.index]][["accuracy", "fp_rate", "fn_rate", "f1", "auc", "waktu_detik"]]
        rk = rk.sort_values("accuracy", ascending=False).rename(columns=ren)
        rk.index = [m.split("-")[1] for m in rk.index]
        st.dataframe(rk.style.format("{:.4f}"), **W_DF)

    seksi("Matriks konfusi (rata-rata seluruh replikasi)")
    b_cm = st.selectbox("Batas kontrol yang ditampilkan", bs, index=bs.index(bu), key="cm_batas")
    plot(ch.fig_confusion(df, ests, b_cm), key="fig_cm")
    callout("Tiap panel adalah satu estimator. <b>Baris</b> = kondisi aktual, <b>kolom</b> = keputusan diagram. Angka = rata-rata jumlah observasi per replikasi; "
            "persen dihitung per baris. Kiri-atas (TN) &amp; kanan-bawah (TP) = keputusan benar; kanan-atas (FP) = alarm palsu; kiri-bawah (FN) = outlier lolos. "
            "Warna makin biru = persentase baris makin besar.", ip_confusion(res, b_cm))

    seksi("Perbandingan Hit rate, FP rate, dan FN rate", "💡 Klik sebuah batang untuk melihat rincian kombinasi tersebut di bawah grafik. Garis kecil pada batang = ±1 simpangan baku antar replikasi.")
    ev = plot(ch.fig_bar_metrics(rata, sd, ests, bs, cfg["alpha"]), key="fig_bar", select=True)
    pts = titik_terpilih(ev)
    if pts:
        pt = pts[0]
        cd = pt.get("customdata")
        m = cd[0] if isinstance(cd, (list, tuple)) and cd and isinstance(cd[0], str) else None
        if m is None and isinstance(pt.get("x"), str):
            m = f"T2-{pt['x']}-{bs[pt.get('curve_number', 0) % len(bs)]}"
        if m in rata.index:
            x = rata.loc[m]
            d = df[df.metode == m]
            st.markdown(f"**Rincian {pm(m)}**")
            kpi_row([("Hit rate", f"{x.accuracy:.4f}", f"± {sd.loc[m, 'accuracy']:.4f}"),
                     ("FP rate", f"{x.fp_rate:.4f}", "OK ≤ α" if x.fp_rate <= cfg["alpha"] else "melebihi α", "#eb6834"),
                     ("FN rate", f"{x.fn_rate:.4f}", f"± {sd.loc[m, 'fn_rate']:.4f}", "#1baf7a"),
                     ("AUC", f"{x.auc:.4f}", f"CL = {x.CL:.3f}", "#4a3aa7"),
                     ("TP / FN / FP / TN", f"{d.TP.mean():.0f} / {d.FN.mean():.0f} / {d.FP.mean():.1f} / {d.TN.mean():.0f}", "rata-rata per replikasi", "#eda100")])
    else:
        st.caption("Belum ada batang yang dipilih.")
    callout("Tiga panel membandingkan hit rate (tinggi lebih baik), FP rate (garis putus-putus = α, sebaiknya di bawahnya) dan FN rate (rendah lebih baik). "
            "Warna batang = jenis batas kontrol; klik legenda untuk menyembunyikan salah satu batas.", ip_bars(res))


def page_diagram(res):
    cfg, cl = res["cfg"], res["cl"]
    ests, bs, bu = cfg["est_aktif"], cfg["batas_aktif"], cfg["batas_utama"]
    t2s, y, X = res["last"]
    seksi("Diagram kontrol T² interaktif", f"Data dari replikasi terakhir (ke-{cfg['reps']}): {int(y.sum())} outlier aktual dari {len(y)} observasi.")
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 1])
    e = c1.selectbox("Estimator", [x for x in ests if x in t2s], key="cc_e")
    b = c2.selectbox("Batas untuk klasifikasi warna", bs, index=bs.index(bu), key="cc_b")
    mode = c3.radio("Mode", ["Klik & jelajahi", "Animasi pemantauan"], horizontal=True, key="cc_mode")
    log = c4.checkbox("Skala log", value=False, key="cc_log", disabled=mode != "Klik & jelajahi")
    if mode == "Klik & jelajahi":
        st.caption("Klik satu titik, atau gunakan ikon kotak/lasso pada toolbar grafik untuk memilih banyak titik; rinciannya muncul di bawah.")
        ev = plot(ch.fig_control_static(t2s[e], y, cl, e, bs, b, log), key=f"cc_{e}_{b}_{log}", select=True)
        pts = titik_terpilih(ev)
        obs = sorted({int(round(p["x"])) for p in pts if "x" in p})[:300]
        if obs:
            idx = np.array(obs) - 1
            tab = pd.DataFrame({"Observasi": obs, "Aktual": np.where(y[idx], "Outlier", "Normal"), "T²": t2s[e][idx]})
            for bb in bs:
                tab[f"Batas {bb} (CL {cl[f'T2-{e}-{bb}']:.2f})"] = np.where(t2s[e][idx] > cl[f"T2-{e}-{bb}"], "🚨 alarm", "aman")
            st.markdown(f"**{len(obs)} titik terpilih** · {int(y[idx].sum())} outlier aktual · "
                        f"{int((t2s[e][idx] > cl[f'T2-{e}-{b}']).sum())} alarm pada batas {b}")
            st.dataframe(tab.style.format({"T²": "{:.3f}"}), hide_index=True, **W_DF)
    else:
        a1, a2 = st.columns(2)
        nf = a1.slider("Jumlah tahap animasi", 10, 100, 40, 5, key="cc_nf")
        dur = a2.slider("Kecepatan (ms per tahap)", 50, 1000, 150, 50, key="cc_dur")
        plot(ch.fig_control_anim(t2s[e], y, cl, e, bs, b, nf, dur), key=f"cca_{e}_{b}_{nf}_{dur}")
    callout("Sumbu-x = nomor observasi, sumbu-y = statistik T². Garis putus-putus = batas kontrol (F, KDE, Bootstrap); titik di atas garis memicu alarm. "
            "Warna menunjukkan hasil klasifikasi: abu-abu = normal benar, merah = outlier terdeteksi, oranye ◆ = outlier lolos, biru ✕ = alarm palsu. "
            "Mode animasi memutar ulang pemantauan observasi demi observasi.", ip_control(t2s[e], y, cl, e, bs, b))

    with st.expander("Lihat semua estimator sekaligus"):
        plot(ch.fig_control_all(t2s, y, cl, [x for x in ests if x in t2s], bs), key="cc_all")

    seksi("Dari mana batas kontrol berasal? (Fase I)", "Sebaran T² data in-control dan letak masing-masing batas.")
    fi = [x for x in ests if x in res["faseI"]]
    if fi:
        ef = st.selectbox("Estimator", fi, key="fi_e")
        plot(ch.fig_hist_fase1(res["faseI"][ef], cl, ef, bs, cfg["alpha"]), key=f"fi_{ef}")
        callout("Batang biru muda = histogram T² dari dataset bersih (Fase I), kurva gelap = estimasi kepadatan kernel. Garis vertikal = batas kontrol; "
                "luas ekor di kanan garis seharusnya sekitar α. Arahkan kursor ke penanda untuk melihat persentase data Fase I di atas batas.",
                ip_hist(res["faseI"][ef], cl, ef, bs, cfg["alpha"]))
    else:
        st.info("Batas KDE/Bootstrap tidak dipilih sehingga tidak ada data Fase I. Aktifkan salah satunya di panel konfigurasi.")

    seksi("Sebaran T² observasi normal vs outlier")
    plot(ch.fig_dist_t2(t2s, y, cl, [x for x in ests if x in t2s], b), key="dist_box")
    callout("Kotak abu-abu = T² observasi normal, kotak merah = T² outlier (skala log). Penanda hitam = batas kontrol terpilih. "
            "Diagram yang baik memiliki kotak merah jauh di atas penanda, dan kotak abu-abu di bawahnya.", ip_dist(t2s, y, cl, [x for x in ests if x in t2s], b))

    seksi("Perbandingan nilai batas kontrol")
    plot(ch.fig_cl_compare(cl, ests, bs), key="cl_cmp")
    callout("Tinggi batang = nilai CL. Jika suatu estimator memiliki CL jauh lebih tinggi dari yang lain, T² Fase I miliknya besar sehingga batas menjadi lebih longgar.",
            ip_cl(cl, ests, bs))


def page_lanjut(res):
    cfg, df, rata, cl = res["cfg"], res["df"], res["rata"], res["cl"]
    ests, bs, bu = cfg["est_aktif"], cfg["batas_aktif"], cfg["batas_utama"]
    t2s, y, X = res["last"]
    ests_t = [e for e in ests if e in t2s]
    opsi = ["Kurva ROC", "Stabilitas replikasi", "Profil multi-metrik", "Peta panas kinerja", "Efisiensi waktu",
            "Konvergensi simulasi", "Proyeksi 2D (animasi)", "DD-plot"]
    st.session_state.setdefault("sub", opsi[0])
    pilih = nav_widget(opsi, "sub") or opsi[0]
    st.write("")

    if pilih == "Kurva ROC":
        c1, c2 = st.columns([1, 1])
        b = c1.selectbox("Batas untuk titik kerja (◆)", bs, index=bs.index(bu), key="roc_b")
        zoom = c2.checkbox("Perbesar sudut kiri-atas", value=False, key="roc_zoom")
        plot(ch.fig_roc(t2s, y, cl, ests_t, b, zoom), key=f"roc_{b}_{zoom}")
        callout("Kurva ROC menelusuri semua kemungkinan batas: sumbu-x = alarm palsu (FPR), sumbu-y = outlier terdeteksi (TPR). Kurva yang menempel ke sudut kiri-atas = pemisahan hampir sempurna; "
                "diagonal putus-putus = tebakan acak. Belah ketupat (◆) = posisi diagram pada batas yang dipilih. Klik legenda untuk menyembunyikan estimator.",
                ip_roc(t2s, y, cl, ests_t, b))
    elif pilih == "Stabilitas replikasi":
        c1, c2 = st.columns(2)
        met = c1.selectbox("Metrik", list(MET), format_func=lambda k: MET[k][0], key="st_m")
        b = c2.selectbox("Batas kontrol", bs, index=bs.index(bu), key="st_b")
        plot(ch.fig_box_replikasi(df, ests, b, met, MET[met][0]), key=f"box_{met}_{b}")
        callout("Setiap kotak merangkum nilai metrik dari seluruh replikasi: garis tengah = median, ◆ = rata-rata, tinggi kotak = rentang antar-kuartil, titik di luar = pencilan. "
                "Kotak pendek berarti hasil konsisten antar dataset.", ip_stab(df, ests, b, met, MET[met][0]))
    elif pilih == "Profil multi-metrik":
        b = pilih_batas(res, "rd_b")
        plot(ch.fig_radar(rata, ests, b), key=f"radar_{b}")
        callout("Setiap sumbu adalah satu metrik (semuanya: makin ke luar makin baik; FP dan FN dibalik menjadi 1 − rate). Poligon yang luas dan membulat = estimator yang baik di semua aspek. "
                "Klik nama estimator di legenda untuk menampilkan/menyembunyikannya.", ip_radar(rata, ests, b))
    elif pilih == "Peta panas kinerja":
        plot(ch.fig_heatmap_metrics(rata, ests, bs), key="heat")
        callout("Baris = kombinasi estimator–batas, kolom = metrik. Angka asli ditampilkan pada sel, sedangkan warna menunjukkan peringkat relatif dalam kolom: "
                "hijau = terbaik, merah = terburuk (untuk FP, FN, dan waktu, nilai kecil diberi warna hijau).", ip_heat(rata, ests, bs))
    elif pilih == "Efisiensi waktu":
        c1, c2 = st.columns(2)
        b = c1.selectbox("Batas kontrol", bs, index=bs.index(bu), key="tm_b")
        ym = c2.selectbox("Metrik kualitas", ["accuracy", "f1", "auc", "balanced_accuracy"], format_func=lambda k: MET[k][0], key="tm_m")
        plot(ch.fig_time_acc(rata, ests, b, ym, MET[ym][0]), key=f"time_{b}_{ym}")
        callout("Sumbu-x = waktu komputasi per replikasi (log; makin kiri makin cepat), sumbu-y = kualitas (makin atas makin baik). Titik ideal ada di kiri-atas. "
                "Garis putus-putus (Pareto) menghubungkan estimator yang tidak terkalahkan pada kedua sisi sekaligus.", ip_time(rata, ests, b, ym, MET[ym][0]))
    elif pilih == "Konvergensi simulasi":
        c1, c2 = st.columns(2)
        b = c1.selectbox("Batas kontrol", bs, index=bs.index(bu), key="cv_b")
        met = c2.selectbox("Metrik", ["accuracy", "fp_rate", "fn_rate", "f1", "auc"], format_func=lambda k: MET[k][0], key="cv_m")
        plot(ch.fig_convergence(df, ests, b, met, MET[met][0]), key=f"conv_{b}_{met}")
        callout("Garis menunjukkan rata-rata kumulatif metrik ketika replikasi bertambah satu per satu. Di awal garis bergoyang; jika sudah mendatar, "
                "jumlah replikasi telah cukup dan kesimpulan tidak lagi berubah oleh tambahan iterasi.", ip_conv(df, ests, b, met, MET[met][0]))
    elif pilih == "Proyeksi 2D (animasi)":
        b = pilih_batas(res, "pc_b")
        dur = st.slider("Kecepatan animasi (ms per estimator)", 500, 3000, 1400, 100, key="pc_d")
        plot(ch.fig_pca_anim(X, y, t2s, cl, ests_t, b, dur), key=f"pca_{b}_{dur}")
        callout("Data dua dimensi hasil PCA (dua komponen utama). Warna = hasil klasifikasi tiap observasi oleh estimator tertentu; tekan ▶ Putar atau geser slider untuk berpindah antar-estimator. "
                "Sumbu menampilkan persentase variansi yang dijelaskan.", ip_pca(t2s, y, cl, ests_t, b))
        tab = pd.DataFrame({e: dict(zip(["TN", "FP", "FN", "TP"], [int((~y & ~(t2s[e] > cl[f"T2-{e}-{b}"])).sum()), int((~y & (t2s[e] > cl[f"T2-{e}-{b}"])).sum()),
                                                                  int((y & ~(t2s[e] > cl[f"T2-{e}-{b}"])).sum()), int((y & (t2s[e] > cl[f"T2-{e}-{b}"])).sum())])) for e in ests_t}).T
        st.dataframe(tab, **W_DF)
    else:
        c1, c2 = st.columns(2)
        e = c1.selectbox("Estimator", ests_t, key="dd_e")
        b = c2.selectbox("Batas kontrol", bs, index=bs.index(bu), key="dd_b")
        fig, stt = ch.fig_dd(X, t2s[e], y, e, cl[f"T2-{e}-{b}"], cfg["alpha"], b)
        plot(fig, key=f"dd_{e}_{b}")
        callout("Sumbu-x = jarak Mahalanobis klasik, sumbu-y = jarak robust. Garis hijau vertikal = batas klasik (χ²); garis horizontal = batas robust. "
                "Outlier yang berada di <b>atas garis horizontal</b> tetapi di <b>kiri garis hijau</b> adalah outlier yang tersamarkan oleh metode klasik namun terungkap oleh estimator robust.",
                ip_dd(stt, e, b))


def page_grid(res):
    cfg = ambil_cfg()
    ests, bs = cfg["est_aktif"], cfg["batas_aktif"]
    seksi("Simulasi grid p × ε", "Mereplikasi format Tabel 4–9 paper: hit rate, FP rate, dan FN rate untuk tiap kombinasi jumlah variabel (p) dan proporsi outlier (ε).")
    st.markdown('<div class="card"><b class="h">Konfigurasi grid</b> — parameter lain (n, μ, α, seed, estimator, batas) mengikuti panel kiri.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    gp = c1.multiselect("Nilai p (jumlah variabel)", [2, 3, 5, 10, 15, 20, 30, 40, 50], key="g_p")
    ge = c2.multiselect("Nilai ε (% outlier)", [1, 5, 10, 15, 20, 30, 40, 45, 49], key="g_eps")
    gr = c3.slider("Replikasi per skenario", 10, 1000, step=10, key="g_reps", help="Bawaan 100, maksimum 1000.")
    sc = len(gp) * len(ge)
    galat = cfg_valid(cfg) or ("Pilih minimal satu nilai p dan ε." if sc == 0 else None)
    if not galat and cfg["n"] < 5 * max(gp):
        galat = f"n = {cfg['n']} terlalu kecil untuk p = {max(gp)} (syarat n ≥ 5p). Naikkan n atau kurangi p."
    est_txt = ""
    if res is not None:
        te = res["df"].groupby("estimator").waktu_detik.mean()
        per = float(sum(te.get(e, 0) for e in ests))
        det = sc * gr * per / cfg["n_jobs"]
        if {"KDE", "Boot"} & set(bs):
            det += len(gp) * cfg["n_ref"] * per / cfg["n_jobs"]
        est_txt = f" · perkiraan kasar ≈ {det / 60:.1f} menit (berdasarkan waktu skenario utama)"
    st.caption(f"{sc} skenario × {gr} replikasi × {len(ests)} estimator{est_txt}")
    if galat:
        st.error(galat)
    slot = st.container()
    if st.button("▶ Jalankan Simulasi Grid", type="primary", disabled=bool(galat), key="run_grid"):
        try:
            with slot:
                g = jalankan_grid(cfg, gp, ge, gr, st.container())
            st.session_state["grid"] = g
        except Exception:
            st.error("Terjadi kesalahan saat menjalankan grid.")
            st.code(traceback.format_exc())
    g = st.session_state.get("grid")
    if not g:
        st.info("Belum ada hasil grid. Atur nilai p dan ε di atas, lalu klik tombol jalankan.")
        return
    grid, cg = g["df"], g["cfg"]
    P, EPS = g["P"], g["EPS"]
    est_g = cg["est_aktif"]
    bs_g = cg["batas_aktif"]
    bu_g = cg["batas_utama"]
    st.markdown(f'<span class="status st-ok">● Grid: p ∈ {P} · ε ∈ {EPS}% · {g["reps"]} replikasi/skenario · {g["waktu"]:.0f} dtk</span>', unsafe_allow_html=True)
    if min(EPS) < max(EPS):
        eps_max = st.select_slider("Rata-ratakan untuk ε ≤ (%)", EPS, value=max(EPS), key=f"g_epsmax_{min(EPS)}_{max(EPS)}_{len(EPS)}",
                                   help="Pada paper, ε = 50% selalu gagal untuk semua metode sehingga sering dikecualikan dari rata-rata.")
    else:
        eps_max = max(EPS)
    ring = core.ringkas_grid(grid, cg["alpha"], eps_max)
    ring.index = ring.index.astype(str)

    seksi("Peringkat seluruh kombinasi")
    tb = ring.copy()
    tb.insert(1, "Kombinasi", [pm(m) for m in tb.index])
    st.dataframe(tb.reset_index(drop=True), hide_index=True, column_config={
        "Peringkat": st.column_config.NumberColumn(width="small"),
        "Hit rate": st.column_config.ProgressColumn("Hit rate", min_value=float(max(0, ring["Hit rate"].min() - 0.05)), max_value=1.0, format="%.4f"),
        "FP rate": st.column_config.NumberColumn(format="%.4f"), "FN rate": st.column_config.NumberColumn(format="%.4f"),
        "FP maks": st.column_config.NumberColumn(format="%.4f"), "Waktu (dtk)": st.column_config.NumberColumn(format="%.4f")}, **W_DF)
    ev = plot(ch.fig_rank_bar(ring), key="rank_bar", select=True)
    callout("Batang = hit rate rata-rata pada seluruh skenario p × ε (warna = estimator). Klik batang untuk memilih kombinasi pada grafik-grafik di bawah. "
            "Kolom 'Skenario FP > α' menghitung berapa skenario yang melanggar batas alarm palsu.", ip_rank(ring))
    pts = titik_terpilih(ev)
    if pts:
        cd = pts[0].get("customdata")
        if isinstance(cd, (list, tuple)) and cd and isinstance(cd[0], str) and cd[0] in ring.index:
            st.session_state["grid_pick"] = cd[0]
    metode_grid = [f"T2-{e}-{b}" for e in est_g for b in bs_g]
    pick = st.session_state.get("grid_pick")
    if pick not in metode_grid:
        pick = ring.index[0]

    seksi("Peta panas p × ε")
    c1, c2 = st.columns(2)
    mm = c1.selectbox("Kombinasi", metode_grid, index=metode_grid.index(pick), format_func=pm, key="gh_m")
    mt = c2.selectbox("Metrik", ["accuracy", "fp_rate", "fn_rate", "f1", "auc"], format_func=lambda k: MET[k][0], key="gh_t")
    lab, naik = MET[mt]
    plot(ch.fig_grid_heat(grid, mm, mt, lab, naik, P, EPS), key=f"gh_{mm}_{mt}")
    callout("Tiap sel = rata-rata metrik untuk kombinasi (p, ε). Baris ke bawah = dimensi bertambah; kolom ke kanan = outlier bertambah. Hijau = baik, merah = buruk.",
            ip_grid_heat(grid, mm, mt, lab, naik))

    seksi("Animasi kinerja antar-skenario", "Tekan ▶ Putar atau geser slider untuk melihat bagaimana kurva estimator berubah ketika p (atau ε) berganti.")
    c1, c2, c3 = st.columns(3)
    ba = c1.selectbox("Batas kontrol", bs_g, index=bs_g.index(bu_g), key="ga_b")
    ma = c2.selectbox("Metrik", ["accuracy", "fp_rate", "fn_rate", "f1", "auc"], format_func=lambda k: MET[k][0], key="ga_m")
    by = c3.radio("Animasikan berdasarkan", ["p", "ε"], horizontal=True, key="ga_by")
    la, na = MET[ma]
    gg = grid.copy()
    gg["estimator"] = gg.metode.map(lambda m: m.split("-")[1])
    plot(ch.fig_grid_anim(gg, est_g, ba, ma, la, P, EPS, "p" if by == "p" else "eps", 900), key=f"ga_{ba}_{ma}_{by}")
    callout("Setiap garis = satu estimator. Pada mode 'p' sumbu-x adalah proporsi outlier ε dan setiap tahap animasi adalah satu nilai p; pada mode 'ε' sebaliknya.",
            ip_grid_anim(gg, est_g, ba, ma, la, na))

    seksi("Permukaan 3D")
    c1, c2 = st.columns(2)
    m3 = c1.selectbox("Kombinasi", metode_grid, index=metode_grid.index(pick), format_func=pm, key="g3_m")
    t3 = c2.selectbox("Metrik", ["accuracy", "fp_rate", "fn_rate", "f1", "auc"], format_func=lambda k: MET[k][0], key="g3_t")
    if len(P) >= 2 and len(EPS) >= 2:
        plot(ch.fig_grid_surface(grid, m3, t3, MET[t3][0], P, EPS), key=f"g3_{m3}_{t3}")
        callout("Permukaan yang sama dengan peta panas, tetapi tinggi = nilai metrik. Putar dengan menyeret mouse, zoom dengan scroll. Lembah yang dalam menunjukkan kondisi (p, ε) yang sulit bagi kombinasi ini.",
                ip_grid_heat(grid, m3, t3, MET[t3][0], MET[t3][1]))
    else:
        st.info("Permukaan 3D membutuhkan minimal 2 nilai p dan 2 nilai ε.")

    seksi("FP rate rata-rata")
    plot(ch.fig_grid_fp(grid, est_g, bs_g, cg["alpha"], eps_max), key="g_fp")
    callout("Batang = rata-rata FP rate seluruh skenario; garis putus-putus = α. Batang di bawah garis berarti alarm palsu terkendali.",
            ip_grid_fp(grid, est_g, bs_g, cg["alpha"], eps_max))

    with st.expander("Tabel format paper (Hit rate / FP rate / FN rate)"):
        mt_ = st.selectbox("Kombinasi", metode_grid, format_func=pm, key="gt_m")
        blok = [EPS[i:i + 3] for i in range(0, len(EPS), 3)]
        for bb in blok:
            st.dataframe(core.tabel_paper(grid, mt_, bb, P).style.format("{:.4f}"), **W_DF)


def page_unduh(res):
    cfg = res["cfg"]
    seksi("Unduh hasil")
    g = st.session_state.get("grid")
    ring = core.ringkas_grid(g["df"], g["cfg"]["alpha"]) if g else None
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><b class="h">Workbook Excel</b><br>Ringkasan skenario utama, simpangan baku, batas kontrol, data per replikasi'
                    + (", serta seluruh hasil grid." if g else ".") + "</div>", unsafe_allow_html=True)
        xls = core.buat_excel(res, g["df"] if g else None, ring, g["P"] if g else None, g["EPS"] if g else None, tabel_cfg(cfg))
        st.download_button("⬇ Unduh Excel (.xlsx)", xls, file_name=f"hasil_t2_{cfg['dist']}_n{cfg['n']}_p{cfg['p']}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **W_DL)
    with c2:
        st.markdown('<div class="card"><b class="h">CSV per replikasi</b><br>Seluruh metrik setiap replikasi untuk analisis lanjutan di R/Python.</div>', unsafe_allow_html=True)
        st.download_button("⬇ Unduh CSV replikasi", res["df"].to_csv(index=False).encode("utf-8"),
                           file_name="per_replikasi.csv", mime="text/csv", **W_DL)
    with c3:
        st.markdown('<div class="card"><b class="h">CSV grid</b><br>Hasil rata-rata tiap skenario p × ε.</div>', unsafe_allow_html=True)
        if g:
            st.download_button("⬇ Unduh CSV grid", g["df"].to_csv(index=False).encode("utf-8"), file_name="grid_data.csv",
                               mime="text/csv", **W_DL)
        else:
            st.caption("Belum ada hasil grid.")
    seksi("Konfigurasi yang dipakai")
    st.dataframe(tabel_cfg(cfg), hide_index=True, **W_DF)


# ======================================================================================
# Main
# ======================================================================================
def main():
    run = sidebar()
    res = st.session_state.get("res")
    header(res)
    slot = st.empty()
    if run:
        cfg = ambil_cfg()
        selesai = False
        try:
            with slot.container():
                res = jalankan_utama(cfg, st.container())
            st.session_state["res"] = res
            st.session_state["nav"] = NAV[1]
            st.session_state.pop("grid_pick", None)
            selesai = True
        except Exception:
            st.error("Terjadi kesalahan saat menjalankan simulasi.")
            st.code(traceback.format_exc())
        if selesai:
            slot.empty()
            st.rerun()
    halaman = nav_widget(NAV, "nav") or NAV[0]
    st.write("")
    if halaman == NAV[0]:
        page_beranda(res)
    elif halaman == NAV[4]:
        page_grid(res)
    elif res is None:
        st.info("Halaman ini membutuhkan hasil simulasi. Klik **▶ Jalankan Simulasi** di panel kiri terlebih dahulu.", icon="👈")
    elif halaman == NAV[1]:
        page_utama(res)
    elif halaman == NAV[2]:
        page_diagram(res)
    elif halaman == NAV[3]:
        page_lanjut(res)
    elif halaman == NAV[5]:
        page_unduh(res)


main()
