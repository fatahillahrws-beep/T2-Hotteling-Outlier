"""
charts.py — Pembuat grafik Plotly (interaktif, dapat diklik, dan beranimasi)
===========================================================================
Semua fungsi mengembalikan `plotly.graph_objects.Figure` sehingga dapat ditampilkan dengan
`st.plotly_chart`. Grafik beranimasi memakai `frames` + tombol ▶ Putar / ⏸ Jeda + slider.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from core import WARNA_BATAS, WARNA_EST, auc_score, mahal2, roc_points

FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
C_TN, C_TP, C_FN, C_FP = "#94a3b8", "#e34948", "#f59e0b", "#2a78d6"
GRID = "#e8edf4"

GRUP = [  # (label, warna, simbol)
    ("Normal & lolos (TN)", C_TN, "circle"),
    ("Outlier terdeteksi (TP)", C_TP, "circle"),
    ("Outlier lolos (FN)", C_FN, "diamond"),
    ("Alarm palsu (FP)", C_FP, "x"),
]


def pm(m: str) -> str:
    return m.replace("T2-", "T²-")


def _base(fig, title=None, h=440, legend=True):
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=13, color="#1f2937"),
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)) if title else None,
        height=h,
        margin=dict(l=55, r=25, t=85 if title else 45, b=55),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT, bordercolor="#cbd5e1"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    fig.update_xaxes(showline=True, linecolor="#cbd5e1", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showline=True, linecolor="#cbd5e1", gridcolor=GRID, zeroline=False)
    return fig


def _outcome_masks(y, pred):
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    return [~y & ~pred, y & pred, y & ~pred, ~y & pred]  # TN, TP, FN, FP


def _anim_controls(fig, frame_names, step_labels, dur=150, y_pos=-0.16, prefix=""):
    """Tambahkan tombol Putar/Jeda/Ulang + slider ke figure yang sudah punya frames."""
    play = dict(frame=dict(duration=dur, redraw=True), transition=dict(duration=0),
                fromcurrent=True, mode="immediate")
    stop = dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=y_pos, xanchor="left", yanchor="top",
            pad=dict(t=6, r=6),
            buttons=[
                dict(label="▶ Putar", method="animate", args=[None, play]),
                dict(label="⏸ Jeda", method="animate", args=[[None], stop]),
                dict(label="↺ Ulang", method="animate",
                     args=[[frame_names[0]], dict(frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0), mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, x=0.28, y=y_pos, len=0.72, xanchor="left", yanchor="top", pad=dict(t=6, b=0),
            currentvalue=dict(prefix=prefix, font=dict(size=12), visible=True, xanchor="left"),
            steps=[dict(method="animate", label=str(lab),
                        args=[[nm], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))])
                   for nm, lab in zip(frame_names, step_labels)])],
    )
    return fig


# ======================================================================================
# Skenario utama
# ======================================================================================
def fig_confusion(df, ests, batas):
    k = len(ests)
    ncol = min(4, k)
    nrow = math.ceil(k / ncol)
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[f"T²-{e}-{batas}" for e in ests],
                        horizontal_spacing=0.07, vertical_spacing=0.17 if nrow > 1 else 0.1)
    kelas = ["Normal", "Outlier"]
    lab = [["TN", "FP"], ["FN", "TP"]]
    ket = [["benar: normal dikenali normal", "alarm palsu: normal ditandai outlier"],
           ["outlier lolos: dianggap normal", "benar: outlier terdeteksi"]]
    for i, e in enumerate(ests):
        d = df[df.metode == f"T2-{e}-{batas}"]
        M = d[["TN", "FP", "FN", "TP"]].mean().values.reshape(2, 2)
        pct = M / M.sum(1, keepdims=True) * 100
        txt = [[f"<b>{lab[a][b]}</b><br>{M[a, b]:,.1f}<br>({pct[a, b]:.1f}%)" for b in range(2)] for a in range(2)]
        hov = [[f"<b>{lab[a][b]}</b> — {ket[a][b]}<br>Rata-rata {M[a, b]:,.2f} observasi/replikasi"
                f"<br>{pct[a, b]:.2f}% dari baris aktual '{kelas[a]}'" for b in range(2)] for a in range(2)]
        r, c = divmod(i, ncol)
        fig.add_trace(go.Heatmap(
            z=pct, x=kelas, y=kelas, text=txt, texttemplate="%{text}", hovertext=hov, hoverinfo="text",
            colorscale=[[0, "#f8fafc"], [1, "#2a78d6"]], zmin=0, zmax=100, showscale=False,
            xgap=3, ygap=3, textfont=dict(size=13)), row=r + 1, col=c + 1)
        fig.update_yaxes(autorange="reversed", row=r + 1, col=c + 1,
                         title_text="Aktual" if c == 0 else None)
        fig.update_xaxes(title_text="Prediksi", row=r + 1, col=c + 1)
    fig.update_annotations(font=dict(size=13, family=FONT))
    _base(fig, h=340 * nrow + 40, legend=False)
    fig.update_layout(margin=dict(l=60, r=20, t=50, b=50))
    return fig


def fig_bar_metrics(rata, sd, ests, bataslist, alpha):
    """3 panel: Hit rate, FP rate, FN rate. Batang dapat diklik (drill-down)."""
    judul = ["Hit rate (akurasi)", "FP rate (alarm palsu)", "FN rate (outlier lolos)"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=judul, horizontal_spacing=0.07)
    for j, met in enumerate(["accuracy", "fp_rate", "fn_rate"], 1):
        for b in bataslist:
            ms = [f"T2-{e}-{b}" for e in ests]
            fig.add_trace(go.Bar(
                x=ests, y=[rata.loc[m, met] for m in ms], name=f"Batas {b}", legendgroup=b,
                showlegend=(j == 1), marker_color=WARNA_BATAS[b], opacity=0.92,
                error_y=dict(type="data", array=[sd.loc[m, met] for m in ms], color="#475569", thickness=1,
                             width=3),
                customdata=[[m] for m in ms],
                hovertemplate="<b>%{customdata[0]}</b><br>" + judul[j - 1] + ": %{y:.4f}<extra></extra>"),
                row=1, col=j)
    fig.add_hline(y=alpha, line_dash="dash", line_color="#111827", line_width=1, row=1, col=2,
                  annotation_text="α", annotation_position="top left")
    lo = max(0.0, float(rata.accuracy.min()) - 0.03)
    fig.update_yaxes(range=[lo, 1.0], row=1, col=1)
    fig.update_layout(barmode="group", bargap=0.2, clickmode="event+select")
    _base(fig, h=430)
    fig.update_layout(margin=dict(l=55, r=20, t=90, b=55))
    return fig


# ======================================================================================
# Diagram kontrol
# ======================================================================================
def fig_control_static(t2, y, cl, e, bataslist, batas_utama, log=False):
    n = len(t2)
    obs = np.arange(1, n + 1)
    pred = t2 > cl[f"T2-{e}-{batas_utama}"]
    fig = go.Figure()
    for (lab, col, sym), mk in zip(GRUP, _outcome_masks(y, pred)):
        fig.add_trace(go.Scatter(
            x=obs[mk], y=t2[mk], mode="markers", name=f"{lab} · {int(mk.sum())}",
            marker=dict(color=col, size=7 if sym != "x" else 8, symbol=sym, opacity=0.85,
                        line=dict(width=0.5, color="white") if sym == "circle" else None),
            hovertemplate="Observasi #%{x}<br>T² = %{y:.3f}<extra>" + lab + "</extra>"))
    for b in bataslist:
        c = cl[f"T2-{e}-{b}"]
        fig.add_hline(y=c, line_color=WARNA_BATAS[b], line_dash="dash" if b != "F" else "dot", line_width=2,
                      annotation_text=f"{b}: {c:.2f}", annotation_position="top right",
                      annotation_font=dict(color=WARNA_BATAS[b], size=12))
    _base(fig, title=f"Diagram kontrol T²-{e} · klasifikasi memakai batas {batas_utama}", h=500)
    fig.update_layout(clickmode="event+select", dragmode="zoom", margin=dict(l=55, r=70, t=100, b=55))
    fig.update_xaxes(title_text="Nomor observasi")
    fig.update_yaxes(title_text="Statistik T²", type="log" if log else "linear")
    return fig


def fig_control_anim(t2, y, cl, e, bataslist, batas_utama, n_frames=40, dur=150):
    """Simulasi pemantauan bertahap: observasi muncul satu per satu, hitung alarm berjalan."""
    n = len(t2)
    obs = np.arange(1, n + 1)
    pred = t2 > cl[f"T2-{e}-{batas_utama}"]
    masks = _outcome_masks(y, pred)
    marks = sorted(set(int(round(n * k / n_frames)) for k in range(1, n_frames + 1)))

    def data_upto(m):
        return [go.Scatter(x=obs[mk & (obs <= m)], y=t2[mk & (obs <= m)]) for mk in masks]

    def judul(m):
        s = [int((mk & (obs <= m)).sum()) for mk in masks]
        return (f"Observasi 1–{m} dari {n} · alarm = {s[1] + s[3]} · "
                f"outlier lolos (FN) = {s[2]} · alarm palsu (FP) = {s[3]}")

    fig = go.Figure(data=[
        go.Scatter(x=obs[mk & (obs <= marks[0])], y=t2[mk & (obs <= marks[0])], mode="markers",
                   name=lab, marker=dict(color=col, size=7, symbol=sym, opacity=0.85),
                   hovertemplate="Observasi #%{x}<br>T² = %{y:.3f}<extra>" + lab + "</extra>")
        for (lab, col, sym), mk in zip(GRUP, masks)])
    fig.frames = [go.Frame(data=data_upto(m), traces=[0, 1, 2, 3], name=str(m),
                           layout=go.Layout(title=dict(text=judul(m), x=0.0, xanchor="left", font=dict(size=15))))
                  for m in marks]
    for b in bataslist:
        c = cl[f"T2-{e}-{b}"]
        fig.add_hline(y=c, line_color=WARNA_BATAS[b], line_dash="dash" if b != "F" else "dot", line_width=2,
                      annotation_text=f"{b}: {c:.2f}", annotation_position="top right",
                      annotation_font=dict(color=WARNA_BATAS[b], size=12))
    ymax = float(max(t2.max(), max(cl[f"T2-{e}-{b}"] for b in bataslist))) * 1.08
    _base(fig, title=judul(marks[0]), h=560)
    fig.update_layout(margin=dict(l=55, r=70, t=100, b=120))
    fig.update_xaxes(title_text="Nomor observasi", range=[0, n + 1])
    fig.update_yaxes(title_text="Statistik T²", range=[0, ymax])
    _anim_controls(fig, [str(m) for m in marks], marks, dur=dur, y_pos=-0.17, prefix="Observasi ke-")
    return fig


def fig_control_all(t2_akhir, y_akhir, cl, ests, bataslist):
    n = len(y_akhir)
    obs = np.arange(1, n + 1)
    fig = make_subplots(rows=len(ests), cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        subplot_titles=[f"T²-{e}" for e in ests])
    for i, e in enumerate(ests, 1):
        t2 = t2_akhir[e]
        for lab, mk, col in [("Normal (aktual)", ~y_akhir, C_TN), ("Outlier (aktual)", y_akhir, C_TP)]:
            fig.add_trace(go.Scattergl(x=obs[mk], y=t2[mk], mode="markers", name=lab, legendgroup=lab,
                                       showlegend=(i == 1), marker=dict(color=col, size=4, opacity=0.8),
                                       hovertemplate="#%{x} · T² = %{y:.2f}<extra></extra>"), row=i, col=1)
        for b in bataslist:
            c = cl[f"T2-{e}-{b}"]
            fig.add_trace(go.Scatter(x=[1, n], y=[c, c], mode="lines", name=f"Batas {b}", legendgroup=b,
                                     showlegend=(i == 1), line=dict(color=WARNA_BATAS[b], width=1.8,
                                                                     dash="dash" if b != "F" else "dot"),
                                     hovertemplate=f"{b}: CL = {c:.3f}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(title_text="T²", row=i, col=1)
    fig.update_xaxes(title_text="Nomor observasi", row=len(ests), col=1)
    _base(fig, h=max(420, 240 * len(ests)))
    fig.update_layout(margin=dict(l=55, r=20, t=70, b=55))
    fig.update_annotations(font=dict(size=13, family=FONT), xanchor="left", x=0.0)
    return fig


def fig_hist_fase1(t2, cl, e, bataslist, alpha):
    """Histogram T² Fase I + kurva KDE + garis batas kontrol."""
    t2 = np.asarray(t2)
    hi = float(max([cl[f"T2-{e}-{b}"] for b in bataslist] + [np.quantile(t2, 0.999)])) * 1.15
    dens, edges = np.histogram(t2[t2 <= hi], bins=80, range=(0, hi), density=True)
    ctr = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ctr, y=dens, name="Histogram T² Fase I", marker_color="#cfe0f7",
                         width=(edges[1] - edges[0]) * 0.95,
                         hovertemplate="T² ≈ %{x:.2f}<br>kepadatan %{y:.4f}<extra></extra>"))
    rng = np.random.default_rng(0)
    sub = rng.choice(t2, min(3000, len(t2)), replace=False)
    sd = t2.std(ddof=1)
    iqr = np.subtract(*np.percentile(t2, [75, 25]))
    h = 0.9 * (min(sd, iqr / 1.34) if iqr > 0 else sd) * len(t2) ** (-0.2)
    xs = np.linspace(0, hi, 300)
    kde = stats.norm.pdf((xs[:, None] - sub[None, :]) / h).mean(1) / h
    fig.add_trace(go.Scatter(x=xs, y=kde, mode="lines", name="KDE Gaussian (Silverman)",
                             line=dict(color="#0f2a5c", width=2.4)))
    for b in bataslist:
        c = cl[f"T2-{e}-{b}"]
        emp = float((t2 > c).mean())
        fig.add_vline(x=c, line_color=WARNA_BATAS[b], line_dash="dash" if b != "F" else "dot", line_width=2.2)
        fig.add_trace(go.Scatter(x=[c], y=[float(kde.max()) * (0.95 - 0.12 * bataslist.index(b))],
                                 mode="markers+text", text=[f" {b}: {c:.2f}"], textposition="middle right",
                                 marker=dict(color=WARNA_BATAS[b], size=9), name=f"Batas {b}",
                                 hovertemplate=f"Batas {b}<br>CL = {c:.3f}<br>T² Fase I di atas CL: "
                                               f"{100 * emp:.3f}% (target α = {100 * alpha:.3f}%)<extra></extra>"))
    _base(fig, title=f"Distribusi T² Fase I (data in-control) — T²-{e}", h=440)
    fig.update_layout(bargap=0)
    fig.update_xaxes(title_text="Statistik T²")
    fig.update_yaxes(title_text="Kepadatan")
    return fig


def fig_dist_t2(t2_akhir, y_akhir, cl, ests, batas_utama):
    fig = go.Figure()
    for lab, mk, col in [("Normal (aktual)", ~y_akhir, C_TN), ("Outlier (aktual)", y_akhir, C_TP)]:
        xs, ys = [], []
        for e in ests:
            xs += [e] * int(mk.sum())
            ys += list(t2_akhir[e][mk])
        fig.add_trace(go.Box(x=xs, y=ys, name=lab, marker_color=col, boxpoints=False, line=dict(width=1.5),
                             hoverinfo="y+name"))
    fig.add_trace(go.Scatter(
        x=ests, y=[cl[f"T2-{e}-{batas_utama}"] for e in ests], mode="markers", name=f"CL batas {batas_utama}",
        marker=dict(symbol="line-ew", size=52, line=dict(width=3.5, color="#111827")),
        hovertemplate="%{x}<br>CL = %{y:.3f}<extra></extra>"))
    _base(fig, title="Sebaran T² observasi normal vs outlier (skala log)", h=470)
    fig.update_layout(boxmode="group")
    fig.update_yaxes(type="log", title_text="T² (skala log)")
    return fig


def fig_cl_compare(cl, ests, bataslist):
    fig = go.Figure()
    for b in bataslist:
        v = [cl[f"T2-{e}-{b}"] for e in ests]
        fig.add_trace(go.Bar(x=ests, y=v, name=f"Batas {b}", marker_color=WARNA_BATAS[b],
                             text=[f"{a:.1f}" for a in v], textposition="outside",
                             hovertemplate="%{x}<br>CL = %{y:.3f}<extra>" + b + "</extra>"))
    _base(fig, title="Nilai batas kontrol (CL) tiap estimator", h=420)
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(title_text="Nilai CL")
    return fig


# ======================================================================================
# Analisis lanjutan
# ======================================================================================
def fig_roc(t2_akhir, y_akhir, cl, ests, batas_utama, zoom=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#94a3b8", dash="dash"),
                             name="Acak (AUC = 0,5)", hoverinfo="skip"))
    neg, pos = int((~y_akhir).sum()), int(y_akhir.sum())
    for e in ests:
        t2 = t2_akhir[e]
        fpr, tpr = roc_points(y_akhir, t2)
        auc = auc_score(y_akhir, t2)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{e} (AUC {auc:.4f})", legendgroup=e,
                                 line=dict(color=WARNA_EST[e], width=2.6 if e == "IRDetMCD" else 1.8),
                                 hovertemplate="FPR %{x:.4f}<br>TPR %{y:.4f}<extra>" + e + "</extra>"))
        pred = t2 > cl[f"T2-{e}-{batas_utama}"]
        fig.add_trace(go.Scatter(
            x=[(pred & ~y_akhir).sum() / max(neg, 1)], y=[(pred & y_akhir).sum() / max(pos, 1)],
            mode="markers", legendgroup=e, showlegend=False,
            marker=dict(color=WARNA_EST[e], size=13, symbol="diamond", line=dict(color="white", width=1.5)),
            hovertemplate=f"<b>{e}</b> pada batas {batas_utama}<br>FPR %{{x:.4f}}<br>TPR %{{y:.4f}}<extra></extra>"))
    _base(fig, title=f"Kurva ROC (◆ = titik kerja pada batas {batas_utama})", h=520)
    fig.update_xaxes(title_text="False Positive Rate (alarm palsu)", range=[0, 0.1] if zoom else [0, 1])
    fig.update_yaxes(title_text="True Positive Rate (outlier terdeteksi)", range=[0.8, 1.005] if zoom else [0, 1.02])
    return fig


def fig_box_replikasi(df, ests, batas, metric, label):
    fig = go.Figure()
    for e in ests:
        d = df[(df.estimator == e) & (df.batas == batas)]
        fig.add_trace(go.Box(y=d[metric], name=e, marker_color=WARNA_EST[e], boxmean=True, boxpoints="outliers",
                             hovertemplate="%{y:.4f}<extra>" + e + "</extra>"))
    _base(fig, title=f"Sebaran {label} antar replikasi (batas {batas}) — ◆ = rata-rata", h=450, legend=False)
    fig.update_yaxes(title_text=label)
    return fig


def fig_radar(rata, ests, batas):
    sumbu = ["Hit rate", "Spesifisitas (1−FP)", "Deteksi outlier (1−FN)", "F1", "Balanced acc.", "AUC"]
    vals = {}
    for e in ests:
        r = rata.loc[f"T2-{e}-{batas}"]
        vals[e] = [r.accuracy, 1 - r.fp_rate, 1 - r.fn_rate, r.f1, r.balanced_accuracy, r.auc]
    allv = np.array(list(vals.values()), float)
    lo = max(0.0, float(np.nanmin(allv)) - 0.05)
    fig = go.Figure()
    for e in ests:
        v = list(vals[e]) + [vals[e][0]]
        fig.add_trace(go.Scatterpolar(r=v, theta=sumbu + [sumbu[0]], name=e, fill="toself", opacity=0.55,
                                      line=dict(color=WARNA_EST[e], width=2.6 if e == "IRDetMCD" else 1.6),
                                      hovertemplate="%{theta}: %{r:.4f}<extra>" + e + "</extra>"))
    _base(fig, title=f"Profil multi-metrik (batas {batas})", h=560)
    fig.update_layout(polar=dict(radialaxis=dict(range=[lo, 1.0], gridcolor=GRID, tickfont=dict(size=10)),
                                 angularaxis=dict(gridcolor=GRID), bgcolor="rgba(0,0,0,0)"),
                      legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left"))
    return fig


def fig_heatmap_metrics(rata, ests, bataslist):
    kol = [("accuracy", "Hit rate", True), ("fp_rate", "FP rate", False), ("fn_rate", "FN rate", False),
           ("f1", "F1", True), ("balanced_accuracy", "Balanced acc.", True), ("auc", "AUC", True),
           ("waktu_detik", "Waktu (dtk)", False)]
    ms = [f"T2-{e}-{b}" for e in ests for b in bataslist]
    z, txt = [], []
    for m in ms:
        z.append([])
        txt.append([])
    for j, (c, _, naik) in enumerate(kol):
        v = rata.loc[ms, c].astype(float).values
        rg = np.nanmax(v) - np.nanmin(v)
        nz = np.full_like(v, 0.5) if rg <= 1e-12 else (v - np.nanmin(v)) / rg
        if not naik:
            nz = 1 - nz
        for i in range(len(ms)):
            z[i].append(float(nz[i]))
            txt[i].append(f"{v[i]:.4f}" if c != "waktu_detik" else f"{v[i]:.3f}")
    fig = go.Figure(go.Heatmap(
        z=z, x=[k[1] for k in kol], y=[pm(m) for m in ms], text=txt, texttemplate="%{text}",
        colorscale=[[0, "#f8d3d3"], [0.5, "#fff4cf"], [1, "#bfe9cd"]], zmin=0, zmax=1, showscale=False,
        xgap=2, ygap=2, hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>"))
    _base(fig, title="Peta panas kinerja (hijau = lebih baik dalam kolom tersebut)", h=max(420, 30 * len(ms) + 120),
          legend=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig.update_layout(margin=dict(l=130, r=20, t=110, b=30))
    return fig


def fig_time_acc(rata, ests, batas, ymetric="accuracy", ylabel="Hit rate"):
    fig = go.Figure()
    pts = []
    for e in ests:
        r = rata.loc[f"T2-{e}-{batas}"]
        pts.append((e, float(r.waktu_detik), float(r[ymetric])))
        fig.add_trace(go.Scatter(
            x=[r.waktu_detik], y=[r[ymetric]], mode="markers+text", text=[e], textposition="top center",
            name=e, marker=dict(color=WARNA_EST[e], size=22, line=dict(color="white", width=2)),
            hovertemplate=f"<b>{e}</b><br>Waktu {r.waktu_detik:.4f} dtk<br>{ylabel} {r[ymetric]:.4f}<extra></extra>"))
    # Garis Pareto (lebih cepat & lebih akurat)
    pareto, best = [], -np.inf
    for e, t, v in sorted(pts, key=lambda a: a[1]):
        if v > best:
            pareto.append((t, v))
            best = v
    fig.add_trace(go.Scatter(x=[a[0] for a in pareto], y=[a[1] for a in pareto], mode="lines",
                             line=dict(color="#94a3b8", dash="dot", shape="hv"), name="Garis Pareto",
                             hoverinfo="skip"))
    _base(fig, title=f"Efisiensi: waktu komputasi vs {ylabel} (batas {batas})", h=470)
    fig.update_xaxes(title_text="Waktu per replikasi (detik, skala log)", type="log")
    fig.update_yaxes(title_text=ylabel)
    return fig


def fig_convergence(df, ests, batas, metric="accuracy", label="Hit rate"):
    piv = df[df.batas == batas].pivot(index="replikasi", columns="estimator", values=metric)
    cm = piv.expanding().mean()
    fig = go.Figure()
    for e in ests:
        if e in cm:
            fig.add_trace(go.Scatter(x=cm.index, y=cm[e], mode="lines", name=e,
                                     line=dict(color=WARNA_EST[e], width=2.6 if e == "IRDetMCD" else 1.6),
                                     hovertemplate="Replikasi %{x}<br>rata-rata berjalan %{y:.5f}<extra>" + e + "</extra>"))
    _base(fig, title=f"Konvergensi rata-rata berjalan {label} (batas {batas})", h=440)
    fig.update_xaxes(title_text="Jumlah replikasi")
    fig.update_yaxes(title_text=f"Rata-rata {label} kumulatif")
    return fig


def pca_2d(X):
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    var = S[:2] ** 2 / np.sum(S ** 2) * 100
    return Z, var


def fig_pca_anim(X, y, t2_akhir, cl, ests, batas, dur=1400):
    Z, var = pca_2d(X)
    ids = np.arange(1, len(y) + 1)

    def tr(e):
        pred = t2_akhir[e] > cl[f"T2-{e}-{batas}"]
        return _outcome_masks(y, pred), pred

    def judul(e):
        (tn, tp, fn, fp), _ = tr(e)
        return f"T²-{e}-{batas} · TP = {int(tp.sum())} · FN = {int(fn.sum())} · FP = {int(fp.sum())} · TN = {int(tn.sum())}"

    def traces(e):
        ms, _ = tr(e)
        out = []
        for (lab, col, sym), mk in zip(GRUP, ms):
            out.append(go.Scatter(x=Z[mk, 0], y=Z[mk, 1], customdata=np.c_[ids[mk], t2_akhir[e][mk]],
                                  mode="markers", name=lab,
                                  marker=dict(color=col, size=7, symbol=sym, opacity=0.8),
                                  hovertemplate="Obs #%{customdata[0]}<br>T² = %{customdata[1]:.2f}<extra>" + lab + "</extra>"))
        return out

    fig = go.Figure(data=traces(ests[0]))
    fig.frames = [go.Frame(data=traces(e), traces=[0, 1, 2, 3], name=e,
                           layout=go.Layout(title=dict(text=judul(e), x=0.0, xanchor="left", font=dict(size=15))))
                  for e in ests]
    _base(fig, title=judul(ests[0]), h=580)
    fig.update_layout(margin=dict(l=55, r=20, t=100, b=120), clickmode="event+select")
    fig.update_xaxes(title_text=f"PC1 ({var[0]:.1f}% variansi)", range=[Z[:, 0].min() * 1.05, Z[:, 0].max() * 1.05])
    fig.update_yaxes(title_text=f"PC2 ({var[1]:.1f}% variansi)", range=[Z[:, 1].min() * 1.05, Z[:, 1].max() * 1.05])
    _anim_controls(fig, list(ests), list(ests), dur=dur, y_pos=-0.17, prefix="Estimator: ")
    return fig


def fig_dd(X, t2, y, e, cl_val, alpha, batas):
    p = X.shape[1]
    d_cl = np.sqrt(mahal2(X, X.mean(0), np.cov(X, rowvar=False)))
    d_rb = np.sqrt(t2)
    cut_cl = float(np.sqrt(stats.chi2.ppf(1 - alpha, p)))
    cut_rb = float(np.sqrt(cl_val))
    ids = np.arange(1, len(y) + 1)
    fig = go.Figure()
    for lab, mk, col in [("Normal (aktual)", ~y, C_TN), ("Outlier (aktual)", y, C_TP)]:
        fig.add_trace(go.Scatter(x=d_cl[mk], y=d_rb[mk], mode="markers", name=lab, customdata=ids[mk],
                                 marker=dict(color=col, size=6, opacity=0.75),
                                 hovertemplate="Obs #%{customdata}<br>jarak klasik %{x:.2f}<br>jarak robust %{y:.2f}<extra>" + lab + "</extra>"))
    mx = float(max(d_cl.max(), d_rb.max())) * 1.03
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#94a3b8", dash="dot"),
                             name="y = x", hoverinfo="skip"))
    fig.add_vline(x=cut_cl, line_color="#0f766e", line_dash="dash",
                  annotation_text=f"batas klasik χ² = {cut_cl:.2f}", annotation_position="top right")
    fig.add_hline(y=cut_rb, line_color=WARNA_BATAS[batas], line_dash="dash",
                  annotation_text=f"batas {batas} (√CL) = {cut_rb:.2f}", annotation_position="top left")
    _base(fig, title=f"DD-plot: jarak Mahalanobis klasik vs robust — {e}", h=520)
    fig.update_xaxes(title_text="Jarak Mahalanobis klasik (rata-rata & kovarians biasa)")
    fig.update_yaxes(title_text=f"Jarak robust √T² ({e})")
    stat = dict(
        n_out=int(y.sum()),
        masked=int(((d_cl < cut_cl) & y).sum()),
        masked_found=int(((d_cl < cut_cl) & y & (d_rb > cut_rb)).sum()),
        found_cl=int(((d_cl >= cut_cl) & y).sum()),
        found_rb=int(((d_rb > cut_rb) & y).sum()),
        cut_cl=cut_cl, cut_rb=cut_rb)
    return fig, stat


# ======================================================================================
# Simulasi grid
# ======================================================================================
_SKALA_BAIK = [[0, "#f8d3d3"], [0.5, "#fff4cf"], [1, "#bfe9cd"]]


def fig_grid_heat(grid, metode, metric, label, naik, p_list, e_list):
    g = grid[grid.metode == metode]
    pv = g.pivot_table(index="p", columns="pers", values=metric).reindex(index=p_list, columns=e_list)
    z = pv.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{e}%" for e in e_list], y=[f"p = {p}" for p in p_list],
        text=[[("–" if np.isnan(v) else f"{v:.4f}") for v in row] for row in z], texttemplate="%{text}",
        colorscale=_SKALA_BAIK if naik else [[0, "#bfe9cd"], [0.5, "#fff4cf"], [1, "#f8d3d3"]],
        xgap=3, ygap=3, colorbar=dict(title=label, thickness=12),
        hovertemplate="%{y} · ε = %{x}<br>" + label + " = %{z:.4f}<extra></extra>"))
    _base(fig, title=f"{label} — {pm(metode)} (p × ε)", h=max(340, 60 * len(p_list) + 130), legend=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Proporsi outlier ε")
    return fig


def fig_grid_anim(grid, ests, batas, metric, label, p_list, e_list, by="p", dur=900):
    """Garis kinerja per estimator; animasi menggeser p (sumbu-x = ε) atau ε (sumbu-x = p)."""
    x_col, f_col = ("pers", "p") if by == "p" else ("p", "pers")
    x_vals = e_list if by == "p" else p_list
    f_vals = p_list if by == "p" else e_list
    g = grid[grid.metode.map(lambda m: m.split("-")[-1]) == batas]
    yv = g[metric].astype(float)
    pad = (yv.max() - yv.min()) * 0.08 + 1e-4

    def traces(fv):
        out = []
        for e in ests:
            s = g[(g.metode == f"T2-{e}-{batas}") & (g[f_col] == fv)].sort_values(x_col)
            out.append(go.Scatter(x=s[x_col], y=s[metric], mode="lines+markers", name=e,
                                  line=dict(color=WARNA_EST[e], width=3.4 if e == "IRDetMCD" else 2),
                                  marker=dict(size=8),
                                  hovertemplate=("ε = " if by == "p" else "p = ") + "%{x}<br>" + label +
                                                " = %{y:.4f}<extra>" + e + "</extra>"))
        return out

    nm = lambda v: f"p = {v}" if by == "p" else f"ε = {v}%"
    fig = go.Figure(data=traces(f_vals[0]))
    fig.frames = [go.Frame(data=traces(v), traces=list(range(len(ests))), name=nm(v),
                           layout=go.Layout(title=dict(text=f"{label} — {nm(v)} (batas {batas})", x=0.0,
                                                       xanchor="left", font=dict(size=16))))
                  for v in f_vals]
    _base(fig, title=f"{label} — {nm(f_vals[0])} (batas {batas})", h=540)
    fig.update_layout(margin=dict(l=55, r=20, t=95, b=120))
    fig.update_xaxes(title_text="Proporsi outlier ε (%)" if by == "p" else "Jumlah variabel p",
                     tickvals=list(x_vals), range=[min(x_vals) - 1, max(x_vals) + 1])
    fig.update_yaxes(title_text=label, range=[yv.min() - pad, yv.max() + pad])
    _anim_controls(fig, [nm(v) for v in f_vals], [nm(v) for v in f_vals], dur=dur, y_pos=-0.17,
                   prefix="Skenario: ")
    return fig


def fig_grid_surface(grid, metode, metric, label, p_list, e_list):
    g = grid[grid.metode == metode]
    pv = g.pivot_table(index="p", columns="pers", values=metric).reindex(index=p_list, columns=e_list)
    fig = go.Figure(go.Surface(
        z=pv.values.astype(float), x=e_list, y=p_list, colorscale="Viridis", showscale=True,
        colorbar=dict(title=label, thickness=12),
        hovertemplate="ε = %{x}%<br>p = %{y}<br>" + label + " = %{z:.4f}<extra></extra>"))
    _base(fig, title=f"Permukaan {label} — {pm(metode)} (putar dengan drag mouse)", h=540, legend=False)
    fig.update_layout(scene=dict(xaxis_title="ε (%)", yaxis_title="p", zaxis_title=label,
                                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
                      margin=dict(l=0, r=0, t=70, b=0))
    return fig


def fig_grid_fp(grid, ests, bataslist, alpha, eps_max):
    g = grid[grid.pers <= eps_max]
    fp = g.groupby("metode").fp_rate.mean()
    fig = go.Figure()
    for b in bataslist:
        v = [fp.get(f"T2-{e}-{b}", np.nan) for e in ests]
        fig.add_trace(go.Bar(x=ests, y=v, name=f"Batas {b}", marker_color=WARNA_BATAS[b],
                             hovertemplate="%{x}<br>FP rata-rata %{y:.5f}<extra>" + b + "</extra>"))
    fig.add_hline(y=alpha, line_dash="dash", line_color="#111827",
                  annotation_text=f"α = {alpha:g}", annotation_position="top left")
    _base(fig, title=f"FP rate rata-rata seluruh skenario (ε ≤ {eps_max}%)", h=420)
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(title_text="FP rate")
    return fig


def fig_rank_bar(ring):
    d = ring.sort_values("Hit rate", ascending=True)
    warna = [WARNA_EST[m.split("-")[1]] for m in d.index]
    fig = go.Figure(go.Bar(
        y=[pm(m) for m in d.index], x=d["Hit rate"], orientation="h", marker_color=warna,
        text=[f"{v:.4f}" for v in d["Hit rate"]], textposition="outside", customdata=[[m] for m in d.index],
        hovertemplate="<b>%{y}</b><br>Hit rate rata-rata %{x:.4f}<extra></extra>"))
    _base(fig, title="Peringkat hit rate rata-rata (seluruh skenario grid)", h=max(380, 26 * len(d) + 120),
          legend=False)
    lo = max(0.0, float(d["Hit rate"].min()) - 0.05)
    fig.update_xaxes(range=[lo, min(1.0, float(d["Hit rate"].max()) + 0.03)], title_text="Hit rate")
    fig.update_layout(clickmode="event+select", margin=dict(l=130, r=50, t=80, b=50))
    return fig
