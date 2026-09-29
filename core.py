"""
core.py — Mesin komputasi Diagram Kontrol Hotelling T² Robust
=============================================================
7 estimator (IRDetMCD, DetMCD, S, BACON, OGK, SD, MRWCD) × 3 batas kontrol (F, KDE, Boot).

Seluruh algoritma disalin dari notebook Colab asli. Perubahan hanya pada cara parameter
dikirim: konstanta global notebook diganti dengan dictionary `prm` (parameter estimator) dan
argumen fungsi, sehingga semuanya dapat diatur dari antarmuka Streamlit dan aman dipakai
oleh proses paralel (joblib/loky).
"""
from __future__ import annotations

import io
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import integrate, optimize, stats
from statsmodels.robust.scale import qn_scale

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Konstanta & nilai bawaan
# --------------------------------------------------------------------------------------
EST_SEMUA = ["IRDetMCD", "DetMCD", "S", "BACON", "OGK", "SD", "MRWCD"]
BATAS_SEMUA = ["F", "KDE", "Boot"]
DIST_SEMUA = ["normal", "gamma", "eksponensial", "weibull"]

WARNA_EST = {"IRDetMCD": "#2a78d6", "DetMCD": "#eb6834", "S": "#1baf7a", "BACON": "#eda100",
             "OGK": "#e87ba4", "SD": "#8b7bff", "MRWCD": "#d09a66"}
WARNA_BATAS = {"F": "#8a8984", "KDE": "#d62728", "Boot": "#4da3e8"}

KOLOM = ["accuracy", "fp_rate", "fn_rate", "sensitivity", "specificity", "f1", "balanced_accuracy"]
KOLOM_EXTRA = KOLOM + ["auc"]
LEBIH_BESAR_BAIK = {"accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy", "auc"}

# Parameter khusus tiap estimator (nilai asli notebook)
DEFAULT_PRM = dict(
    # BACON (Billor, Hadi & Velleman, 2000)
    c_bacon=4, bacon_version=2, max_bacon_iter=100,
    # OGK (Maronna & Zamar, 2002) versi teman
    n_iter_ogk=2,
    # Stahel–Donoho
    n_directions=500, sd_cutoff=3.0,
    # MRWCD
    h_frac=0.75, reg_rho=0.10, mrwcd_max_iter=100,
    # IRDetMCD
    irdet_q=0.975, irdet_max_iter=50,
    # S-estimator
    s_max_iter=100, s_tol=1e-7,
    # alpha untuk cutoff BACON (diisi otomatis = alpha skenario)
    bacon_alpha=0.00273,
)

_HAS_GEN = tuple(int(x) for x in joblib.__version__.split(".")[:2]) >= (1, 3)


def paralel(tasks, n_jobs=-1, cb=None):
    """Jalankan daftar `delayed(...)` secara paralel; `cb(selesai, total)` dipanggil tiap tugas selesai."""
    total, out = len(tasks), []
    if _HAS_GEN:
        for i, r in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(tasks), 1):
            out.append(r)
            if cb:
                cb(i, total)
    else:  # joblib lama: tanpa progres per tugas
        out = Parallel(n_jobs=n_jobs)(tasks)
        if cb:
            cb(total, total)
    return out


# --------------------------------------------------------------------------------------
# 3 · Pembangkitan data
# --------------------------------------------------------------------------------------
def sigma_rho(p, rho):
    return np.full((p, p), rho) - np.diag(np.full(p, rho)) + np.eye(p)


SD_DIST = {"normal": 1.0, "gamma": np.sqrt(2.0), "eksponensial": 0.5, "weibull": np.sqrt(1 - np.pi / 4)}


def acak_dist(rng, dist, m, p, L):
    if dist == "normal":
        return rng.standard_normal((m, p)) @ L.T
    if dist == "gamma":
        return rng.gamma(shape=2.0, scale=1.0, size=(m, p))
    if dist == "eksponensial":
        return rng.exponential(scale=1 / 2.0, size=(m, p))
    if dist == "weibull":
        return rng.weibull(2.0, size=(m, p)) * 1.0
    raise ValueError(dist)


def bangkitkan_data(rng, n, p, pers, mu_out, rho, dist="normal"):
    n2 = int(round(n * pers / 100))
    n3 = n - n2
    L = np.linalg.cholesky(sigma_rho(p, rho))
    geser = mu_out * SD_DIST[dist]
    a3 = acak_dist(rng, dist, n3, p, L)           # data bersih, label 0
    a2 = acak_dist(rng, dist, n2, p, L) + geser   # outlier, label 1
    X = np.vstack([a2, a3])
    y = np.r_[np.ones(n2, bool), np.zeros(n3, bool)]
    idx = rng.permutation(n)
    return X[idx], y[idx]


# --------------------------------------------------------------------------------------
# 4a · DetMCD, S-estimator, IRDetMCD (metode kami)
# --------------------------------------------------------------------------------------
def mahal2(X, mu, S):
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = (S + S.T) / 2
        w, V = np.linalg.eigh(S)
        L = np.linalg.cholesky(V @ np.diag(np.maximum(w, 1e-8)) @ V.T)
    Z = np.linalg.solve(L, (X - mu).T)
    return (Z ** 2).sum(0)


def tau_scale(Z, c1=4.5, c2=3.0):
    med = np.median(Z, 0)
    mad = np.median(np.abs(Z - med), 0)
    mad = np.where(mad <= 0, 1e-12, mad)
    u = (Z - med) / (c1 * mad)
    w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    loc = (w * Z).sum(0) / w.sum(0)
    s2 = np.minimum(((Z - loc) / mad) ** 2, c2 ** 2).mean(0)
    Ec = 2 * ((1 - c2 ** 2) * stats.norm.cdf(c2) - c2 * stats.norm.pdf(c2) + c2 ** 2) - 1
    return loc, mad * np.sqrt(s2 / Ec)


def ogk_raw(Z):
    n, p = Z.shape
    A = np.eye(p)
    W = Z.copy()
    iu = np.triu_indices(p, 1)
    for _ in range(2):
        _, s = tau_scale(W)
        Y = W / s
        _, sp = tau_scale(Y[:, iu[0]] + Y[:, iu[1]])
        _, sm = tau_scale(Y[:, iu[0]] - Y[:, iu[1]])
        U = np.eye(p)
        U[iu] = .25 * (sp ** 2 - sm ** 2)
        U[(iu[1], iu[0])] = U[iu]
        _, E = np.linalg.eigh(U)
        W = Y @ E
        A = A @ np.diag(1 / s) @ E
    loc, sc = tau_scale(W)
    Ai = np.linalg.inv(A)
    return loc @ Ai, Ai.T @ np.diag(sc ** 2) @ Ai


def c_steps(X, idx, h, max_iter=100):
    best = np.inf
    for _ in range(max_iter):
        mu = X[idx].mean(0)
        S = np.cov(X[idx], rowvar=False)
        sgn, logdet = np.linalg.slogdet(S)
        if sgn <= 0:
            return np.inf, mu, S
        new = np.argpartition(mahal2(X, mu, S), h - 1)[:h]
        if logdet >= best - 1e-10:
            break
        best, idx = logdet, new
    return logdet, mu, S


def est_detmcd(X, seed=None, prm=None):
    n, p = X.shape
    h = (n + p + 1) // 2
    h0 = int(np.ceil(n / 2))
    med = np.median(X, 0)
    qn = np.array([qn_scale(X[:, j]) for j in range(p)])
    qn[qn <= 0] = 1e-12
    Z = (X - med) / qn
    R = stats.rankdata(Z, axis=0)
    nz = np.linalg.norm(Z, axis=1)
    K = Z / np.where(nz > 0, nz, 1)[:, None]
    awal = [np.corrcoef(np.tanh(Z), rowvar=False),
            np.corrcoef(R, rowvar=False),
            np.corrcoef(stats.norm.ppf((R - 1 / 3) / (n + 1 / 3)), rowvar=False),
            K.T @ K / n]
    subset = []
    for S in awal:
        _, E = np.linalg.eigh(S)
        _, sc = tau_scale(Z @ E)
        Sig = E @ np.diag(sc ** 2) @ E.T
        w, V = np.linalg.eigh(Sig)
        rt = V @ np.diag(np.sqrt(np.maximum(w, 1e-12))) @ V.T
        mu = np.median(Z @ np.linalg.inv(rt), 0) @ rt
        subset.append(np.argpartition(mahal2(Z, mu, Sig), h0 - 1)[:h0])
    subset.append(np.argpartition(nz, h0 - 1)[:h0])
    m6, S6 = ogk_raw(Z)
    subset.append(np.argpartition(mahal2(Z, m6, S6), h0 - 1)[:h0])
    best = (np.inf, None, None)
    for idx in subset:
        mu = Z[idx].mean(0)
        S = np.cov(Z[idx], rowvar=False)
        res = c_steps(Z, np.argpartition(mahal2(Z, mu, S), h - 1)[:h], h)
        if res[0] < best[0]:
            best = res
    _, mu, S = best
    S = S * np.median(mahal2(Z, mu, S)) / stats.chi2.ppf(.5, p)
    keep = mahal2(Z, mu, S) <= stats.chi2.ppf(.975, p)
    D = np.diag(qn)
    return med + Z[keep].mean(0) * qn, D @ np.cov(Z[keep], rowvar=False) @ D


def est_ogk_awal(X):
    """OGK + reweighting, dipakai sebagai titik awal S-estimator."""
    mu, S = ogk_raw(X)
    d2 = mahal2(X, mu, S)
    p = X.shape[1]
    keep = d2 <= stats.chi2.ppf(0.9, p) * np.median(d2) / stats.chi2.ppf(0.5, p)
    return X[keep].mean(0), np.cov(X[keep], rowvar=False)


_S_KONST = {}


def rho_biweight(u, c):
    u = np.minimum(np.abs(u), c)
    return u ** 2 / 2 - u ** 4 / (2 * c ** 2) + u ** 6 / (6 * c ** 4)


def konstanta_s(p, bdp=0.5):
    if p not in _S_KONST:
        E = lambda c: integrate.quad(lambda t: rho_biweight(np.sqrt(t), c) *
                                     stats.chi2.pdf(t, p), 0, np.inf, limit=200)[0]
        c = optimize.brentq(lambda c: E(c) / (c ** 2 / 6) - bdp, 0.1, 50)
        _S_KONST[p] = (c, E(c))
    return _S_KONST[p]


def s_iterasi(X, mu, S, max_iter=100, tol=1e-7):
    n, p = X.shape
    c, b = konstanta_s(p)
    V = S / np.linalg.det(S) ** (1 / p)
    d = np.sqrt(mahal2(X, mu, V))
    sig = np.median(d)
    for _ in range(max_iter):
        for _k in range(30):
            sig_baru = sig * np.sqrt(rho_biweight(d / sig, c).mean() / b)
            if abs(sig_baru / sig - 1) < 1e-10:
                break
            sig = sig_baru
        u = d / sig
        w = np.where(u < c, (1 - (u / c) ** 2) ** 2, 0.0)
        mu_baru = (w[:, None] * X).sum(0) / w.sum()
        R = X - mu_baru
        V_baru = (w[:, None] * R).T @ R
        V_baru /= np.linalg.det(V_baru) ** (1 / p)
        selesai = np.abs(mu_baru - mu).max() < tol and np.abs(V_baru - V).max() < tol
        mu, V = mu_baru, V_baru
        d = np.sqrt(mahal2(X, mu, V))
        if selesai:
            break
    return mu, V, sig


def est_s(X, seed=None, prm=None):
    prm = prm or DEFAULT_PRM
    mu, V, sig = s_iterasi(X, *est_ogk_awal(X), max_iter=prm["s_max_iter"], tol=prm["s_tol"])
    return mu, sig ** 2 * V


def est_irdetmcd(X, seed=None, prm=None):
    """Iteratively Reweighted DetMCD."""
    prm = prm or DEFAULT_PRM
    q, max_iter = prm["irdet_q"], prm["irdet_max_iter"]
    mu, S = est_detmcd(X)
    p = X.shape[1]
    cut = stats.chi2.ppf(q, p)
    faktor = stats.chi2.cdf(cut, p + 2) / q
    keep = None
    for _ in range(max_iter):
        k = mahal2(X, mu, S) <= cut
        if keep is not None and np.array_equal(k, keep):
            break
        keep = k
        mu = X[k].mean(0)
        S = np.cov(X[k], rowvar=False) / faktor
    return mu, S


# --------------------------------------------------------------------------------------
# 4b · BACON, OGK, Stahel–Donoho, MRWCD (metode teman)
# --------------------------------------------------------------------------------------
def make_covariance_safe(S):
    S = np.asarray(S, dtype=float)
    S = (S + S.T) / 2
    eigval, eigvec = np.linalg.eigh(S)
    eigval = np.maximum(eigval, 1e-8)
    return eigvec @ np.diag(eigval) @ eigvec.T


def mahal2_bacon(X, mu, S):
    S = make_covariance_safe(S)
    D = X - mu
    Sinv = np.linalg.pinv(S)
    T2 = np.einsum("ij,jk,ik->i", D, Sinv, D)
    return np.maximum(T2, 0.0)


def bacon_estimator(X, c=4, version=2, max_iter=100, alpha_b=0.00273):
    X = np.asarray(X, dtype=float)
    n_obs, p = X.shape
    if n_obs <= p:
        raise ValueError("BACON membutuhkan n > p.")
    m = int(c * p)
    m = max(m, p + 1)
    m = min(m, n_obs)
    initial_m = m
    if version == 1:
        mu0 = np.mean(X, axis=0)
        S0 = np.cov(X, rowvar=False, ddof=1)
        d0 = np.sqrt(mahal2_bacon(X, mu0, S0))
    elif version == 2:
        med = np.median(X, axis=0)
        d0 = np.linalg.norm(X - med, axis=1)
    else:
        raise ValueError("BACON version harus 1 atau 2.")
    order = np.argsort(d0)
    basic = np.sort(order[:m])
    current_m = len(basic)
    while True:
        Xb = X[basic]
        Sb = np.atleast_2d(np.cov(Xb, rowvar=False, ddof=1))
        if np.linalg.matrix_rank(Sb) == p:
            break
        if current_m >= n_obs:
            raise ValueError("Covariance BACON tidak full rank.")
        current_m += 1
        basic = np.sort(order[:current_m])
    h = int(np.floor((n_obs + p + 1) / 2))
    denominator = n_obs - h - p
    if denominator <= 0:
        raise ValueError("n terlalu kecil dibandingkan p.")
    c_np = 1.0 + (p + 1) / (n_obs - p) + 1.0 / denominator
    chi_quantile = stats.chi2.ppf(1 - alpha_b / n_obs, df=p)
    chi_cutoff = np.sqrt(chi_quantile)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        r = len(basic)
        Xb = X[basic]
        mu_b = np.mean(Xb, axis=0)
        S_b = make_covariance_safe(np.cov(Xb, rowvar=False, ddof=1))
        distance = np.sqrt(mahal2_bacon(X, mu_b, S_b))
        c_hr = max(0.0, (h - r) / (h + r))
        bacon_cutoff = (c_np + c_hr) * chi_cutoff
        new_basic = np.sort(np.where(distance < bacon_cutoff)[0])
        if len(new_basic) < r:
            new_basic = np.sort(np.argsort(distance)[:r])
        if np.array_equal(new_basic, basic):
            basic = new_basic
            break
        basic = new_basic
    X_final = X[basic]
    mu_final = np.mean(X_final, axis=0)
    S_final = make_covariance_safe(np.cov(X_final, rowvar=False, ddof=1))
    return mu_final, S_final, basic, initial_m, len(basic), iteration


def est_bacon(X, seed=None, prm=None):
    prm = prm or DEFAULT_PRM
    mu, S, *_ = bacon_estimator(X, prm["c_bacon"], prm["bacon_version"],
                                prm["max_bacon_iter"], prm["bacon_alpha"])
    return mu, S


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad / 0.6744897501960817
    if scale < 1e-12:
        scale = np.std(x, ddof=1)
    if scale < 1e-12:
        scale = 1e-8
    return scale


def ogk_estimator_teman(X, n_iter_ogk=2):
    X = np.asarray(X, dtype=float)
    _, p = X.shape
    center = np.median(X, axis=0)
    Y = X - center
    transformation = np.eye(p)
    for _ in range(n_iter_ogk):
        scales = np.array([robust_scale(Y[:, j]) for j in range(p)])
        scales = np.maximum(scales, 1e-8)
        Z = Y / scales
        U = np.eye(p)
        for j in range(p):
            for k in range(j + 1, p):
                s_plus = robust_scale(Z[:, j] + Z[:, k])
                s_minus = robust_scale(Z[:, j] - Z[:, k])
                value = (s_plus ** 2 - s_minus ** 2) / 4
                U[j, k] = value
                U[k, j] = value
        _, eigenvectors = np.linalg.eigh(U)
        B = np.diag(scales) @ eigenvectors
        transformation = transformation @ B
        Y = Z @ eigenvectors
        Y = Y - np.median(Y, axis=0)
    final_scales = np.array([robust_scale(Y[:, j]) for j in range(p)])
    final_scales = np.maximum(final_scales, 1e-8)
    covariance = transformation @ np.diag(final_scales ** 2) @ transformation.T
    covariance = (covariance + covariance.T) / 2
    covariance += np.eye(p) * 1e-10
    center = np.median(X, axis=0)
    return center, covariance


def est_ogk(X, seed=None, prm=None):
    prm = prm or DEFAULT_PRM
    return ogk_estimator_teman(X, prm["n_iter_ogk"])


def generate_directions(p, n_directions=500, seed_sd=0):
    rng_sd = np.random.default_rng(seed_sd)
    directions = rng_sd.normal(size=(n_directions, p))
    norms = np.linalg.norm(directions, axis=1)
    norms[norms < 1e-12] = 1.0
    return directions / norms[:, None]


def stahel_donoho_outlyingness(X, n_directions=500, seed_sd=0):
    X = np.asarray(X, dtype=float)
    directions = generate_directions(X.shape[1], n_directions, seed_sd)
    projections = X @ directions.T
    med = np.median(projections, axis=0)
    mad = np.median(np.abs(projections - med), axis=0) / 0.6744897501960817
    mad = np.maximum(mad, 1e-8)
    return np.max(np.abs(projections - med) / mad, axis=1)


def stahel_donoho_weights(outlyingness, cutoff=3.0):
    r = np.asarray(outlyingness, dtype=float)
    w = np.ones_like(r)
    mask = r > cutoff
    w[mask] = (cutoff / r[mask]) ** 2
    return w


def est_sd(X, seed=None, prm=None):
    prm = prm or DEFAULT_PRM
    seed_sd = 0 if seed is None else int(seed)
    out = stahel_donoho_outlyingness(X, int(prm["n_directions"]), seed_sd)
    w = stahel_donoho_weights(out, prm["sd_cutoff"])
    sw = np.sum(w)
    if sw <= 0:
        w = np.ones(X.shape[0])
        sw = len(w)
    mu = np.sum(X * w[:, None], axis=0) / sw
    R = X - mu
    S = (R.T @ (R * w[:, None])) / sw
    S = (S + S.T) / 2
    S += np.eye(X.shape[1]) * 1e-8
    return mu, S


def safe_cov(S):
    S = np.asarray(S, float)
    S = (S + S.T) / 2
    val, vec = np.linalg.eigh(S)
    val = np.maximum(val, 1e-8)
    return vec @ np.diag(val) @ vec.T


def mahal2_mrwcd(X, mu, S):
    S = safe_cov(S)
    D = X - mu
    try:
        Z = np.linalg.solve(np.linalg.cholesky(S), D.T)
        return (Z ** 2).sum(axis=0)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
        return np.einsum("ij,jk,ik->i", D, Sinv, D)


def target_matrix(X):
    scales = np.array([np.median(np.abs(X[:, j] - np.median(X[:, j]))) / 0.67448975
                       for j in range(X.shape[1])])
    scales = np.maximum(scales, 1e-8)
    return np.diag(scales ** 2)


def mrwcd_estimator(X, h_frac=0.75, reg_rho=0.10, max_iter=100):
    X = np.asarray(X, float)
    n_obs, p = X.shape
    h = max(p + 2, int(np.floor(h_frac * n_obs)))
    h = min(h, n_obs)
    T = target_matrix(X)
    mu0 = np.median(X, axis=0)
    scale = np.sqrt(np.diag(T))
    Z = (X - mu0) / scale
    d0 = np.sum(Z ** 2, axis=1)
    subset = np.argsort(d0)[:h]
    for _ in range(max_iter):
        Xh = X[subset]
        mu = np.mean(Xh, axis=0)
        S = safe_cov(np.cov(Xh, rowvar=False, ddof=1))
        d2 = mahal2_mrwcd(Xh, mu, S)
        weights = 1 / (1 + d2)
        weights = weights / weights.sum()
        mu_w = np.sum(Xh * weights[:, None], axis=0)
        D = Xh - mu_w
        S_w = safe_cov(D.T @ (D * weights[:, None]))
        S_reg = safe_cov(reg_rho * T + (1 - reg_rho) * S_w)
        d_all = mahal2_mrwcd(X, mu_w, S_reg)
        new_subset = np.argsort(d_all)[:h]
        if np.array_equal(np.sort(new_subset), np.sort(subset)):
            subset = new_subset
            break
        subset = new_subset
    Xh = X[subset]
    mu_h = np.mean(Xh, axis=0)
    S_h = safe_cov(np.cov(Xh, rowvar=False, ddof=1))
    d2 = mahal2_mrwcd(Xh, mu_h, S_h)
    weights = 1 / (1 + d2)
    weights = weights / weights.sum()
    mu = np.sum(Xh * weights[:, None], axis=0)
    D = Xh - mu
    S_w = safe_cov(D.T @ (D * weights[:, None]))
    S = safe_cov(reg_rho * T + (1 - reg_rho) * S_w)
    return mu, S


def est_mrwcd(X, seed=None, prm=None):
    prm = prm or DEFAULT_PRM
    return mrwcd_estimator(X, prm["h_frac"], prm["reg_rho"], int(prm["mrwcd_max_iter"]))


# --------------------------------------------------------------------------------------
# 4c · Registri estimator & statistik T²
# --------------------------------------------------------------------------------------
ESTIMATOR = {"IRDetMCD": est_irdetmcd, "DetMCD": est_detmcd, "S": est_s,
             "BACON": est_bacon, "OGK": est_ogk, "SD": est_sd, "MRWCD": est_mrwcd}


def statistik_t2(e, X, seed=0, prm=None):
    mu, S = ESTIMATOR[e](X, seed=seed, prm=prm)
    return mahal2(X, mu, S)


# --------------------------------------------------------------------------------------
# 5 · Tiga batas kontrol: F, KDE, bootstrap
# --------------------------------------------------------------------------------------
def ucl_F(n, p, alpha):
    return p * (n - 1) * (n + 1) / (n * (n - p)) * stats.f.ppf(1 - alpha, p, n - p)


def bw_silverman(x):
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 0.9 * (min(sd, iqr / 1.34) if iqr > 0 else sd) * len(x) ** (-0.2)


def cl_kde(t2, alpha):
    t2 = np.sort(t2)
    h = bw_silverman(t2)
    F = lambda t: stats.norm.cdf((t - t2) / h).mean() - (1 - alpha)
    return optimize.brentq(F, t2[0] - 5 * h, t2[-1] + 5 * h, xtol=1e-8)


def cl_boot(t2, alpha, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    t2 = np.asarray(t2)
    q = [np.quantile(rng.choice(t2, len(t2), replace=True), 1 - alpha) for _ in range(B)]
    return float(np.mean(q))


def _t2_bersih(e, n, p, rho, s, dist, prm):
    X, _ = bangkitkan_data(np.random.default_rng(s), n, p, 0, 0, rho, dist)
    return statistik_t2(e, X, seed=s, prm=prm)


def hitung_batas(n, p, rho, alpha, n_ref, seed, est_aktif, batas_aktif, dist, prm, b_boot,
                 n_jobs=-1, cb=None):
    """Hitung batas kontrol tiap kombinasi.

    Fase I (T² data in-control) dihitung SEKALI per estimator lalu dipakai untuk KDE & bootstrap.
    Mengembalikan (cl, faseI): cl[`T2-<est>-<batas>`] dan faseI[est] = larik T² Fase I.
    `cb(selesai, total, label)` dipanggil untuk memperbarui progres.
    """
    cl, faseI = {}, {}
    if "F" in batas_aktif:
        for e in est_aktif:
            cl[f"T2-{e}-F"] = float(ucl_F(n, p, alpha))
    butuh = [e for e in est_aktif if ("KDE" in batas_aktif or "Boot" in batas_aktif)]
    total = len(butuh) * n_ref
    for ie, e in enumerate(butuh):
        tasks = [delayed(_t2_bersih)(e, n, p, rho, seed + 7919 * k, dist, prm) for k in range(n_ref)]
        base = ie * n_ref
        res = paralel(tasks, n_jobs,
                      (lambda i, t, base=base, e=e: cb(base + i, total, f"Fase I · {e}")) if cb else None)
        t2 = np.concatenate(res)
        faseI[e] = t2
        if "KDE" in batas_aktif:
            cl[f"T2-{e}-KDE"] = cl_kde(t2, alpha)
        if "Boot" in batas_aktif:
            cl[f"T2-{e}-Boot"] = cl_boot(t2, alpha, b_boot, seed)
    return cl, faseI


# --------------------------------------------------------------------------------------
# 6 · Metrik
# --------------------------------------------------------------------------------------
def metrik(y, pred, positive=0):
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    TP = int(np.sum(pred & y))
    FN = int(np.sum(~pred & y))
    FP = int(np.sum(pred & ~y))
    TN = int(np.sum(~pred & ~y))
    d = lambda a, b: a / b if b else np.nan
    tp, fn, fp, tn = (TN, FP, FN, TP) if positive == 0 else (TP, FN, FP, TN)
    sens, spec, prec = d(tp, tp + fn), d(tn, tn + fp), d(tp, tp + fp)
    return dict(TN=TN, FP=FP, FN=FN, TP=TP,
                accuracy=(TP + TN) / len(y),
                fp_rate=d(FP, TN + FP),
                fn_rate=d(FN, TP + FN),
                sensitivity=sens, specificity=spec,
                f1=d(2 * prec * sens, prec + sens),
                balanced_accuracy=(sens + spec) / 2)


def auc_score(y, s):
    """AUC (Mann–Whitney) dengan outlier sebagai kelas positif dan T² sebagai skor."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or np.isnan(s).any():
        return np.nan
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_points(y, s, max_pts=500):
    y = np.asarray(y, bool)
    o = np.argsort(-np.asarray(s))
    ys = y[o]
    tpr = np.cumsum(ys) / max(int(ys.sum()), 1)
    fpr = np.cumsum(~ys) / max(int((~ys).sum()), 1)
    idx = np.unique(np.linspace(0, len(s) - 1, min(max_pts, len(s))).astype(int))
    return np.r_[0.0, fpr[idx]], np.r_[0.0, tpr[idx]]


# --------------------------------------------------------------------------------------
# 7 · Simulasi
# --------------------------------------------------------------------------------------
def satu_replikasi(r, n, p, pers, mu_out, rho, cl, seed, dist, est_aktif, batas_aktif, prm,
                   positive, simpan=False):
    X, y = bangkitkan_data(np.random.default_rng(seed + r), n, p, pers, mu_out, rho, dist)
    baris, t2_simpan = [], {}
    nan_row = {k: np.nan for k in ["TN", "FP", "FN", "TP"] + KOLOM}
    for e in est_aktif:
        t0 = time.perf_counter()
        try:
            t2 = statistik_t2(e, X, seed=seed + r, prm=prm)
            if not np.all(np.isfinite(t2)):
                raise FloatingPointError("T2 tidak berhingga")
        except Exception:  # estimator gagal pada dataset ini -> catat sebagai NaN
            t2 = None
        dt = time.perf_counter() - t0
        auc = auc_score(y, t2) if t2 is not None else np.nan
        for b in batas_aktif:
            m = f"T2-{e}-{b}"
            met = metrik(y, t2 > cl[m], positive) if t2 is not None else dict(nan_row)
            baris.append(dict(replikasi=r + 1, metode=m, estimator=e, batas=b, **met, auc=auc,
                              waktu_detik=dt))
        if simpan and t2 is not None:
            t2_simpan[e] = t2
    return baris, ((t2_simpan, y, X) if simpan else None)


def simulasi(n, p, pers, mu_out, rho, reps, seed, cl, dist, est_aktif, batas_aktif, prm, positive,
             n_jobs=-1, cb=None):
    tasks = [delayed(satu_replikasi)(r, n, p, pers, mu_out, rho, cl, seed, dist, est_aktif,
                                     batas_aktif, prm, positive, r == reps - 1) for r in range(reps)]
    hasil = paralel(tasks, n_jobs, cb)
    df = pd.DataFrame([b for bb, _ in hasil for b in bb])
    terakhir = [x for _, x in hasil if x is not None][0]
    return df, terakhir


# --------------------------------------------------------------------------------------
# 9 · Tabel format paper & ringkasan grid
# --------------------------------------------------------------------------------------
def tabel_paper(g_all, metode, daftar_eps, p_grid):
    g = g_all[g_all.metode == metode]
    kol = {}
    for e in daftar_eps:
        s = g[g.pers == e].set_index("p").reindex(p_grid)
        kol[(f"ε={e}%", "Hit rate")] = s["accuracy"]
        kol[(f"ε={e}%", "FP rate")] = s["fp_rate"]
        kol[(f"ε={e}%", "FN rate")] = s["fn_rate"]
    t = pd.DataFrame(kol)
    t.index = [f"p = {q}" for q in t.index]
    return t


def ringkas_grid(grid, alpha, eps_max=100):
    g = grid[grid.pers <= eps_max]
    r = g.groupby("metode").agg(**{
        "Hit rate": ("accuracy", "mean"), "FP rate": ("fp_rate", "mean"), "FN rate": ("fn_rate", "mean"),
        "FP maks": ("fp_rate", "max"), "Skenario FP > α": ("fp_rate", lambda s: int((s > alpha).sum())),
        "Waktu (dtk)": ("waktu_detik", "mean")}).sort_values("Hit rate", ascending=False)
    r.insert(0, "Peringkat", range(1, len(r) + 1))
    return r


def buat_excel(res, grid=None, ringkasan=None, p_grid=None, pers_grid=None, cfg_tabel=None):
    """Susun workbook Excel (bytes) berisi hasil skenario utama dan (opsional) grid."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        if cfg_tabel is not None:
            cfg_tabel.to_excel(w, sheet_name="Konfigurasi", index=False)
        res["rata"].sort_values("accuracy", ascending=False).round(6).to_excel(w, sheet_name="Skenario utama")
        res["sd"].round(6).to_excel(w, sheet_name="SD skenario utama")
        res["df"].to_excel(w, sheet_name="Per replikasi", index=False)
        pd.DataFrame({"CL": res["cl"]}).round(6).to_excel(w, sheet_name="Batas kontrol")
        if grid is not None and len(grid):
            if ringkasan is not None:
                ringkasan.round(6).to_excel(w, sheet_name="Ringkasan grid")
            grid.to_excel(w, sheet_name="Grid data", index=False)
            blok = [pers_grid[i:i + 3] for i in range(0, len(pers_grid), 3)]
            for m in grid.metode.unique():
                r0 = 0
                for b in blok:
                    t = tabel_paper(grid, m, b, p_grid)
                    t.round(4).to_excel(w, sheet_name=m[:31], startrow=r0)
                    r0 += len(t) + 5
    return buf.getvalue()
