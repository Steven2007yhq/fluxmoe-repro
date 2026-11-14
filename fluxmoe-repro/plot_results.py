#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FluxMoE 动机实验复现 —— 数据汇总与出图

三种用法：
  python plot_results.py --ingest results/raw --out results/results.csv     # 把 raw/*.json+.meta 汇总成 CSV
  python plot_results.py --csv results/results.csv --outdir results/figs    # 正式出图
  python plot_results.py --demo --outdir results/figs                       # 无数据时预览预期形状
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# ---------------- 中文字体 ----------------
CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Micro Hei", "PingFang SC"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_has_cjk = False
for _f in CJK_CANDIDATES:
    if _f in _available:
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        _has_cjk = True
        break
if not _has_cjk:
    print("[warn] 未找到中文字体，图中中文将显示为方块。", file=sys.stderr)

# ---------------- 配色 ----------------
TIER = ["#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
CTX_COLORS = {1024: "#0EA5E9", 2048: "#F59E0B", 4096: "#EF4444"}
INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"


def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


# ---------------- ingest ----------------
FNAME = re.compile(r"u(?P<util>[\d.]+)_c(?P<ctx>\d+)_b(?P<conc>\d+)_r(?P<rep>\d+)\.json$")


def _pick(d, keys, default=np.nan):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return default


def ingest(raw_dir, out_csv):
    rows = []
    for js in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        m = FNAME.search(os.path.basename(js))
        if not m:
            continue
        tag = js[:-5]
        meta = {}
        if os.path.exists(tag + ".meta"):
            for line in open(tag + ".meta", encoding="utf-8"):
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    meta[k] = v
        try:
            d = json.load(open(js, encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rows.append({
            "model": meta.get("model", ""),
            "tp": float(meta.get("tp", 1)),
            "util": float(m.group("util")),
            "rep": int(m.group("rep")),
            "ctx": int(m.group("ctx")),
            "conc": int(m.group("conc")),
            "kv_tokens": float(meta.get("kv_tokens", 0)),
            "num_prompts": int(float(meta.get("num_prompts", 0))),
            "out_throughput": _pick(d, ["output_throughput", "total_token_throughput"]),
            "request_throughput": _pick(d, ["request_throughput"]),
            "tpot_ms": _pick(d, ["mean_tpot_ms", "mean_itl_ms"]),
            "ttft_ms": _pick(d, ["mean_ttft_ms"]),
            "preempt_delta": float(meta.get("preempt_delta", 0)),
            "cache_usage": float(meta.get("cache_usage", 0)),
            "gpu_temp": float(meta.get("gpu_temp", 0)),
            "gpu_clock": float(meta.get("gpu_clock", 0)),
            "rc": int(float(meta.get("rc", 0))),
        })
    if not rows:
        print("[ingest] 没有可解析的结果", file=sys.stderr)
        return None
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[ingest] {len(df)} 行 → {out_csv}")
    return df


# ---------------- demo ----------------
def demo_data():
    """合成数据，用来预览四张图的形状（不是真实测量）。"""
    kv = {0.60: 172_000, 0.70: 344_000, 0.80: 516_000, 0.90: 640_000}
    r = {1024: 9.0, 2048: 6.5, 4096: 4.2}
    ceiling = {1024: 720.0, 2048: 690.0, 4096: 620.0}
    rows = []
    for util, kvt in kv.items():
        for ctx in (1024, 2048, 4096):
            cap = kvt / ctx
            for conc in (16, 32, 64, 128, 256):
                eff = min(conc, cap)
                thr = min(eff * r[ctx], ceiling[ctx])
                if conc > cap:
                    thr *= (cap / conc) ** 1.2
                rows.append({
                    "util": util, "ctx": ctx, "conc": conc, "rep": 1,
                    "kv_tokens": kvt, "out_throughput": thr,
                    "tpot_ms": 1000 / r[ctx] * (1 + conc / 400),
                    "ttft_ms": 180 + ctx * 0.55 + conc * 9,
                    "preempt_delta": max(0.0, conc - cap) * 2.2 * (ctx / 4096),
                    "cache_usage": min(1.0, conc * ctx / kvt),
                })
    df = pd.DataFrame(rows)
    df.attrs["demo"] = True
    return df


# ---------------- plots ----------------
def _med(df, by):
    cols = [c for c in ["out_throughput", "tpot_ms", "ttft_ms", "preempt_delta",
                        "cache_usage", "kv_tokens"]
            if c in df.columns]
    return df.groupby(by, as_index=False)[cols].median()


def fig1_thr_vs_conc(df, outdir):
    g = _med(df, ["util", "ctx", "conc"])
    utils = sorted(g.util.unique())
    n = len(utils)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.2 * nrows),
                             sharey=True, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, u in zip(axes.ravel(), utils):
        sub = g[g.util == u]
        for ctx in sorted(sub.ctx.unique()):
            s = sub[sub.ctx == ctx].sort_values("conc")
            c = CTX_COLORS.get(int(ctx), MUTED)
            ax.plot(s.conc, s.out_throughput, "-o", ms=4.5, lw=2, color=c,
                    label=f"ctx {int(ctx)}")
            ax.annotate(f"{s.out_throughput.iloc[-1]:.0f}",
                        (s.conc.iloc[-1], s.out_throughput.iloc[-1]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8, color=c)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub.conc.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(f"KV 预算 util={u:.2f}")
        ax.set_xlabel("并发（max-concurrency）")
        if ax in axes[:, 0]:
            ax.set_ylabel("吞吐（output tok/s）")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Fig 1｜吞吐-并发：KV 预算越大，拐点越靠右", y=1.01,
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_throughput_vs_concurrency.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig2_thr_vs_kv(df, outdir):
    g = _med(df, ["util", "ctx", "conc"])
    conc_max = g.conc.max()
    sub = g[g.conc == conc_max]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for ctx in sorted(sub.ctx.unique()):
        s = sub[sub.ctx == ctx].sort_values("kv_tokens")
        if s.kv_tokens.max() <= 0:
            continue
        c = CTX_COLORS.get(int(ctx), MUTED)
        ax.plot(s.kv_tokens / 1e6, s.out_throughput, "-o", ms=6, lw=2, color=c,
                label=f"ctx {int(ctx)}")
    ax.set_xlabel("实际 KV 容量（百万 token，来自启动日志）")
    ax.set_ylabel(f"吞吐（output tok/s，conc={int(conc_max)}）")
    ax.set_title("Fig 2｜吞吐随 KV 容量上升并饱和")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_throughput_vs_kv_capacity.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig3_preempt(df, outdir):
    g = _med(df, ["util", "ctx", "conc"])
    ctx_sel = int(g.ctx.max())
    sub = g[g.ctx == ctx_sel]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax2 = ax.twinx()
    for i, u in enumerate(sorted(sub.util.unique())):
        s = sub[sub.util == u].sort_values("conc")
        c = TIER[i % len(TIER)]
        ax.plot(s.conc, s.preempt_delta, "-o", ms=5, lw=2, color=c, label=f"preempt  util={u:.2f}")
        ax2.plot(s.conc, s.out_throughput, "--", lw=1.2, alpha=0.55, color=c)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub.conc.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("并发（max-concurrency）")
    ax.set_ylabel("preemption 增量（次）")
    ax2.set_ylabel("吞吐（output tok/s，虚线）")
    ax2.grid(False)
    ax.set_title(f"Fig 3｜拐点与 preemption 起跳点重合（ctx={ctx_sel}）")
    h1, l1 = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h1.append(Line2D([0], [0], ls="--", lw=1.2, color=MUTED))
    l1.append("吞吐（右轴，虚线）")
    ax.legend(h1, l1, loc="upper left", fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_preemption_vs_concurrency.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig4_thr_vs_ctx(df, outdir):
    g = _med(df, ["util", "ctx", "conc"])
    conc_max = g.conc.max()
    sub = g[g.conc == conc_max]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for i, u in enumerate(sorted(sub.util.unique())):
        s = sub[sub.util == u].sort_values("ctx")
        c = TIER[i % len(TIER)]
        ax.plot(s.ctx, s.out_throughput, "-o", ms=6, lw=2, color=c, label=f"util={u:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub.ctx.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("上下文长度（token）")
    ax.set_ylabel(f"吞吐（output tok/s，conc={int(conc_max)}）")
    ax.set_title("Fig 4｜上下文变长吞吐下降，KV 预算大可缓解")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_throughput_vs_context.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def composite(paths, outdir):
    from matplotlib import image as mpimg
    imgs = [mpimg.imread(p) for p in paths]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout()
    p = os.path.join(outdir, "fig0_overview.png")
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", metavar="RAWDIR")
    ap.add_argument("--out")
    ap.add_argument("--csv")
    ap.add_argument("--outdir", default="results/figs")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.ingest:
        ingest(a.ingest, a.out or "results/results.csv")
        return

    if a.demo:
        df = demo_data()
        print("[demo] 使用合成数据预览形状（不是真实结果）")
    elif a.csv:
        df = pd.read_csv(a.csv)
        df = df[df.get("rc", 1) == 0] if "rc" in df.columns else df
        if df.empty:
            sys.exit("CSV 里没有 rc=0 的有效行")
    else:
        sys.exit("需要 --csv 或 --demo")

    os.makedirs(a.outdir, exist_ok=True)
    style()
    paths = [fig1_thr_vs_conc(df, a.outdir), fig2_thr_vs_kv(df, a.outdir),
             fig3_preempt(df, a.outdir), fig4_thr_vs_ctx(df, a.outdir)]
    comp = composite(paths, a.outdir)
    for p in paths + [comp]:
        print("[fig]", p)


if __name__ == "__main__":
    main()
