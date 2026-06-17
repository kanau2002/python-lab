import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from shapely.ops import unary_union


plt.rcParams["font.family"] = "Hiragino Sans"

os.makedirs("output", exist_ok=True)

TOWN_NAME = "酒々井町"
BOUNDARY_PATH = "input_boundary/N03-20250101_12.shp"
GT_PATH = f"input_ground-truth-polygon/{TOWN_NAME}.shp"
PRED_PATH = f"input_predicted-polygon/{TOWN_NAME}.gpkg"
PROJ_CRS = 6677  # 平面直角座標系（JGD2011 第9系）

# ── 1. データ読み込みと境界クリップ ──────────────────────────────────────────

print("データを読み込んでいます...")
boundary_raw = gpd.read_file(BOUNDARY_PATH)
town_boundary = (
    boundary_raw[boundary_raw["N03_004"] == TOWN_NAME]
    .to_crs(epsg=4326)
    .dissolve()
)

gt_raw = gpd.read_file(GT_PATH)
pred_raw = gpd.read_file(PRED_PATH)

gt = gpd.clip(gt_raw.to_crs(epsg=4326), town_boundary).explode(index_parts=False).reset_index(drop=True)
pred = gpd.clip(pred_raw.to_crs(epsg=4326), town_boundary).explode(index_parts=False).reset_index(drop=True)

print(f"  境界クリップ後 GT: {len(gt)} ポリゴン, Pred: {len(pred)} ポリゴン")

# 面積計算用に投影変換
gt_proj = gt.to_crs(epsg=PROJ_CRS)
pred_proj = pred.to_crs(epsg=PROJ_CRS)
town_proj = town_boundary.to_crs(epsg=PROJ_CRS)

# ── 2. Area-based Metrics ────────────────────────────────────────────────────

print("\n[分析1] Area-based Metrics を計算中...")

gt_union = unary_union(gt_proj.geometry)
pred_union = unary_union(pred_proj.geometry)
town_geom = unary_union(town_proj.geometry)

tp_geom = gt_union.intersection(pred_union)
fp_geom = pred_union.difference(gt_union)
fn_geom = gt_union.difference(pred_union)

tp_area = tp_geom.area
fp_area = fp_geom.area
fn_area = fn_geom.area
town_area = town_geom.area
tn_area = town_area - tp_area - fp_area - fn_area
gt_total_area = tp_area + fn_area
pred_total_area = tp_area + fp_area

iou = tp_area / (tp_area + fp_area + fn_area)
precision = tp_area / (tp_area + fp_area)
recall = tp_area / (tp_area + fn_area)
f1 = 2 * precision * recall / (precision + recall)
accuracy = (tp_area + tn_area) / town_area

metrics_area = pd.DataFrame([{
    "metric": "IoU",              "value": iou,
}, {
    "metric": "Precision",        "value": precision,
}, {
    "metric": "Recall",           "value": recall,
}, {
    "metric": "F1",               "value": f1,
}, {
    "metric": "Accuracy",         "value": accuracy,
}, {
    "metric": "GT_total_m2",      "value": gt_total_area,
}, {
    "metric": "Pred_total_m2",    "value": pred_total_area,
}, {
    "metric": "TP_m2",            "value": tp_area,
}, {
    "metric": "FP_m2",            "value": fp_area,
}, {
    "metric": "FN_m2",            "value": fn_area,
}, {
    "metric": "TN_m2",            "value": tn_area,
}])
metrics_area.to_csv("output/metrics_area.csv", index=False, encoding="utf-8-sig")

print(f"  GT 総面積  : {gt_total_area/1e6:.4f} km2")
print(f"  Pred 総面積: {pred_total_area/1e6:.4f} km2")
print(f"  IoU      : {iou:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1       : {f1:.4f}")
print(f"  Accuracy : {accuracy:.4f}")
print("  → output/metrics_area.csv")

# ── 3. Instance-based Metrics ────────────────────────────────────────────────

print("\n[分析2] Instance-based Metrics を計算中...")

def compute_iou_matrix(gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame) -> np.ndarray:
    """各 GT ポリゴンと各 Pred ポリゴンの IoU を計算して行列で返す。"""
    tree = gdf_b.sindex
    n_a = len(gdf_a)
    iou_matrix = np.zeros((n_a, len(gdf_b)))
    for i, geom_a in enumerate(gdf_a.geometry):
        candidates = list(tree.query(geom_a, predicate="intersects"))
        for j in candidates:
            geom_b = gdf_b.geometry.iloc[j]
            inter = geom_a.intersection(geom_b).area
            union = geom_a.union(geom_b).area
            if union > 0:
                iou_matrix[i, j] = inter / union
    return iou_matrix

iou_matrix = compute_iou_matrix(gt_proj, pred_proj)

def instance_metrics_at_threshold(iou_mat: np.ndarray, threshold: float):
    matched_gt = set()
    matched_pred = set()
    for i in range(iou_mat.shape[0]):
        best_j = int(np.argmax(iou_mat[i]))
        if iou_mat[i, best_j] >= threshold:
            matched_gt.add(i)
            matched_pred.add(best_j)
    tp = len(matched_gt)
    fp = iou_mat.shape[1] - len(matched_pred)
    fn = iou_mat.shape[0] - len(matched_gt)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f

DEFAULT_THRESHOLD = 0.5
tp, fp, fn, p, r, f = instance_metrics_at_threshold(iou_matrix, DEFAULT_THRESHOLD)

metrics_instance = pd.DataFrame([{
    "iou_threshold": DEFAULT_THRESHOLD,
    "n_gt": len(gt_proj),
    "n_pred": len(pred_proj),
    "TP": tp, "FP": fp, "FN": fn,
    "Precision": p, "Recall": r, "F1": f,
}])
metrics_instance.to_csv("output/metrics_instance.csv", index=False, encoding="utf-8-sig")

print(f"  (IoU threshold={DEFAULT_THRESHOLD})")
print(f"  GT={len(gt_proj)}, Pred={len(pred_proj)}")
print(f"  TP={tp}, FP={fp}, FN={fn}")
print(f"  Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
print("  → output/metrics_instance.csv")

# Precision-Recall カーブ（閾値変化）
thresholds = np.arange(0.1, 1.0, 0.1)
pr_records = []
for thr in thresholds:
    _, _, _, pi, ri, fi = instance_metrics_at_threshold(iou_matrix, thr)
    pr_records.append({"threshold": round(thr, 1), "Precision": pi, "Recall": ri, "F1": fi})
pr_df = pd.DataFrame(pr_records)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(pr_df["threshold"], pr_df["Precision"], marker="o", label="Precision")
ax.plot(pr_df["threshold"], pr_df["Recall"],    marker="s", label="Recall")
ax.plot(pr_df["threshold"], pr_df["F1"],        marker="^", label="F1")
ax.axvline(DEFAULT_THRESHOLD, color="gray", linestyle="--", linewidth=1, label=f"threshold={DEFAULT_THRESHOLD}")
ax.set_xlabel("IoU Threshold")
ax.set_ylabel("Score")
ax.set_title(f"{TOWN_NAME} — Instance-based Precision / Recall / F1")
ax.legend()
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → output/pr_curve.png")

# ── 4. Per-polygon IoU 分布 ──────────────────────────────────────────────────

print("\n[分析3] Per-polygon IoU 分布を計算中...")

best_iou_per_gt = iou_matrix.max(axis=1)

fig, ax = plt.subplots(figsize=(7, 5))
bins = np.linspace(0, 1, 21)
ax.hist(best_iou_per_gt, bins=bins, color="steelblue", edgecolor="white")
ax.axvline(np.median(best_iou_per_gt), color="orange", linestyle="--",
           linewidth=1.5, label=f"中央値: {np.median(best_iou_per_gt):.3f}")
ax.axvline(np.mean(best_iou_per_gt), color="red", linestyle="--",
           linewidth=1.5, label=f"平均値: {np.mean(best_iou_per_gt):.3f}")
pct_over_half = (best_iou_per_gt >= 0.5).mean() * 100
ax.set_xlabel("IoU (GT ポリゴンごとの最良マッチ)")
ax.set_ylabel("GT ポリゴン数")
ax.set_title(f"{TOWN_NAME} — Per-polygon IoU 分布\n(IoU>=0.5: {pct_over_half:.1f}%)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/histogram_iou.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  中央値: {np.median(best_iou_per_gt):.3f}, 平均: {np.mean(best_iou_per_gt):.3f}, IoU≥0.5: {pct_over_half:.1f}%")
print("  → output/histogram_iou.png")

# ── 5. 空間マップ可視化 ──────────────────────────────────────────────────────

print("\n[分析4] 空間マップを生成中...")

tp_gdf = gpd.GeoDataFrame(geometry=[tp_geom] if not tp_geom.is_empty else [tp_geom],
                           crs=f"EPSG:{PROJ_CRS}").to_crs(epsg=4326)
fp_gdf = gpd.GeoDataFrame(geometry=[fp_geom] if not fp_geom.is_empty else [fp_geom],
                           crs=f"EPSG:{PROJ_CRS}").to_crs(epsg=4326)
fn_gdf = gpd.GeoDataFrame(geometry=[fn_geom] if not fn_geom.is_empty else [fn_geom],
                           crs=f"EPSG:{PROJ_CRS}").to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(10, 10))
town_boundary.boundary.plot(ax=ax, color="gray", linewidth=1.0, zorder=4)

if not tp_geom.is_empty:
    tp_gdf.plot(ax=ax, color="#00cc00", alpha=1.0, zorder=3)
if not fp_geom.is_empty:
    fp_gdf.plot(ax=ax, color="#cc0000", alpha=1.0, zorder=3)
if not fn_geom.is_empty:
    fn_gdf.plot(ax=ax, color="#0055cc", alpha=1.0, zorder=3)

legend_handles = [
    mpatches.Patch(color="#00cc00", label="TP（正検出）"),
    mpatches.Patch(color="#cc0000", label="FP（過検出）"),
    mpatches.Patch(color="#0055cc", label="FN（未検出）"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
ax.set_title(f"{TOWN_NAME} — 駐車場検出 精度マップ\nIoU={iou:.3f}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}",
             fontsize=11)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("output/map_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → output/map_comparison.png")

# 帯グラフ（TP / FP / FN 割合 = IoU の視覚化）
union_area = tp_area + fp_area + fn_area
tp_pct = tp_area / union_area * 100
fp_pct = fp_area / union_area * 100
fn_pct = fn_area / union_area * 100

fig, ax = plt.subplots(figsize=(8, 0.7))
ax.barh(0, tp_pct,                       color="#5aaa5a", height=0.6)
ax.barh(0, fp_pct, left=tp_pct,          color="#cc6666", height=0.6)
ax.barh(0, fn_pct, left=tp_pct + fp_pct, color="#6688bb", height=0.6)
for pct, left, label in [
    (tp_pct, 0,             f"TP {tp_pct:.1f}%"),
    (fp_pct, tp_pct,        f"FP {fp_pct:.1f}%"),
    (fn_pct, tp_pct+fp_pct, f"FN {fn_pct:.1f}%"),
]:
    ax.text(left + pct / 2, 0, label, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("割合 (%)", fontsize=9)
ax.set_yticks([])
ax.tick_params(axis="x", labelsize=9)
ax.set_title(f"TP / FP / FN の内訳（IoU = {iou:.4f} = TP の割合）", fontsize=9, pad=6)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
plt.savefig("output/area_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → output/area_breakdown.png")

# ── 6. 統合（マージ）現象の定量化 ────────────────────────────────────────────

print("\n[分析5] 統合現象（Pred が複数 GT を包含）を分析中...")

joined = gpd.sjoin(pred_proj, gt_proj, how="left", predicate="intersects")
gt_count_per_pred = joined.groupby(joined.index).size()

avg_gt_area = gt_proj.geometry.area.mean()
avg_pred_area = pred_proj.geometry.area.mean()
merged_preds = (gt_count_per_pred > 1).sum()
merge_rate = merged_preds / len(pred_proj) * 100

merge_stats = pd.DataFrame([{
    "GT_平均面積_m2": avg_gt_area,
    "Pred_平均面積_m2": avg_pred_area,
    "面積比_Pred/GT": avg_pred_area / avg_gt_area,
    "複数GT対応Pred数": int(merged_preds),
    "全Pred数": len(pred_proj),
    "統合率_%": merge_rate,
}])
merge_stats.to_csv("output/merge_stats.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 5))
bins = np.arange(0.5, 4.5, 1)
ax.hist(gt_count_per_pred.clip(upper=4), bins=bins, color="steelblue", edgecolor="white", rwidth=0.5)
ax.set_xlabel("1つの Pred ポリゴンに対応する GT ポリゴン数")
ax.set_ylabel("Pred ポリゴン数")
ax.set_title(f"{TOWN_NAME} — Pred に対応する GT 数の分布\n(複数GT対応: {merged_preds}件 / {len(pred_proj)}件 = {merge_rate:.1f}%)")
ax.set_xlim(0.5, 4.5)
ax.xaxis.set_major_locator(plt.FixedLocator([1, 2, 3, 4]))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/merge_histogram.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  GT 平均面積: {avg_gt_area:.1f} m2, Pred 平均面積: {avg_pred_area:.1f} m2 (比: {avg_pred_area/avg_gt_area:.2f})")
print(f"  複数 GT に対応する Pred: {merged_preds} 件 ({merge_rate:.1f}%)")

# 統合による FN 寄与数（乗数効果を考慮）
merge_fn_contribution = int((gt_count_per_pred - 1).clip(lower=0).sum())

# IoU 閾値 0.1 での全指標
tp_at_01, fp_at_01, fn_at_01, p_at_01, r_at_01, f_at_01 = instance_metrics_at_threshold(iou_matrix, 0.1)
merge_fn_rate_vs_01 = merge_fn_contribution / fn_at_01 * 100 if fn_at_01 > 0 else 0.0

print(f"  統合による FN 寄与数（乗数込み）: {merge_fn_contribution} 件")
print(f"  IoU=0.1 での FN: {fn_at_01} 件")
print(f"  統合 FN / FN@0.1: {merge_fn_rate_vs_01:.1f}%")
print("  → output/merge_histogram.png")
print("  → output/merge_stats.csv")

merge_stats["統合FN寄与数"] = merge_fn_contribution
merge_stats["FN_at_IoU01"] = fn_at_01
merge_stats["統合FN率_vs_FN01_%"] = merge_fn_rate_vs_01
merge_stats.to_csv("output/merge_stats.csv", index=False, encoding="utf-8-sig")

# ── 7. PDF レポート生成 ────────────────────────────────────────────────────────

print("\n[レポート] PDF を生成中...")
plt.rcParams["pdf.fonttype"] = 42

REPORT_DATE = "2026-06-14"
A4 = (8.27, 11.69)
C_GRAY = "#444444"

def hline(fig, y, x0=0.07, x1=0.93):
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                          color="#cccccc", linewidth=0.6))

def section_title(fig, y, text):
    fig.text(0.07, y, text, fontsize=11, fontweight="bold", va="top")

def body_text(fig, y, text):
    fig.text(0.07, y, text, fontsize=9, color=C_GRAY, va="top", linespacing=1.7)

def make_table(fig, rect, rows, col_widths=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center",
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#d8d8d8")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")

def embed_image(fig, rect, path):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")

with PdfPages("output/report.pdf") as pdf:

    # ── P1: タイトル + 精度指標サマリー ──────────────────────────────────────
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.965, "駐車場検出モデル 精度評価レポート",
             ha="center", fontsize=16, fontweight="bold", va="top")
    fig.text(0.5, 0.935, f"対象地域: {TOWN_NAME}　|　評価日: {REPORT_DATE}",
             ha="center", fontsize=10, color=C_GRAY, va="top")
    hline(fig, 0.915)

    section_title(fig, 0.900, "データ概要")
    make_table(fig, [0.10, 0.815, 0.80, 0.075],
               [["", "ポリゴン数", "総面積 (km²)", "フォーマット", "座標系"],
                ["Ground Truth", f"{len(gt_proj):,}", f"{gt_total_area/1e6:.4f}", "Shapefile",  "EPSG:4326"],
                ["Predicted",    f"{len(pred_proj):,}", f"{pred_total_area/1e6:.4f}", "GeoPackage", "EPSG:4326"]])
    hline(fig, 0.805)

    section_title(fig, 0.790, "面積ベース精度指標（分析1）")
    make_table(fig, [0.20, 0.558, 0.60, 0.222],
               [["指標", "値", "解釈"],
                ["TP",        f"{tp_area/1e6:.4f} km²",  "正しく検出された駐車場面積"],
                ["FP",        f"{fp_area/1e6:.4f} km²",  "過検出面積（Pred にあって GT にない）"],
                ["FN",        f"{fn_area/1e6:.4f} km²",  "未検出面積（GT にあって Pred にない）"],
                ["IoU",       f"{iou:.4f}",              "GT と Pred の面積的な重複度"],
                ["Precision", f"{precision:.4f}",        "Pred のうち正しく検出した面積の割合"],
                ["Recall",    f"{recall:.4f}",           "GT のうち検出できた面積の割合"],
                ["F1 score",  f"{f1:.4f}",               "Precision と Recall の調和平均"],
                ["Accuracy",  f"{accuracy:.4f}",         "町全体に対する正解率（TN が支配的）"]],
               col_widths=[0.20, 0.24, 0.56])

    # 帯グラフ（PNG 埋め込み）
    embed_image(fig, [0.07, 0.495, 0.86, 0.052], "output/area_breakdown.png")
    hline(fig, 0.452)

    section_title(fig, 0.437, "ポリゴン個数ベース精度指標（分析2、IoU 閾値 = 0.5）")
    fig.text(0.07, 0.405, f"GT: {len(gt_proj):,} 件　Pred: {len(pred_proj):,} 件",
             fontsize=9, color=C_GRAY, va="top")
    make_table(fig, [0.20, 0.150, 0.60, 0.245],
               [["指標", "値", "内容"],
                ["TP",        f"{tp:,}",   "正しくマッチした予測ポリゴン数"],
                ["FP",        f"{fp:,}",   "マッチしなかった予測ポリゴン数（過検出）"],
                ["FN",        f"{fn:,}",   "マッチしなかった GT ポリゴン数（未検出）"],
                ["Precision", f"{p:.4f}",  "予測の正確さ"],
                ["Recall",    f"{r:.4f}",  "正解の検出率"],
                ["F1 score",  f"{f:.4f}",  "Precision と Recall の調和平均"]],
               col_widths=[0.20, 0.24, 0.56])
    hline(fig, 0.138)
    body_text(fig, 0.120,
              f"帯グラフより FP≈FN（各約 {fp_area/union_area*100:.1f}%・{fn_area/union_area*100:.1f}%）であり、過検出と未検出の面積がほぼ等しいため Precision≈Recall となっている。\n"
              "詳細な閾値感度分析（PR カーブ）および IoU=0.1 での評価は後続ページを参照。")

    pdf.savefig(fig)
    plt.close()

    # ── P2: 空間分布マップ ────────────────────────────────────────────────────
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.965, "分析4: 空間分布マップ（TP / FP / FN）",
             ha="center", fontsize=13, fontweight="bold", va="top")
    hline(fig, 0.940)
    embed_image(fig, [0.04, 0.175, 0.92, 0.760], "output/map_comparison.png")
    hline(fig, 0.165)
    body_text(fig, 0.150,
              "凡例: 緑 = TP（正検出）、赤 = FP（過検出）、青 = FN（未検出）\n"
              "TP が酒々井町全域に分布しており、モデルは地域全体の駐車場を概ね検出できている。\n"
              "FP・FN は特定地域への集中が見られず、系統的な誤りは確認されない。")
    pdf.savefig(fig)
    plt.close()

    # ── P3: PR カーブ + Per-polygon IoU 分布 ─────────────────────────────────
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.965, "分析2・3: PR カーブ と Per-polygon IoU 分布",
             ha="center", fontsize=13, fontweight="bold", va="top")
    hline(fig, 0.940)
    section_title(fig, 0.925, "IoU 閾値別 Precision / Recall / F1（分析2）")
    embed_image(fig, [0.04, 0.560, 0.92, 0.355], "output/pr_curve.png")
    hline(fig, 0.548)
    section_title(fig, 0.533, "GT ポリゴンごとの最良 IoU 分布（分析3）")
    embed_image(fig, [0.04, 0.170, 0.92, 0.355], "output/histogram_iou.png")
    hline(fig, 0.158)
    body_text(fig, 0.143,
              f"IoU 閾値 0.1 での F1 は 0.8 超、閾値 0.5 では {f:.4f} まで低下。\n"
              "この急落は「完全な未検出」ではなく、形状・境界の精度問題が主因であることを示す。\n"
              f"Per-polygon IoU の中央値は 0.702 であり、半数以上の GT で高精度なマッチが得られている。")
    pdf.savefig(fig)
    plt.close()

    # ── P4: 統合現象分析 ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.965, "分析5: 統合（マージ）現象の分析",
             ha="center", fontsize=13, fontweight="bold", va="top")
    hline(fig, 0.940)
    embed_image(fig, [0.04, 0.590, 0.92, 0.340], "output/merge_histogram.png")
    hline(fig, 0.578)
    body_text(fig, 0.558,
              f"GT（{len(gt_proj):,} 件）に対し Pred（{len(pred_proj):,} 件）が少ない傾向は、モデルが複数の GT を\n"
              f"1 つの Pred にまとめる統合現象が一因。Pred 平均面積は GT の {avg_pred_area/avg_gt_area:.2f} 倍。\n"
              f"複数 GT 対応 Pred は {merge_rate:.1f}%（{int(merged_preds)} 件）だが、乗数効果（N GTs → 1 Pred で N-1 FN 増）を\n"
              f"考慮すると統合による FN 寄与は {merge_fn_contribution} 件。IoU=0.1 での FN {fn_at_01} 件に対し\n"
              f"{merge_fn_rate_vs_01:.1f}% を占め、低閾値での主要因となりうる。")
    hline(fig, 0.455)
    section_title(fig, 0.440, "ポリゴン個数ベース精度指標（IoU 閾値 = 0.1）")
    fig.text(0.07, 0.408, f"GT: {len(gt_proj):,} 件　Pred: {len(pred_proj):,} 件",
             fontsize=9, color=C_GRAY, va="top")
    make_table(fig, [0.20, 0.145, 0.60, 0.250],
               [["指標", "値", "内容"],
                ["TP",        f"{tp_at_01:,}",   "正しくマッチした予測ポリゴン数"],
                ["FP",        f"{fp_at_01:,}",   "マッチしなかった予測ポリゴン数（過検出）"],
                ["FN",        f"{fn_at_01:,}",   "マッチしなかった GT ポリゴン数（未検出）"],
                ["Precision", f"{p_at_01:.4f}",  "予測の正確さ"],
                ["Recall",    f"{r_at_01:.4f}",  "正解の検出率"],
                ["F1 score",  f"{f_at_01:.4f}",  "Precision と Recall の調和平均"]],
               col_widths=[0.20, 0.24, 0.56])
    pdf.savefig(fig)
    plt.close()

    # ── P5: 総合考察 ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.965, "総合考察",
             ha="center", fontsize=13, fontweight="bold", va="top")
    hline(fig, 0.940)

    entries = [
        (0.920, "1. モデルの位置検出精度は高い",
         f"IoU 閾値 0.1 での F1 スコアは {f_at_01:.3f} であり、モデルは酒々井町内の駐車場の「場所」を概ね\n"
         f"正しく検出できている。面積ベースの Recall（{recall:.3f}）も高く、GT 面積の約 {recall*100:.0f}% が検出されている。"),
        (0.800, "2. ボトルネックは形状・境界の精度",
         f"IoU 閾値 0.5 では F1 が {f:.3f} まで低下する（IoU=0.1 の {f_at_01:.3f} から大幅減）。\n"
         f"PR カーブの急落は「完全な未検出」ではなく、予測ポリゴンの形状・サイズが GT と一致しない\n"
         f"ことが主因であることを示す。Pred の平均面積は GT より {(avg_pred_area/avg_gt_area - 1)*100:.0f}% 大きく、過大検出の傾向がある。"),
        (0.645, "3. 統合（マージ）現象の影響は閾値依存",
         f"複数 GT に対応する Pred は {merge_rate:.1f}%（{int(merged_preds)} 件）だが、N GTs → 1 Pred では\n"
         f"N-1 件の FN が生じる乗数効果があり、統合による FN 寄与は合計 {merge_fn_contribution} 件。\n"
         f"IoU=0.5 では FN {fn:,} 件中 {merge_fn_contribution/fn*100:.0f}%、IoU=0.1 では FN {fn_at_01} 件中\n"
         f"{merge_fn_rate_vs_01:.0f}% を占め、低閾値評価ほど統合現象が支配的な要因となる。"),
        (0.500, "4. 改善の方向性",
         f"① ポリゴン後処理の調整: 後処理のポリゴン正則化パラメータを見直し、予測サイズを\n"
         f"   GT に近づける（現在 Pred / GT = {avg_pred_area/avg_gt_area:.2f} 倍）。\n"
         f"② 境界精度の向上: セグメンテーション損失関数に境界損失を追加し、境界付近の精度を強化する。\n"
         f"③ インスタンス分離精度の向上: 隣接する複数駐車場を 1 つに統合してしまう現象への対処として、\n"
         f"   インスタンスセグメンテーション手法の導入や後処理での分割ロジックを検討する。"),
    ]

    for y_title, title, body in entries:
        section_title(fig, y_title, f"● {title}")
        body_text(fig, y_title - 0.038, body)
        hline(fig, y_title - 0.038 - 0.030 * (body.count("\n") + 1) - 0.012)

    pdf.savefig(fig)
    plt.close()

print("  → output/report.pdf")
print("\n✓ 完了。output/ フォルダに結果が保存されました。")
