from pathlib import Path

import polars as pl
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


from cedalion.vis.anatomy import sensitivity_matrix
from cedalion import dot
from cedalion.dataclasses import PointType
from cedalion.io import read_snirf, load_Adot


IMG_PATH: Path = Path("img")
PPT: pl.DataFrame = pl.read_csv(Path("data/participants.csv"))
DPI: int = 1_000

COND_COLOR: dict[str, str] = {"ST": "tab:blue", "DW": "tab:orange", "DS": "tab:green"}


def plot_demo(ppt: pl.DataFrame):
    fig, axes = plt.subplots(1, 3, sharey=False, sharex=False, figsize=(9, 3))

    vars = {
        "age": "Age (hours)",
        "gest_weeks": "Gestation weeks",
        "birth_weight": "Birth weight (g)",
    }
    for (var, var_label), ax in zip(vars.items(), axes):
        avg, sd = ppt[var].mean(), ppt[var].std()
        n, *_ = ax.hist(ppt[var].floor(), lw=1, edgecolor="w", color="tab:grey")
        ax.errorbar(x=avg, xerr=sd, y=n.max() * 1.1, c="k", capsize=3, lw=2, capthick=2)
        ax.scatter(avg, n.max() * 1.1, c="k", s=50)
        ax.text(avg, y=n.max() * 1.2, s=f"M={avg:.2f} (SD={sd:.2f})", ha="center")
        ax.set_title(var_label)
        ax.set_ybound(0, n.max() * 1.4)
        ax.set_yticks(range(0, round((n.max() * 1.4)), 5))

    fig.savefig(IMG_PATH / "demo.png", dpi=DPI)


def plot_sensitivity(
    path: Path = Path("data/sub-003_task-words_nirs.snirf"),
    sensitivity_path: Path = Path("data/sensitivity.h5"),
    fluence_path: Path = Path("data/fluence.h5"),
):
    rec = read_snirf(path)[0]
    head_ijk = dot.get_standard_headmodel("icbm152")
    geo_3d = rec.geo3d
    head_ijk = dot.get_standard_headmodel("icbm152")
    geo3d_snapped_ijk = head_ijk.align_and_snap_to_scalp(geo_3d)

    sensitivity_path, fluence_path = Path(sensitivity_path), Path(fluence_path)
    fwm = dot.ForwardModel(head_ijk, geo3d_snapped_ijk, rec._measurement_lists["amp"])

    if not sensitivity_path.exists():
        fwm.compute_sensitivity(fluence_path, sensitivity_path)

    Adot = load_Adot(str(sensitivity_path))
    geo3d_plot = geo3d_snapped_ijk[geo3d_snapped_ijk.type != PointType.LANDMARK]

    plotter = sensitivity_matrix.Main(
        sensitivity=Adot,
        brain_surface=head_ijk.brain,
        head_surface=head_ijk.scalp,
        labeled_points=geo3d_plot,
        wavelength=850,
    )
    plotter.plot(high_th=0, low_th=-3)
    plotter.plt.show()


def plot_nblocks(ppt: pd.DataFrame):

    trial_types = ["ST", "DW", "DS"]

    data = ppt.filter(pl.col("status") == "included").select(
        pl.col([f"n_epochs_{tt.lower()}" for tt in trial_types])
    )
    data = data.to_numpy()

    fig, ax = plt.subplots(1, 1)

    ax.hist(
        data,
        label=trial_types,
        histtype="bar",
        stacked=True,
        linewidth=3,
        edgecolor="w",
    )
    ax.set_xlabel("Number of valid blocks")
    fig.legend()
    fig.savefig(IMG_PATH / "nblocks.png", dpi=DPI)


def plot_inclusion(ppt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))

    lookup = {
        "included": "Included",
        "no_data": "Crying/fussy",
        "gest_weeks": "Gestational weeks",
        "birth_weight": "Birth weight",
        "missing_condition": "Missing condition",
        "min_epochs": "Minimum epochs",
    }

    counts = ppt["status"].value_counts(sort=True)
    values = counts["count"].to_list()
    labels = counts["status"].to_list()
    labels = [f"{lookup[k]} (n = {v})" for k, v in zip(labels, values)]
    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.5), startangle=-40)
    n_exc = sum(v for k, v in zip(labels, values) if "Included" not in k)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="w", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        ha = {-1: "right", 1: "left"}[int(np.sign(x))]
        connection = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connection})
        xy_text = (1.35 * np.sign(x), 1.4 * y)
        ax.annotate(labels[i], xy=(x, y), xytext=xy_text, ha=ha, **kw)

    ax.set_title(f"Attrition ({n_exc / len(ppt):.2%})")
    fig.savefig(IMG_PATH / "inclusion.png", dpi=DPI)


if __name__ == "__main__":
    plot_demo(PPT)
    plot_inclusion(PPT)
    plot_nblocks(PPT)
