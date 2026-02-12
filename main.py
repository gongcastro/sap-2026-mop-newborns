from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt
from xarray import DataArray
from cedalion.vis.anatomy import sensitivity_matrix
from cedalion import dot
from cedalion.dataclasses import PointType
from cedalion.io import read_snirf, load_Adot


IMG_PATH: Path = Path("img")
PPT: pl.DataFrame = pl.read_csv(Path("data/participants.tsv"), separator="\t")
DPI: int = 1_000


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


if __name__ == "__main__":
    plot_demo(PPT)
    plot_sensitivity()
