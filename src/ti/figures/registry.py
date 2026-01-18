from ti.figures import (
    fig_bonus_invariance,
    fig_convergence,
    fig_corr_repr_vs_bonus,
    fig_crtr_over_time,
    fig_elliptical_heatmaps,
    fig_end2end,
    fig_envs,
    fig_hparam_sweep,
    fig_rep_factor_sweep,
    fig_rep_sweep,
    fig_repr_invariance,
    fig_spearman,
    fig_teacup_analysis,
    fig_teacup_only,
    fig_tsne,
    fig_xy_regression,
)


REGISTRY = {
    "tsne": fig_tsne.run,
    "envs_overview": fig_envs.run,
    "rep_sweep": fig_rep_sweep.run,
    "repr_invariance": fig_repr_invariance.run,
    "rep_factor_sweep": fig_rep_factor_sweep.run,
    "crtr_over_time": fig_crtr_over_time.run,
    "xy_regression": fig_xy_regression.run,
    "elliptical_heatmaps": fig_elliptical_heatmaps.run,
    "bonus_invariance": fig_bonus_invariance.run,
    "corr_repr_vs_bonus": fig_corr_repr_vs_bonus.run,
    "teacup_analysis": fig_teacup_analysis.run,
    "teacup_elliptical_only": fig_teacup_only.run,
    "end2end": fig_end2end.run,
    "convergence": fig_convergence.run,
    "spearman": fig_spearman.run,
    "hparam_sweep": fig_hparam_sweep.run,
}


def get_generator(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown figure generator: {name}")
    return REGISTRY[name]
