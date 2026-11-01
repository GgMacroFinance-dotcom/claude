"""P9 后处理：读取 p9_grids.npz，补充外部边界的线性外推（R4）、储备余量敏感性，重画图 P9-1。"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotstyle import setup, COL
from p9_constraints import S_GRID, PI_BAR, FX_ROOM, RHO_C, XMAX, boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
setup()
G = np.load(os.path.join(OUT, 'p9_grids.npz'))


def grid(kind, s):
    return G[f'{kind}_{s}_X'], G[f'{kind}_{s}_U']


def ext_linear(kind, s, room):
    """外部边界：可解区间内绑定则用插值，否则按最后 5 个网格点的平均斜率线性外推（R4）。"""
    Xs, U = grid(kind, s)
    b = boundaries(kind, fx_room=room, grid=(Xs, U))[1]
    if np.isfinite(b):
        return b, False
    k = min(5, len(Xs) - 1)
    slope = (U[-1, 1] - U[-1 - k, 1]) / (Xs[-1] - Xs[-1 - k])
    return Xs[-1] + (room - U[-1, 1]) / slope, True


def main():
    rows = []
    for s in S_GRID:
        for kind in ['consumption', 'green']:
            Xs, U = grid(kind, s)
            b = boundaries(kind, grid=(Xs, U))
            trunc = Xs[-1] < XMAX - 1e-9           # 闲置资源先耗尽（u → 1），更大的发行没有解
            for share in [1.0, 0.5, 0.25]:
                ext, extrap = ext_linear(kind, s, FX_ROOM * share)
                carbon = b[2] if np.isfinite(b[2]) else (np.inf if U[-1, 2] <= 0 else (np.nan if trunc else np.inf))
                rows.append(dict(s=s, kind=kind, fx_room_pctGDP=100 * FX_ROOM * share / 4,
                                 X_slack=b[0], X_ext=ext, ext_extrapolated=extrap, X_carbon=carbon,
                                 lam_imp=U[-1, 1] / Xs[-1] / 4 if Xs[-1] > 0 else np.nan))
    with open(os.path.join(OUT, 'p9_fxroom_sensitivity.csv'), 'w') as f:
        f.write('s,kind,fx_room_pctGDP,X_slack_pctGDP,X_ext_pctGDP,ext_linear_extrapolated,X_carbon_pctGDP,'
                'X_eff_pctGDP,binding,lambda_imp\n')
        for r in rows:
            vals = [r['X_slack'], r['X_ext'], r['X_carbon']]
            vv = [v if np.isfinite(v) else np.inf for v in vals]
            vv = [np.inf if np.isnan(v) else v for v in vv]
            eff = min(vv)
            bind = ['闲置资源', '外部', '碳'][int(np.argmin(vv))]
            c = 'inf' if r['X_carbon'] == np.inf else ('n.a.' if np.isnan(r['X_carbon']) else f"{100 * r['X_carbon']:.2f}")
            f.write(f"{r['s']},{r['kind']},{r['fx_room_pctGDP']:.2f},{100 * r['X_slack']:.2f},{100 * r['X_ext']:.2f},"
                    f"{int(r['ext_extrapolated'])},{c},{100 * eff:.2f},{bind},{r['lam_imp']:.3f}\n")

    # 图：基准（储备余量 11.1% GDP）与偏紧情景（2.8% GDP）
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for j, kind in enumerate(['consumption', 'green']):
        sel = lambda share: [r for r in rows if r['kind'] == kind and abs(r['fx_room_pctGDP'] - 100 * FX_ROOM * share / 4) < 1e-6]
        base, tight = sel(1.0), sel(0.25)
        ss = [100 * r['s'] for r in base]
        ax[j].plot(ss, [100 * r['X_slack'] for r in base], 'o-', color=COL[0], lw=1.8, label='闲置资源边界')
        cb = [100 * r['X_carbon'] if np.isfinite(r['X_carbon']) else np.nan for r in base]
        if not all(np.isnan(cb)):
            ax[j].plot(ss, cb, 's-', color=COL[2], lw=1.8, label='碳边界（余量 = 基准排放 1%）')
        ax[j].plot(ss, [100 * r['X_ext'] for r in base], '^-', color=COL[4], lw=1.8, label='外部边界（储备余量 11.1% GDP）')
        ax[j].plot(ss, [100 * r['X_ext'] for r in tight], '^--', color=COL[4], lw=1.4, label='外部边界（储备余量 2.8% GDP）')
        eff = []
        for r in base:
            vals = [r['X_slack'], r['X_ext'], r['X_carbon'] if np.isfinite(r['X_carbon']) else np.inf]
            eff.append(100 * min(vals))
        ax[j].plot(ss, eff, color='k', lw=5, alpha=0.18, label='有效边界（基准）= min')
        ax[j].set_title('消费性发行' if kind == 'consumption' else '绿色发行（排放增量为负，碳边界不绑定）')
        ax[j].set_xlabel('闲置资源率 s（%）')
        ax[j].set_ylim(0, 45)
    ax[0].set_ylabel('两年内可发行规模（% 年 GDP）')
    ax[0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle('图 P9-1  有效发行边界 = 各实际约束边界的最小值（开放 HANK，α = 0.17，通胀容忍度 2 个百分点）', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_p9_boundaries.png'), dpi=160)


if __name__ == '__main__':
    main()
