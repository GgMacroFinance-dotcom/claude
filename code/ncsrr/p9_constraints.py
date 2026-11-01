"""P9 多重实际约束：开放 HANK + 碳记账 + 绿色资本，三个边界与有效边界。

v0.1 的证明方案："在 04 模型上加 3.7 节的碳模块（排放只作记账，不进效用），计算 R4 的三个边界"。
这里的开放模块是 v0.1 3.6 节的"资本管制下的钉住、国际收支由储备吸收"版本（04 的利率平价版本没有代码，未复现）。

发行：X（占年 GDP）在 H=8 个季度内均匀花出。两种用途：
  消费性：中央政府购买 Gc（不形成资本）
  绿色  ：绿色基金投资 Ig（形成绿色资本 K^g，提高产能 theta_g、降低排放强度 psi_sigma），
          全部以货币股权融资（x = 1），与消费性发行的融资方式相同，只有用途不同
三个约束：
  闲置资源：年化通胀峰值上升 <= PI_BAR
  外部    ：20 季内储备最低点 >= 下限（fx0 - fx_floor = 16.1% - 5% 年 GDP）
  碳      ：20 季累计排放增量 <= 碳预算余量（基准排放的 RHO_C）
"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ModelU, ann, HERE
from plotstyle import setup, COL

OUT = os.path.join(HERE, 'output')
setup()

T = 300
H = 8
H_EXT = 20
H_C = 20
PI_BAR = 2.0                          # 年化百分点（⚠ 情景，正文给出敏感性）
FX_ROOM = (0.161 - 0.05) * 4          # 季度 GDP 单位
RHO_C = 0.01                          # 碳预算余量 = 20 季基准排放的 1%（⚠ 情景）
GREEN = dict(theta_g=0.08, psi_sigma=0.2)
S_GRID = [0.02, 0.05, 0.10, 0.15, 0.20]
XMAX = 0.4


def path(X):
    v = np.zeros(T)
    v[:H] = 4 * X / H
    return v


def make_model(kind, **kw):
    kk = dict(GREEN) if kind == 'green' else {}
    kk.update(kw)
    return ModelU(**kk)


def usage_m(m, X, kind):
    """返回 (通胀峰值年化pp, 储备最低点占用（季度 GDP）, 累计排放增量（占 20 季基准排放）, 解)；不可解时 None。"""
    sh = {'Ig': path(X)} if kind == 'green' else {'Gc': path(X)}
    try:
        o = m.solve(sh)
    except Exception:
        return None
    return (ann(o['pi']).max(), -o['fx'][:H_EXT].min(), o['Em'][:H_C].sum() / H_C, o)


def usage(X, kind, **kw):
    return usage_m(make_model(kind, **kw), X, kind)


def scan(kind, step=0.005, **kw):
    """在 X 网格上扫描三个约束的占用，直到求解失败（u 逼近 1，闲置资源耗尽）或 XMAX。"""
    m = make_model(kind, **kw)
    Xs, U = [0.0], [(0.0, 0.0, 0.0)]
    X = step
    while X <= XMAX + 1e-12:
        u = usage_m(m, X, kind)
        if u is None:
            break
        Xs.append(X)
        U.append(u[:3])
        X += step
    return np.array(Xs), np.array(U)


def boundaries(kind, pi_bar=PI_BAR, fx_room=FX_ROOM, rho_c=RHO_C, grid=None, **kw):
    Xs, U = grid if grid is not None else scan(kind, **kw)
    lims = [pi_bar, fx_room, rho_c]
    out = []
    for k in range(3):
        u = U[:, k]
        idx = np.where(u > lims[k])[0]
        if len(idx) == 0:
            # 约束在可解区间内不绑定；若可解区间被闲置资源截断（求解失败），记为截断点
            out.append(np.inf if Xs[-1] >= XMAX - 1e-9 else Xs[-1] if k == 0 else np.inf)
            continue
        j = idx[0]
        out.append(Xs[j - 1] + (lims[k] - u[j - 1]) / (u[j] - u[j - 1]) * (Xs[j] - Xs[j - 1]))
    return out


def unit_usage(kind, X=0.01, **kw):
    """线性近似（R4）：每 1% 年 GDP 发行对三个约束的占用。"""
    u = usage(X, kind, **kw)
    return [u[0] / (100 * X), u[1] / (100 * X), u[2] / (100 * X)]


def fmt(x):
    return 'inf' if not np.isfinite(x) else f'{100 * x:.2f}'


NAMES = ['闲置资源', '外部', '碳']


GRIDS = {}


def _scan_job(args):
    kind, kw = args
    return scan(kind, **kw)


def main():
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    DEC = [('绿色：全部效应', {}),
           ('绿色：只有产能效应（psi_sigma=0）', dict(psi_sigma=0.0)),
           ('绿色：只有减排效应（theta_g=0）', dict(theta_g=0.0)),
           ('绿色：进口含量 alpha_g=0.10', dict(alpha_g=0.10)),
           ('绿色：进口含量 alpha_g=0.30', dict(alpha_g=0.30))]
    jobs = [(kind, dict(s=s, alpha=0.17)) for s in S_GRID for kind in ['consumption', 'green']]
    jobs += [('green', dict(s=0.10, alpha=0.17, **kw)) for _, kw in DEC[1:]]
    jobs += [('consumption', dict(s=s, alpha=0.0)) for s in S_GRID]
    with Pool(int(os.environ.get('NPROC', 3))) as pool:
        res = pool.map(_scan_job, jobs)
    ALL = {(k, tuple(sorted(kw.items()))): g for (k, kw), g in zip(jobs, res)}
    getg = lambda k, **kw: ALL[(k, tuple(sorted(kw.items())))]
    rows = []
    for s in S_GRID:
        for kind in ['consumption', 'green']:
            grid = getg(kind, s=s, alpha=0.17)
            GRIDS[(s, kind)] = grid
            b = boundaries(kind, grid=grid)
            uu = unit_usage(kind, s=s, alpha=0.17)
            eff = min(b)
            bind = NAMES[int(np.argmin(b))]
            r = dict(s=s, kind=kind, X_slack=b[0], X_ext=b[1], X_carbon=b[2], X_eff=eff, binding=bind,
                     u_pi=uu[0], u_fx=uu[1], u_c=uu[2])
            rows.append(r)
            print(s, kind, [fmt(x) for x in b], fmt(eff), bind, [f'{x:.4f}' for x in uu], flush=True)
    with open(os.path.join(OUT, 'p9_boundaries.csv'), 'w') as f:
        f.write('s,kind,X_slack_pctGDP,X_ext_pctGDP,X_carbon_pctGDP,X_eff_pctGDP,binding,'
                'pi_per_1pct,fx_per_1pct_qGDP,carbon_per_1pct\n')
        for r in rows:
            f.write(f"{r['s']},{r['kind']},{fmt(r['X_slack'])},{fmt(r['X_ext'])},{fmt(r['X_carbon'])},"
                    f"{fmt(r['X_eff'])},{r['binding']},{r['u_pi']:.5f},{r['u_fx']:.5f},{r['u_c']:.5f}\n")

    # 敏感性：通胀容忍度与碳预算余量（用同一组扫描结果）
    with open(os.path.join(OUT, 'p9_sensitivity.csv'), 'w') as f:
        f.write('s,kind,pi_bar,rho_c,X_slack_pctGDP,X_ext_pctGDP,X_carbon_pctGDP,X_eff_pctGDP\n')
        for (s_, kind), grid in GRIDS.items():
            for pb in [1.0, 2.0, 3.0]:
                for rc in [0.005, 0.01, 0.02]:
                    b = boundaries(kind, pi_bar=pb, rho_c=rc, grid=grid)
                    f.write(f'{s_},{kind},{pb},{rc},{fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(min(b))}\n')
    np.savez(os.path.join(OUT, 'p9_grids.npz'), **{f'{k[1]}_{k[0]}_X': v[0] for k, v in GRIDS.items()},
             **{f'{k[1]}_{k[0]}_U': v[1] for k, v in GRIDS.items()})

    # 机制分解（s = 10%）：只有产能效应 / 只有减排效应 / 绿色投资的进口含量
    dec = []
    for label, kw in DEC:
        b = boundaries('green', grid=getg('green', s=0.10, alpha=0.17, **kw))
        dec.append((label, b))
        print(label, [fmt(x) for x in b], flush=True)
    b = boundaries('consumption', grid=getg('consumption', s=0.10, alpha=0.17))
    dec.insert(0, ('消费性', b))
    with open(os.path.join(OUT, 'p9_decomposition_s10.csv'), 'w') as f:
        f.write('case,X_slack_pctGDP,X_ext_pctGDP,X_carbon_pctGDP,X_eff_pctGDP\n')
        for label, b in dec:
            f.write(f'{label},{fmt(b[0])},{fmt(b[1])},{fmt(b[2])},{fmt(min(b))}\n')

    # 封闭经济对照（alpha = 0，即 03 的设定）：闲置资源边界与 s 的比例
    with open(os.path.join(OUT, 'p9_closed_slack.csv'), 'w') as f:
        f.write('s,X_slack_closed_pctGDP,ratio_to_s\n')
        for s in S_GRID:
            b = boundaries('consumption', grid=getg('consumption', s=s, alpha=0.0))
            f.write(f'{s},{fmt(b[0])},{b[0] / s:.3f}\n')
            print('closed', s, fmt(b[0]), b[0] / s, flush=True)

    # 图
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for j, kind in enumerate(['consumption', 'green']):
        rr = [r for r in rows if r['kind'] == kind]
        ss = [100 * r['s'] for r in rr]
        for key, lab, c in [('X_slack', '闲置资源边界', COL[0]), ('X_ext', '外部边界', COL[4]),
                            ('X_carbon', '碳边界', COL[2])]:
            y = [100 * r[key] if np.isfinite(r[key]) else np.nan for r in rr]
            ax[j].plot(ss, y, 'o-', color=c, lw=1.6, label=lab)
        ax[j].plot(ss, [100 * r['X_eff'] for r in rr], color='k', lw=3, alpha=0.35, label='有效边界 = min')
        ax[j].set_title('消费性发行' if kind == 'consumption' else '绿色发行（碳边界不绑定的点不画）')
        ax[j].set_xlabel('闲置资源率 s（%）')
        ax[j].set_ylim(0, 100 * XMAX)
    ax[0].set_ylabel('两年内可发行规模（% 年 GDP）')
    ax[0].legend(frameon=False)
    fig.suptitle('图 P9-1  有效发行边界是三个实际约束的最小值；绿色发行的有效边界高于消费性发行', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_p9_boundaries.png'), dpi=160)


if __name__ == '__main__':
    main()
