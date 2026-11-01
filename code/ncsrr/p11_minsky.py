"""P11 Minsky 式地方信用周期：HANK 数值实验。

对应 v0.1 第 6.2 节 P11 的证明方案："03 模型 + 3.4 节的地价方程，比较 phi_L=0 与 phi_L>0 下的实验 4"。
输出：output/p11_*.csv, output/fig_p11_*.png
"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ModelU, ann, HERE
from plotstyle import setup, COL

OUT = os.path.join(HERE, 'output')
setup()

T = 300
RHO_EPS = 0.97
BASE_OMEGA = 0.6
ELB_ROOM = 0.014 / 4          # 政策利率 1.4%/年 的下调空间（⚠ 情景）


def shock(scale):
    # 高度持久的去杠杆冲击（半衰期约 6 年）：受约束单位的可抵押率下降，对应 2021 年以后的城投融资收紧。
    # 不用严格永久冲击：在泰勒规则与 phi_B 规则下，永久性的国家股权供给减少没有零通胀稳态（见 v0.2 正文 3.5）。
    return {'eps_L': -scale * RHO_EPS ** np.arange(T)}


def run(scale, **kw):
    m = ModelU(**kw)
    try:
        o = m.solve(shock(scale))
        if not np.all(np.isfinite(o['dY'])) or np.max(np.abs(o['dY'][-60:])) > 1e-3:   # 有限 T 下“解出”的发散路径
            o = None
    except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
        o = None                                   # 无有界解：地方信用系统不稳定
    return m, o


def calibrate_scale(target=2.4, **kw):
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, o = run(mid, **kw)
        if o is not None and -100 * o['dY'].min() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def summarize(m, o, H=20):
    L = m.p['L']
    lL = o['dL'] / L
    P = np.cumsum(o['pi'])                       # 价格水平（对数偏离）
    return dict(
        Y_trough=100 * o['dY'].min(),
        Y_trough_q=int(o['dY'].argmin()),
        L_trough=100 * lL.min(),
        L_trough_q=int(lL.argmin()),
        dL_flow_min=100 * np.diff(np.r_[0, o['dL']]).min(),   # 单季最大信用净减少（% 季度 GDP）
        pi_trough_ann=ann(o['pi']).min(),
        P_8q=100 * P[7],
        P_20q=100 * P[19],
        cumY_20q=100 * o['dY'][:H].sum() / 4,                # 5 年累计产出损失（% 年 GDP）
        q_trough=100 * o['q'].min(),
        lnK_20q=100 * o['lnK'][19],
        sfc=o['sfc_resid'],
    )


def impact_multiplier():
    """地方支出一次性减少的当期产出乘数（解析式中的 𝓂）。"""
    m = ModelU()
    TR = np.zeros(T); TR[0] = -0.01
    o = m.solve({'TR': TR})
    return o['dY'][0] / -0.01


def analytic(phi_L, phi_Y, omega, rho_c, mm, Lbar, gamma0=0.0):
    Phi = phi_L + phi_Y * mm * (1 - gamma0) * Lbar
    a = (1 - rho_c) * Phi * omega
    lam = (rho_c - a) / (1 - a) if a != 1 else np.inf
    return dict(Phi=Phi, a=a, amp=1 / (1 - a) if a < 1 else np.inf, lam=lam,
                stable=bool(a < (1 + rho_c) / 2),
                omega_bar=(1 + rho_c) / (2 * (1 - rho_c) * Phi) if Phi > 0 else np.inf)


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], (float, np.floating)) else str(r[k]) for k in keys) + '\n')


def main():
    os.makedirs(OUT, exist_ok=True)
    mm = impact_multiplier()
    Lbar = ModelU().p['L']
    print(f'impact multiplier of local spending 𝓂 = {mm:.3f}')

    # 1. 冲击规模校准：phi_L=0、omega=0.6 时产出最低 -2.4%（与 v0.1 引用的 05 实验 h2 基准同一量级）
    scale = calibrate_scale(2.4, omega=BASE_OMEGA)
    print('shock scale', scale)

    # 2. phi_L x phi_Y x ELB
    rowsA, irfs = [], {}
    for elb in [None, ELB_ROOM]:
        for phi_Y in [0.0, 1.0]:
            for phi_L in [0.0, 2.0, 4.0, 6.0]:
                m, o = run(scale, omega=BASE_OMEGA, phi_L=phi_L, phi_Y=phi_Y, elb_room=elb)
                an = analytic(phi_L, phi_Y, BASE_OMEGA, m.p['rho_c'], mm, Lbar)
                r = dict(elb='yes' if elb else 'no', phi_Y=phi_Y, phi_L=phi_L,
                         a=an['a'], amp_analytic=an['amp'], lam=an['lam'], **summarize(m, o))
                rowsA.append(r)
                irfs[(elb is not None, phi_Y, phi_L)] = o
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for phi_L in [0.0, 4.0]:                      # 稳健性：完全费雪债务通缩渠道
        m, o = run(scale, omega=BASE_OMEGA, phi_L=phi_L, phi_Y=1.0, lam_r=1.0)
        an = analytic(phi_L, 1.0, BASE_OMEGA, m.p['rho_c'], mm, Lbar)
        rowsA.append(dict(elb='no_fisher', phi_Y=1.0, phi_L=phi_L, a=an['a'], amp_analytic=an['amp'],
                          lam=an['lam'], **summarize(m, o)))
        print(rowsA[-1])
    write_csv(rowsA, os.path.join(OUT, 'p11_phiL_grid.csv'))

    # 3. 庞氏（受约束）单位占比 omega
    rowsB = []
    for phi_L in [0.0, 4.0]:
        for omega in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
            m, o = run(scale, omega=omega, phi_L=phi_L, phi_Y=1.0)
            an = analytic(phi_L, 1.0, omega, m.p['rho_c'], mm, Lbar)
            summ = summarize(m, o) if o is not None else {k: np.nan for k in summarize(*run(scale, omega=0.2))}
            r = dict(phi_L=phi_L, omega=omega, a=an['a'], amp_analytic=an['amp'],
                     omega_bar=an['omega_bar'], stable_analytic=int(an['stable']),
                     solved=int(o is not None), **summ)
            rowsB.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    write_csv(rowsB, os.path.join(OUT, 'p11_omega_grid.csv'))

    # 3b. 数值稳定阈值：线性区（极小冲击）下二分 omega，判定有界解是否存在
    thr = {}
    for phi_L in [2.0, 4.0, 6.0]:
        lo, hi = 0.0, 1.5
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            _, o = run(1e-5, omega=mid, phi_L=phi_L, phi_Y=1.0)
            if o is not None:
                lo = mid
            else:
                hi = mid
        an = analytic(phi_L, 1.0, 1.0, 0.9, mm, Lbar)
        thr[phi_L] = (0.5 * (lo + hi), an['omega_bar'])
        print('threshold', phi_L, thr[phi_L])
    with open(os.path.join(OUT, 'p11_threshold.csv'), 'w') as f:
        f.write('phi_L,phi_Y,omega_bar_HANK,omega_bar_analytic\n')
        for k, (a1, a2) in thr.items():
            f.write(f'{k},1.0,{a1:.4f},{a2:.4f}\n')

    # 4. 中央以国家股权替代的及时性：冲击后第 d 季开始，TR_t = -gamma ΔL_t
    rowsC, irfC = [], {}
    for elb in [None, ELB_ROOM]:
        for d in ['none', 0, 1, 2, 4, 8]:
            kw = dict(omega=BASE_OMEGA, phi_L=4.0, phi_Y=1.0, elb_room=elb)
            if d != 'none':
                kw.update(gamma=1.0, d=d)
            m, o = run(scale, **kw)
            r = dict(elb='yes' if elb else 'no', delay=d, **summarize(m, o))
            rowsC.append(r)
            irfC[(elb is not None, d)] = o
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    write_csv(rowsC, os.path.join(OUT, 'p11_substitution_delay.csv'))

    # 5. 产能内生（zeta = 0.05）下的长期成本
    rowsD = []
    for phi_L in [0.0, 4.0]:
        for d in ['none', 0, 4]:
            kw = dict(omega=BASE_OMEGA, phi_L=phi_L, phi_Y=1.0, zeta=0.05, elb_room=ELB_ROOM)
            if d != 'none':
                kw.update(gamma=1.0, d=d)
            m, o = run(scale, **kw)
            r = dict(phi_L=phi_L, delay=d, **summarize(m, o))
            rowsD.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    write_csv(rowsD, os.path.join(OUT, 'p11_hysteresis.csv'))

    # ---------------- 图
    H = 24
    tt = np.arange(H)
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
    spec = [('dY', '产出（% 季度 GDP）', 100), ('dL', '地方信用（% 稳态存量）', 100 / Lbar),
            ('q', '地价（% 偏离）', 100), ('pi', '通胀（年化百分点）', 400)]
    for k, (var, title, sc) in enumerate(spec):
        a_ = ax.flat[k]
        for phi_L, c in [(0.0, COL[0]), (4.0, COL[1])]:
            a_.plot(tt, sc * irfs[(True, 1.0, phi_L)][var][:H], color=c, lw=2,
                    label=f'φL = {phi_L:g}')
        a_.axhline(0, color='0.6', lw=0.8)
        a_.set_title(title)
        a_.set_xlabel('季度')
    ax.flat[0].legend(frameon=False)
    fig.suptitle('图 P11-1  同一地方信用冲击下，地价反馈 φL 使收缩更深、更集中（ω = 0.6，φY = 1，含利率下限）', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_p11_irf.png'), dpi=160)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for phi_L, c in [(0.0, COL[0]), (4.0, COL[1])]:
        rr = [r for r in rowsB if r['phi_L'] == phi_L]
        ax[0].plot([r['omega'] for r in rr], [r['Y_trough'] for r in rr], 'o-', color=c, lw=2, label=f'φL = {phi_L:g}')
        ob = rr[0]['omega_bar']
        if ob < 1.05:
            ax[0].axvline(ob, color=c, ls='--', lw=1)
            ax[0].annotate(f'线性区稳定阈值 ω̄ = {ob:.2f}', xy=(ob - 0.01, -1), color=c, fontsize=7,
                           rotation=90, va='top', ha='right')
    ax[0].set_xlabel('受约束（投机 + 庞氏）单位占比 ω')
    ax[0].set_title('产出最低点（%）')
    ax[0].legend(frameon=False)
    for d, c in [('none', COL[3]), (0, COL[0]), (2, COL[2]), (4, COL[4]), (8, COL[1])]:
        lab = '不替代' if d == 'none' else f'滞后 {d} 季'
        ax[1].plot(tt, 100 * irfC[(True, d)]['dY'][:H], color=c, lw=2, label=lab)
    ax[1].axhline(0, color='0.6', lw=0.8)
    ax[1].set_title('中央替代的及时性：产出（%）')
    ax[1].set_xlabel('季度')
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_p11_omega_delay.png'), dpi=160)

    with open(os.path.join(OUT, 'p11_meta.txt'), 'w') as f:
        f.write(f'impact_multiplier={mm:.6f}\nshock_scale={scale:.6f}\nLbar={Lbar}\n'
                f'household_beta={ModelU().hh["beta"]:.6f}\nMPC_optimizers={ModelU().hh["MPC"]:.6f}\n')


if __name__ == '__main__':
    main()
