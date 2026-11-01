"""P10 发行结构：绿色投资中国家股权（货币注资）的最优占比 x* 与闲置资源率 s。

简化问题（v0.2 正文 4.2 节）：
    min_x  (h/2)(1-x)^2 + (1/2) sum_k [U_k(x)/R_k]^2 ,   U_k(x) = a_k + b_k x
    h：债务悬置的边际成本（论文二的三个摩擦的约化形式，以“约束占用”的平方单位计）
    U_k：实际约束 k 的占用（闲置资源：年化通胀峰值；外部：储备消耗；碳：累计排放增量，仅在为正时计入）
    a_k：全部以项目债融资（x=0）时的占用；b_k：把 1 单位项目债换成货币股权的额外占用
闭式解：x* = clip[(h - A)/(h + B), 0, 1]，A = sum a_k b_k / R_k^2，B = sum b_k^2 / R_k^2
本脚本用开放 HANK（与 P9 相同）在 s 网格上计算 (a_k, b_k)，再给出 x*(s; h)。
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
from p9_constraints import GREEN, FX_ROOM, PI_BAR, RHO_C, H_EXT, H_C, path

OUT = os.path.join(HERE, 'output')
setup()

I_SIZE = 0.01                 # 绿色投资规模：两年合计 1% 年 GDP
S_GRID = [0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20]
R = np.array([PI_BAR, FX_ROOM, RHO_C])


def usages(s, x, **kw):
    kk = dict(GREEN, s=s, alpha=0.17, x=x)
    kk.update(kw)
    m = ModelU(**kk)
    o = m.solve({'Ig': path(I_SIZE)})
    return np.array([ann(o['pi']).max(), -o['fx'][:H_EXT].min(), o['Em'][:H_C].sum() / H_C]), o


def coeffs(s, **kw):
    U0, o0 = usages(s, 0.0, **kw)
    U1, o1 = usages(s, 1.0, **kw)
    Uh, _ = usages(s, 0.5, **kw)
    a, b = U0, U1 - U0
    lin_err = np.max(np.abs(Uh - (a + 0.5 * b)) / np.maximum(np.abs(b), 1e-12))
    return a, b, lin_err, (o0, o1)


def xstar(a, b, h, R=R):
    """闭式解；碳约束占用在 [0,1] 上恒为负时不计入（单侧约束）。"""
    use = np.array([True, True, (a[2] > 0) or (a[2] + b[2] > 0)])
    A = np.sum((a * b / R ** 2)[use])
    B = np.sum((b ** 2 / R ** 2)[use])
    return float(np.clip((h - A) / (h + B), 0, 1)), A, B


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in S_GRID:
        a, b, err, (o0, o1) = coeffs(s)
        rows.append((s, a, b, err))
        print(s, 'a', np.round(a, 5), 'b', np.round(b, 5), 'lin_err', round(err, 4), flush=True)

    # 校准 h：基准 s = 5% 时 x* = 76%（论文二的数值，⚠ 只作为刻度，见正文）
    a5, b5 = [(a, b) for s, a, b, _ in rows if s == 0.05][0]
    lo, hi = 1e-8, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if xstar(a5, b5, mid)[0] < 0.76:
            lo = mid
        else:
            hi = mid
    h76 = mid
    print('h calibrated to x*=0.76 at s=5%:', h76)

    H_LIST = [h76 / 4, h76, 4 * h76]
    with open(os.path.join(OUT, 'p10_coefficients.csv'), 'w') as f:
        f.write('s,kappa_s,a_pi,a_fx,a_c,b_pi,b_fx,b_c,linearity_err,' +
                ','.join(f'xstar_h{j}' for j in range(len(H_LIST))) + '\n')
        for s, a, b, err in rows:
            ks = 0.085 * 0.02 * (1 - s) / s
            xs = [xstar(a, b, h)[0] for h in H_LIST]
            f.write(f'{s},{ks:.5f},' + ','.join(f'{v:.6g}' for v in list(a) + list(b)) +
                    f',{err:.4f},' + ','.join(f'{v:.4f}' for v in xs) + '\n')
    with open(os.path.join(OUT, 'p10_meta.txt'), 'w') as f:
        f.write(f'I_SIZE={I_SIZE}\nh76={h76:.6g}\nH_LIST={H_LIST}\n')

    # 18% 对应的闲置率（在 h76 下沿 s 插值）
    xs76 = np.array([xstar(a, b, h76)[0] for _, a, b, _ in rows])
    ss = np.array([s for s, *_ in rows])
    if xs76.min() < 0.18 < xs76.max():
        s18 = np.interp(0.18, xs76, ss)
        print('s at which x*=18%:', s18)
        with open(os.path.join(OUT, 'p10_meta.txt'), 'a') as f:
            f.write(f's_at_x18={s18:.5f}\n')

    # 比较静态：接近外部约束（储备余量缩小）
    with open(os.path.join(OUT, 'p10_fxroom.csv'), 'w') as f:
        f.write('s,fx_room_share,xstar\n')
        for s, a, b, _ in rows:
            for share in [1.0, 0.5, 0.25, 0.1]:
                Rk = np.array([PI_BAR, FX_ROOM * share, RHO_C])
                f.write(f'{s},{share},{xstar(a, b, h76, Rk)[0]:.4f}\n')

    # 图
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for h, c, lab in zip(H_LIST, [COL[0], COL[1], COL[2]], ['h/4', 'h（校准）', '4h']):
        ax[0].plot(100 * ss, [100 * xstar(a, b, h)[0] for _, a, b, _ in rows], 'o-', color=c, lw=2, label=lab)
    ax[0].axhline(76, color='0.6', ls=':', lw=1)
    ax[0].axhline(18, color='0.6', ls=':', lw=1)
    ax[0].set_xlabel('闲置资源率 s（%）')
    ax[0].set_ylabel('最优货币股权占比 x*（%）')
    ax[0].set_title('x* 随 s 上升、随债务悬置 h 上升')
    ax[0].legend(frameon=False, title='债务悬置成本')
    for share, c in zip([1.0, 0.5, 0.25, 0.1], [COL[0], COL[4], COL[1], COL[3]]):
        Rk = np.array([PI_BAR, FX_ROOM * share, RHO_C])
        ax[1].plot(100 * ss, [100 * xstar(a, b, h76, Rk)[0] for _, a, b, _ in rows], 'o-', color=c, lw=2,
                   label=f'{100 * share:.0f}%')
    ax[1].set_xlabel('闲置资源率 s（%）')
    ax[1].set_title('接近外部约束时 x* 下降')
    ax[1].legend(frameon=False, title='储备余量（基准 = 100%）', loc='lower right', fontsize=7, title_fontsize=7)
    fig.suptitle('图 P10-1  绿色投资的最优国家股权（货币注资）占比', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_p10_xstar.png'), dpi=160)


if __name__ == '__main__':
    main()
