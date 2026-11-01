"""模型 U 精简版（v0.2）：序列空间完全预见求解。

模块与 v0.1 第 3 节对应（每个模块都有开关，取零时退化为更小的模型）：
  3.1 家庭       一资产 HANK（household.py）+ 手停口停占比 mu
  3.3 企业       闲置资源菲利普斯曲线（非线性 ln mc），内生产能 zeta、绿色资本 psi_g
  3.4 地方       土地抵押约束 + Minsky 地价反馈 phi_L, phi_Y；受约束单位占比 omega
  3.5 国家部门   合并预算约束，税收规则 phi_B，中央对地方替代规则 (gamma, d)
  3.6 国外       钉住汇率、储备吸收国际收支（v0.1 "或资本管制下的钉住"），进口份额 alpha
  3.7 绿色/碳    排放 E^m = sigma(K^g) Y，绿色投资 I^g 的货币股权 x / 项目债 1-x 融资

单位：季度，稳态季度 GDP = 1；存量均以季度 GDP 计（年 GDP 的比例 ×4）。
所有外生输入和输出都是对稳态的偏离（水平值；Y 的对数偏离 ≈ 水平偏离）。
"""
import os
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")   # 300×300 的小矩阵上多线程 BLAS 反而慢几百倍
import numpy as np
from household import Household

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'output', 'cache')


# ---------------------------------------------------------------- 参数
BASE = dict(
    T=300,
    rbar=0.015 / 4,          # 实际利率 1.5%/年
    mu=0.25,                 # 手停口停占比
    D=1.52 * 4,              # 家庭存款 = E3 = 152% 年 GDP
    L=0.97 * 4,              # 地方信用 97% 年 GDP
    I=0.40, Gc=0.08, Gl=0.12,
    # 供给侧（03 校准）
    kappa_p=0.085, nu=0.02, s=0.05,
    zeta=0.0, rho_k=0.985,
    theta_g=0.0, Kpub=2.5 * 4,   # 绿色/公共资本的产出弹性、公共资本存量
    delta_g=0.02,                # 绿色资本季度折旧
    # 货币与财政
    phi_pi=1.5, phi_B=0.02,
    # 地方（03：rho_c = 0.9）
    rho_c=0.9, omega=1.0, phi_L=0.0, phi_Y=0.0,
    gamma=0.0, d=0,              # 中央替代：t >= d 时 TR_t = -gamma * ΔL_t（拖延 d 季后开始同期替代）
    # 开放
    alpha=0.0, alpha_g=None,     # alpha_g: 绿色投资的进口含量（None 表示等于 alpha）
    # 碳
    psi_sigma=0.0,
    # 绿色融资
    x=1.0, Nd=40, e_g=1.0,       # 货币股权占比、项目债摊还期（季度）、资本形成效率
    lam_r=0.0,                   # 地方债事后实际利率的当季传导（1 = 完全费雪债务通缩渠道）
    elb_room=None,               # 利率下限空间（季度，对稳态名义利率的偏离）；None 表示无下限
)


def household_ss(p):
    """家庭稳态与雅可比（按 (mu, D, rbar, z) 缓存）。"""
    z = 1 - p['I'] - (p['Gc'] + p['rbar'] * (p['D'] - p['L'])) - (p['Gl'] + p['rbar'] * p['L'])
    key = f"hh_mu{p['mu']}_D{p['D']}_r{p['rbar']:.6f}_z{z:.6f}_T{p['T']}"
    path = os.path.join(CACHE, key + '.npz')
    if os.path.exists(path):
        d = np.load(path)
        return dict(z=z, beta=float(d['beta']), MPC=float(d['MPC']), A=float(d['A']),
                    Jcz=d['Jcz'], Jcr=d['Jcr'], Jaz=d['Jaz'], Jar=d['Jar'])
    os.makedirs(CACHE, exist_ok=True)
    hh = Household()
    ss = hh.calibrate_beta(p['rbar'], z, p['D'] / (1 - p['mu']))
    J = hh.jacobians(ss, p['T'])
    np.savez(path, beta=ss['beta'], MPC=ss['MPC'], A=ss['A'],
             Jcz=J['C']['z'], Jcr=J['C']['r'], Jaz=J['A']['z'], Jar=J['A']['r'])
    return household_ss(p)


def lag(X, k=1):
    """X 是 (T, n)，返回滞后 k 期（之前补零）。"""
    if k == 0:
        return X
    out = np.zeros_like(X)
    out[k:] = X[:-k]
    return out


class ModelU:
    def __init__(self, **kw):
        p = dict(BASE)
        p.update(kw)
        if p['alpha_g'] is None:
            p['alpha_g'] = p['alpha']
        self.p = p
        self.hh = household_ss(p)
        self.M = p['D'] - p['L']
        self.ubar = 1 - p['s']
        self.T = p['T']

    def kappa(self):
        p = self.p
        return p['kappa_p'] * p['nu'] * (1 - p['s']) / p['s']

    # ------------------------------------------------------------ 给定产出路径，计算全部变量
    def blocks(self, dY, shocks):
        """dY: (T, n)。shocks: dict of (T,) 外生路径。返回 dict of (T, n)。"""
        p, T = self.p, self.T
        n = dY.shape[1]
        g = lambda k: np.repeat(np.asarray(shocks.get(k, np.zeros(T)), float)[:, None], n, 1)
        Gc, TRx, Trh, Ig = g('Gc'), g('TR'), g('Trh'), g('Ig')
        eps_q, eps_L = g('eps_q'), g('eps_L')
        rb = p['rbar']

        Y = 1 + dY
        yhat = np.log(Y)

        # 绿色资本（含债务悬置造成的形成效率损失 e_g）
        Kg = np.zeros((T, n))
        for t in range(T):
            Kg[t] = (1 - p['delta_g']) * (Kg[t - 1] if t else 0) + p['e_g'] * Ig[t]
        # 产能：滞后效应 zeta + 绿色资本直接进入产能（产出弹性 theta_g）
        # ln K_t = hys_t + theta_g * Kg_{t-1}/Kpub,  hys_t = rho_k hys_{t-1} + zeta (u_{t-1} - ubar)
        # （绿色资本作为生产要素直接进入产能，不经 rho_k 累积；theta_g 是公共资本的产出弹性）
        hys = np.zeros((T, n))
        lnK = np.zeros((T, n))
        u = np.zeros((T, n))
        for t in range(T):
            if t:
                hys[t] = p['rho_k'] * hys[t - 1] + p['zeta'] * (u[t - 1] - self.ubar)
                lnK[t] = hys[t] + p['theta_g'] * Kg[t - 1] / p['Kpub']
            u[t] = self.ubar * Y[t] / np.exp(lnK[t])
        # 闲置资源菲利普斯曲线（非线性）
        u_c = np.minimum(u, 1 - 1e-9)
        lnmc = -p['nu'] * (np.log(1 - u_c) - np.log(1 - self.ubar)) + 1e3 * np.maximum(u - (1 - 1e-9), 0)
        pi = np.zeros((T, n))
        b = 1 / (1 + rb)
        for t in reversed(range(T)):
            pi[t] = p['kappa_p'] * lnmc[t] + (b * pi[t + 1] if t < T - 1 else 0)
        i = p['phi_pi'] * pi
        if p.get('elb_room') is not None:          # 利率下限：i_t >= -elb_room（对稳态的偏离），平滑 max
            k = 4000.0
            x_ = (i + p['elb_room']) * k
            i = -p['elb_room'] + np.where(x_ > 30, x_, np.log1p(np.exp(np.minimum(x_, 30)))) / k
        r = lag(i) - pi                                   # 1 + r_t = (1 + i_{t-1})/(1 + pi_t)

        # 地方信用：受约束单位（占比 omega）的抵押约束 + Minsky 地价反馈
        a = (1 - p['rho_c']) * p['phi_L'] * p['omega']
        lc = np.zeros((T, n))
        for t in range(T):
            prev = lc[t - 1] if t else 0
            lc[t] = ((p['rho_c'] - a) * prev + (1 - p['rho_c']) * (p['phi_Y'] * yhat[t] + eps_q[t] + eps_L[t])) / (1 - a)
        lL = p['omega'] * lc
        q = p['phi_L'] * (lL - lag(lL)) + p['phi_Y'] * yhat + eps_q
        dL = p['L'] * lL
        dDL = dL - lag(dL)
        start = (np.arange(T) >= p['d']).astype(float)[:, None]    # 冲击后第 d 季才开始替代
        TR = TRx - p['gamma'] * start * dDL
        # 地方债的事后实际利率：比例 lam_r 当季进入地方支出（费雪债务通缩渠道），其余由银行吸收，
        # 银行利润 Pib 归家庭（v0.1 3.1 中 z 含 Π^b），保证存量—流量一致
        dGl = TR + dDL - rb * lag(dL) - p['lam_r'] * p['L'] * r
        Pib = -(1 - p['lam_r']) * p['L'] * r

        # 绿色项目债（银行贷款，由使用者付费摊还，付费由家庭承担）
        Dg = np.zeros((T, n))
        chi = np.zeros((T, n))
        for t in range(T):
            prev = Dg[t - 1] if t else 0
            chi[t] = (rb + 1 / p['Nd']) * prev
            Dg[t] = (1 + rb) * prev + (1 - p['x']) * Ig[t] - chi[t]

        # 进口、储备（钉住，r* = rbar）
        # 家庭消费需要先算；进口依赖 C，因此在下面统一求
        # 家庭：z = Y - I - Tc - Tl + Trh - chi
        # 中央：M_t = (1+rb) M_{t-1} + M r_t + Gc + TR + Trh + x Ig - Tc_t + [fx_t - (1+rb) fx_{t-1}]
        #       Tc_t = phi_B M_{t-1}
        # 进口 = alpha (C + Gc + Gl) + alpha_g Ig；fx_t = (1+rb) fx_{t-1} - Δimports
        # M 与 C 相互依赖：Tc 依赖 M_{t-1}，C 依赖整条 z 路径。按以下方式处理：
        #   M 的递推只依赖 C 经由进口项；进口项进入 M 的方式与 fx 相同（-Δimp_t）。
        #   因此先求“不含进口”的 M0 与 Tc0，再解出含进口的线性修正（见 solve 中的残差）。
        self._cache = dict(Y=Y, yhat=yhat, u=u, lnK=lnK, Kg=Kg, pi=pi, i=i, r=r, lc=lc, lL=lL, q=q,
                           dL=dL, TR=TR, dGl=dGl, Pib=Pib, Dg=Dg, chi=chi, Gc=Gc, Trh=Trh, Ig=Ig)
        return self._cache

    def household_and_fiscal(self, v, dC_guess=None):
        """给定 blocks 输出，求税收、家庭消费、进口、储备、M（线性，用不动点精确求解）。"""
        p, T = self.p, self.T
        rb, phiB = p['rbar'], p['phi_B']
        hh = self.hh
        mu = p['mu']
        dY, r = v['Y'] - 1, v['r']
        exo_dM = v['Gc'] + v['TR'] + v['Trh'] + p['x'] * v['Ig'] + self.M * r
        # 线性算子：M = Phi (exo_dM - imp_flow)，其中 Phi 是 (1+rb-phiB) 的递推
        if not hasattr(self, '_Phi'):
            rho_M = 1 + rb - phiB
            self._Phi = rho_M ** np.subtract.outer(np.arange(T), np.arange(T)).clip(0) * np.tri(T)
            self._Kz = phiB * np.eye(T, k=-1) @ self._Phi
        Phi = self._Phi
        LagM = np.eye(T, k=-1)
        # Tc = phiB * Lag M ；z = dY - Tc + Trh - chi
        # C = mu z + (1-mu)(Jcz z + Jcr r)
        # imp = alpha (C + Gc + Gl) + alpha_g Ig ；Δfx 进入 M： M = Phi(exo + fx - (1+rb) Lag fx)
        #   fx = Psi(-imp)，Psi 为 (1+rb) 递推；于是 fx - (1+rb) Lag fx = -imp
        # => M = Phi(exo - imp)
        Cz = mu * np.eye(T) + (1 - mu) * hh['Jcz']
        base_z = dY + v['Trh'] - v['chi'] + v['Pib']
        # z = base_z - phiB Lag Phi (exo - imp)
        # imp = alpha (Cz z + (1-mu) Jcr r + Gc + Gl) + alpha_g Ig
        al = p['alpha']
        Kz = self._Kz
        # z = base_z - Kz exo + Kz imp ;  imp = al Cz z + rest
        rest = al * ((1 - mu) * hh['Jcr'] @ r + v['Gc'] + v['dGl']) + p['alpha_g'] * v['Ig']
        if not hasattr(self, '_Ainv'):
            self._Ainv = np.linalg.inv(np.eye(T) - Kz @ (al * Cz))
        z = self._Ainv @ (base_z - Kz @ exo_dM + Kz @ rest)
        C = Cz @ z + (1 - mu) * hh['Jcr'] @ r
        imp = al * (C + v['Gc'] + v['dGl']) + p['alpha_g'] * v['Ig']
        dM = Phi @ (exo_dM - imp)
        Tc = phiB * LagM @ dM
        fx = np.zeros_like(imp)
        for t in range(T):
            fx[t] = (1 + rb) * (fx[t - 1] if t else 0) - imp[t]
        Ah = (1 - mu) * (hh['Jaz'] @ z + hh['Jar'] @ r)
        v.update(z=z, C=C, imp=imp, dM=dM, Tc=Tc, fx=fx, Ah=Ah)
        # 排放：E^m = sigma(Kg) Y，sigma 归一化为 1
        v['Em'] = v['Y'] * np.exp(-p['psi_sigma'] * v['Kg']) - 1
        return v

    def residual(self, dY, shocks):
        v = self.household_and_fiscal(self.blocks(dY, shocks))
        p = self.p
        demand = (1 - p['alpha']) * (v['C'] + v['Gc'] + v['dGl']) + (1 - p['alpha_g']) * v['Ig'] \
            + p['alpha'] * 0  # 出口外生
        # 注意：稳态时 Y = C + I + G（净出口为零），偏离量 dY = 国内吸收偏离 - 进口偏离
        return dY - demand, v

    def solve(self, shocks, tol=1e-10, maxit=40, verbose=False):
        T = self.T
        dY = np.zeros((T, 1))
        if not hasattr(self, '_J0'):
            self._J0 = self.jacobian(dY, {})             # 稳态处的雅可比，拟牛顿复用
        J, fresh = self._J0, False
        for it in range(maxit):
            R, v = self.residual(dY, shocks)
            err = np.max(np.abs(R))
            if verbose:
                print(it, err)
            if not np.isfinite(err):
                raise RuntimeError('non-finite residual')
            if err < tol:
                break
            step = np.linalg.solve(J, R)
            lam, ok = 1.0, False
            for _ in range(12):                     # 回溯线搜索（u 接近 1 时非线性很强）
                trial = dY - lam * step
                Rt, _ = self.residual(trial, shocks)
                if np.all(np.isfinite(Rt)) and np.max(np.abs(Rt)) < err:
                    ok = True
                    break
                lam /= 2
            if ok:
                dY = trial
                if lam < 1 and not fresh:            # 非线性强：在当前点重算雅可比
                    J, fresh = self.jacobian(dY, shocks), True
                else:
                    fresh = False
            elif not fresh:
                J, fresh = self.jacobian(dY, shocks), True
            else:
                raise RuntimeError(f'line search failed, err={err}')
        else:
            raise RuntimeError(f'not converged, err={err}')
        out = {k: (val[:, 0] if isinstance(val, np.ndarray) and val.ndim == 2 else val) for k, val in v.items()}
        out['dY'] = dY[:, 0]
        out['iters'] = it
        out['sfc_resid'] = np.max(np.abs(out['Ah'] - (out['dM'] + out['dL'] + out['Dg'])))
        return out

    def jacobian(self, dY, shocks, h=1e-6):
        """残差对 dY 的雅可比。各块对列向量化，household_and_fiscal 是线性的，
        因此用：J = I - d(demand)/dY，demand 关于 blocks 输出线性。"""
        T = self.T
        base = np.repeat(dY, T, axis=1) + h * np.eye(T)
        R1 = self._residual_matrix(base, shocks)
        R0 = self._residual_matrix(dY, shocks)
        return (R1 - R0) / h

    def _residual_matrix(self, dYmat, shocks):
        # 对多列同时计算残差（household_and_fiscal 对列线性，矩阵运算可直接作用于多列）
        v = self.blocks(dYmat, shocks)
        v = self.household_and_fiscal(v)
        p = self.p
        demand = (1 - p['alpha']) * (v['C'] + v['Gc'] + v['dGl']) + (1 - p['alpha_g']) * v['Ig']
        return dYmat - demand


def ann(x):
    """季度通胀 -> 年化百分点。"""
    return 400 * np.asarray(x)
