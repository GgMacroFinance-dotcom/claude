"""一资产 HANK 家庭块：EGM 稳态 + fake-news 序列空间雅可比。

记号对应 v0.1 第 3.1 节：
    c_t + a_t = (1 + r_t) a_{t-1} + z_t e_t,   a_t >= 0
其中 r_t 是 t 期实现的存款实际收益（作用于 a_{t-1}），z_t 是人均税后非资产收入。
手停口停家庭（占比 mu）在 model.py 中另行处理，这里只是跨期优化的那 1 - mu 部分。

本文件不依赖 03—05 的代码，是 v0.2 为 P9—P11 重新写的精简实现。
"""
import os as _os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")   # 300×300 的小矩阵上多线程 BLAS 反而慢几百倍
import numpy as np


# ---------------------------------------------------------------- 网格与收入过程
def rouwenhorst(n, rho, sigma):
    """Rouwenhorst 离散化 AR(1) 对数收入，返回 (e 网格, 转移矩阵, 平稳分布)，E[e]=1。"""
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])
    for m in range(3, n + 1):
        Z = np.zeros((m, m))
        Z[:-1, :-1] += p * Pi
        Z[:-1, 1:] += (1 - p) * Pi
        Z[1:, :-1] += (1 - p) * Pi
        Z[1:, 1:] += p * Pi
        Z[1:-1] /= 2
        Pi = Z
    # 平稳分布
    pi = np.ones(n) / n
    for _ in range(10000):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new - pi)) < 1e-14:
            break
        pi = pi_new
    sd = sigma * np.sqrt(n - 1)
    x = np.linspace(-sd, sd, n)
    e = np.exp(x)
    e = e / (pi @ e)
    return e, Pi, pi


def agrid(amax, n, amin=0.0):
    """双指数加密网格，靠近借贷约束处更密。"""
    u = np.linspace(0, 1, n)
    k = 6.0
    return amin + (amax - amin) * (np.exp(k * u) - 1) / (np.exp(k) - 1)


# ---------------------------------------------------------------- 家庭问题
class Household:
    def __init__(self, sigma=2.0, nE=7, rho_e=0.966, sd_e=0.5, nA=300, amax=200.0):
        self.sigma = sigma
        self.e, self.Pi, self.pi_e = rouwenhorst(nE, rho_e, sd_e * np.sqrt(1 - rho_e ** 2))
        self.a = agrid(amax, nA)
        self.nE, self.nA = nE, nA

    # 一步 EGM 反向迭代：给定下一期 Va'，本期 (r, z) -> (Va, a', c)
    def backward(self, Va_next, beta, r, z):
        sig = self.sigma
        W = beta * (self.Pi @ Va_next)                  # (nE, nA')，对 a' 网格
        c_endo = W ** (-1 / sig)
        coh_endo = c_endo + self.a[None, :]             # c + a'
        a_prev_endo = (coh_endo - z * self.e[:, None]) / (1 + r)
        ap = np.empty((self.nE, self.nA))
        for i in range(self.nE):
            ap[i] = np.interp(self.a, a_prev_endo[i], self.a)
        ap = np.maximum(ap, self.a[0])
        coh = (1 + r) * self.a[None, :] + z * self.e[:, None]
        ap = np.minimum(ap, coh - 1e-10)
        c = coh - ap
        Va = (1 + r) * c ** (-sig)
        return Va, ap, c

    # 抽签法（Young 2010）的插值权重
    def lottery(self, ap):
        idx = np.searchsorted(self.a, ap, side='right') - 1
        idx = np.clip(idx, 0, self.nA - 2)
        w = (self.a[idx + 1] - ap) / (self.a[idx + 1] - self.a[idx])
        w = np.clip(w, 0.0, 1.0)
        return idx, w

    def forward(self, D, idx, w):
        """分布前推一期：先按资产政策，再按收入转移。"""
        Dn = np.zeros_like(D)
        for i in range(self.nE):
            np.add.at(Dn[i], idx[i], w[i] * D[i])
            np.add.at(Dn[i], idx[i] + 1, (1 - w[i]) * D[i])
        return self.Pi.T @ Dn

    def expect_forward(self, X, idx, w):
        """forward 的转置：E_t[X_{t+1}] 在 t 期状态上的取值。"""
        Xe = self.Pi @ X
        out = np.empty_like(X)
        for i in range(self.nE):
            out[i] = w[i] * Xe[i, idx[i]] + (1 - w[i]) * Xe[i, idx[i] + 1]
        return out

    def steady_state(self, beta, r, z, tol=1e-11, maxit=20000):
        coh = (1 + r) * self.a[None, :] + z * self.e[:, None]
        Va = (1 + r) * (0.1 * coh) ** (-self.sigma)
        for it in range(maxit):
            Va_new, ap, c = self.backward(Va, beta, r, z)
            if np.max(np.abs(Va_new - Va)) < tol:
                Va = Va_new
                break
            Va = Va_new
        Va, ap, c = self.backward(Va, beta, r, z)
        idx, w = self.lottery(ap)
        D = np.outer(self.pi_e, np.ones(self.nA)) / self.nA
        for it in range(100000):
            Dn = self.forward(D, idx, w)
            if np.max(np.abs(Dn - D)) < 1e-13:
                D = Dn
                break
            D = Dn
        ss = dict(beta=beta, r=r, z=z, Va=Va, ap=ap, c=c, D=D, idx=idx, w=w,
                  A=np.sum(D * ap), C=np.sum(D * c))
        # 边际消费倾向（季度，对一次性收入）
        dz = 1e-4
        coh = (1 + r) * self.a[None, :] + z * self.e[:, None]
        c_up = np.empty_like(c)
        for i in range(self.nE):
            c_up[i] = np.interp(coh[i] + dz, coh[i], c[i])
        ss['MPC'] = np.sum(D * (c_up - c)) / dz
        return ss

    def calibrate_beta(self, r, z, A_target, lo=0.95, hi=None):
        hi = hi or (1 / (1 + r) - 1e-6)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            A = self.steady_state(mid, r, z)['A']
            if A > A_target:
                hi = mid
            else:
                lo = mid
            if abs(A - A_target) < 1e-7:
                break
        return self.steady_state(mid, r, z)

    # ------------------------------------------------------------ fake-news 雅可比
    def jacobians(self, ss, T, h=1e-5):
        """返回 {'C': {'r': J, 'z': J}, 'A': {...}}，T×T 矩阵。"""
        beta, r, z = ss['beta'], ss['r'], ss['z']
        Va_ss, ap_ss, c_ss, D_ss = ss['Va'], ss['ap'], ss['c'], ss['D']
        idx, w = ss['idx'], ss['w']

        # 期望向量 E_t: (Λ')^t 作用于结果
        def expectations(X):
            E = np.empty((T - 1,) + X.shape)
            E[0] = X
            for t in range(1, T - 1):
                E[t] = self.expect_forward(E[t - 1], idx, w)
            return E
        Ec, Ea = expectations(c_ss), expectations(ap_ss)

        out = {'C': {}, 'A': {}}
        for inp in ['r', 'z']:
            # 第 0 期冲击：直接改变当期政策；之后冲击：经 Va 反向传递
            dc = np.empty((T, self.nE, self.nA))
            da = np.empty((T, self.nE, self.nA))
            dr = h if inp == 'r' else 0.0
            dz = h if inp == 'z' else 0.0
            Va1, ap1, c1 = self.backward(Va_ss, beta, r + dr, z + dz)
            dc[0], da[0] = (c1 - c_ss) / h, (ap1 - ap_ss) / h
            Va_cur = Va1
            for s in range(1, T):
                Va_cur, ap_s, c_s = self.backward(Va_cur, beta, r, z)
                dc[s], da[s] = (c_s - c_ss) / h, (ap_s - ap_ss) / h
            # 分布在第 1 期的变化
            dD1 = np.empty((T, self.nE, self.nA))
            for s in range(T):
                idx_s, w_s = self.lottery(ap_ss + h * da[s])
                dD1[s] = (self.forward(D_ss, idx_s, w_s) - self.forward(D_ss, idx, w)) / h
            for out_name, d_pol, E in [('C', dc, Ec), ('A', da, Ea)]:
                F = np.empty((T, T))
                F[0] = np.einsum('sij,ij->s', d_pol, D_ss)
                F[1:] = np.einsum('tij,sij->ts', E, dD1)
                J = F.copy()
                for t in range(1, T):
                    J[t, 1:] += J[t - 1, :-1]
                out[out_name][inp] = J
        return out
