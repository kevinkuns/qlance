# This is temporarily directly copied from Lee McCuller's wavestate AAA module
# before it is available on pypi. The original code is here
# https://github.com/wavestate/wavestate-AAA

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@mit.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np
import scipy.linalg
import itertools


def residuals(xfer, fit, w, rtype):
    if callable(rtype):
        return rtype(xfer, fit)
    R = fit / xfer
    if rtype == "zeros":
        return w * (R - 1)
    elif rtype == "poles":
        return w * (1 / R - 1)
    elif rtype == "dualA":
        return w * (0.5 * R + 0.5 / R - 1)
    elif rtype == "dualB":
        return w * (R - 1 / R) / 2
    elif rtype == "log":
        R_abs = abs(R)
        log_re = w * np.log(R_abs)
        log_im = w * R.imag / R_abs
        return log_re + 1j * log_im
    else:
        raise RuntimeError("Unrecognized residuals type")


def tf_bary_interp(F_Hz, zvals, fvals, wvals):
    sF_Hz = 1j * F_Hz
    w_idx = 0
    N = 0
    D = 0
    idx_f_repl = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for idx, z in enumerate(zvals):
            f = fvals[idx]

            if z == 0:
                w = wvals[w_idx]
                w_idx += 1
                assert abs(f.imag / f.real) < 1e-13
                bary_Dw = w / (sF_Hz - z)
                for idx in np.argwhere(~np.isfinite(bary_Dw))[:, 0]:
                    idx_f_repl.append((idx, f))
                N = N + f * bary_Dw
                D = D + bary_Dw
            else:
                w_r = wvals[w_idx]
                w_i = wvals[w_idx + 1]
                w_idx += 2
                bary_D = 1 / (sF_Hz - z)
                bary_Dc = 1 / (sF_Hz - z.conjugate())
                for idx in np.argwhere(~np.isfinite(bary_D))[:, 0]:
                    idx_f_repl.append((idx, f))
                for idx in np.argwhere(~np.isfinite(bary_Dc))[:, 0]:
                    idx_f_repl.append((idx, f))

                # this is the TF-symmetric version with real weights
                N = N + (
                    w_r * (f * bary_D + f.conjugate() * bary_Dc)
                    - 1j * w_i * (f * bary_D - f.conjugate() * bary_Dc)
                )
                D = D + (w_r * (bary_D + bary_Dc) - 1j * w_i * (bary_D - bary_Dc))
        xfer = N / D
    for idx, f in idx_f_repl:
        xfer[idx] = f
    return xfer


def tf_bary_zpk(
    zvals,
    fvals,
    wvals,
    minreal_cutoff=1e-2,
):
    # evaluate poles and zeros in arrowhead form
    # these are modified for the symmetry conditions to be a real matrix

    # if zero is present, it must be the first element
    assert not np.any(zvals[1:] == 0)

    # len(zvals) must be p_order
    # len(wvals) must be order

    p_order = len(wvals)
    B = np.eye(p_order + 1)
    B[0, 0] = 0

    Ep = np.zeros((p_order + 1, p_order + 1))
    Ep[1:, 0] = 1
    Ez = np.zeros((p_order + 1, p_order + 1))
    Ez[1:, 0] = 1
    if zvals[0] == 0:
        Ep[0, 1] = wvals[0]
        Ep[0, 2::2] = wvals[1::2] + wvals[2::2]
        Ep[0, 3::2] = wvals[1::2] - wvals[2::2]
        # gain_d = wvals[0] + 2*np.sum(wvals[1::2])

        Ez[0, 1] = (wvals[0] * fvals[0]).real
        c = (wvals[1::2] + wvals[2::2]) + (wvals[1::2] - wvals[2::2]) * 1j
        cx = c * fvals[1:]
        Ez[0, 2::2] = cx.real
        Ez[0, 3::2] = cx.imag
        # gain_n = (wvals[0] * fvals[0].real) + 2*np.sum(wvals[1::2]*fvals[1:].real + wvals[2::2]*fvals[1:].imag)
        offs = 1
    else:
        Ep[0, 1::2] = wvals[0::2] + wvals[1::2]
        Ep[0, 2::2] = wvals[0::2] - wvals[1::2]
        # gain_d = 2*np.sum(wvals[0::2])

        c = (wvals[0::2] + wvals[1::2]) + (wvals[0::2] - wvals[1::2]) * 1j
        cx = c * fvals[0:]
        Ez[0, 1::2] = cx.real
        Ez[0, 2::2] = cx.imag
        # gain_n = 2*np.sum(wvals[0::2]*fvals.real + wvals[1::2]*fvals.imag)
        offs = 0
    # TODO, use numpy tricks for diag/offdiag filling instead this for-loop
    for idx, f in enumerate(zvals[offs:]):
        Ep[offs + 1 + 2 * idx, offs + 1 + 2 * idx] = f.real
        Ep[offs + 2 + 2 * idx, offs + 2 + 2 * idx] = f.real
        Ep[offs + 1 + 2 * idx, offs + 2 + 2 * idx] = f.imag
        Ep[offs + 2 + 2 * idx, offs + 1 + 2 * idx] = -f.imag
        Ez[offs + 1 + 2 * idx, offs + 1 + 2 * idx] = f.real
        Ez[offs + 2 + 2 * idx, offs + 2 + 2 * idx] = f.real
        Ez[offs + 1 + 2 * idx, offs + 2 + 2 * idx] = f.imag
        Ez[offs + 2 + 2 * idx, offs + 1 + 2 * idx] = -f.imag
    poles = scipy.linalg.eig(Ep, B, left=False, right=False)
    poles = poles[np.isfinite(poles)]
    zeros = scipy.linalg.eig(Ez, B, left=False, right=False)
    zeros = zeros[np.isfinite(zeros)]

    zeros, poles = order_reduce_zp(zeros, poles, Q_rank_cutoff=minreal_cutoff)

    TFvals_rel = []
    for f, z in zip(fvals, zvals):
        Gz = z - zeros
        Gp = z - poles
        TF = np.prod([gz / gp for gz, gp in itertools.zip_longest(Gz, Gp, fillvalue=1)])
        TFvals_rel.append(f / TF)
    TFvals_rel = np.asarray(TFvals_rel)
    # print(TFvals_rel)
    gain = np.median(TFvals_rel.real)
    # this may also get computed using the gain_n/gain_d above, but that fails
    # when poles or zeros are dropped since one of gain_n or gain_d will be
    # numerically 0 in that case

    return zeros, poles, gain


def tfAAA(
    F_Hz,
    xfer,
    exact=True,
    res_tol=None,
    s_tol=None,
    w=1,
    w_res=None,
    degree_max=30,
    nconv=None,
    nrel=10,
    rtype="log",
    lf_eager=True,
    supports=(),
    minreal_cutoff=None,
):
    if exact:
        if res_tol is None:
            res_tol = 1e-12
        if s_tol is None:
            s_tol = 0
        if nconv is None:
            nconv = 1
        if minreal_cutoff is None:
            minreal_cutoff = (1e-3,)
    else:
        if res_tol is None:
            res_tol = 0
        if s_tol is None:
            s_tol = 0
        if nconv is None:
            nconv = 2
        if minreal_cutoff is None:
            minreal_cutoff = (1e-3,)

    F_Hz = np.asarray(F_Hz)
    xfer = np.asarray(xfer)
    w = np.asarray(w)
    if w_res is None:
        w_res = w
    w_res = np.asarray(w_res)

    F_Hz, xfer, w, w_res = domain_sort(F_Hz, xfer, w, w_res)

    sF_Hz = 1j * F_Hz

    fit_list = []
    # these are the matrices and data related to the fit
    fvals = []
    zvals = []
    Vn_list = []
    Vd_list = []
    # and the domain and data
    xfer_drop = xfer.copy()
    sF_Hz_drop = sF_Hz.copy()
    w_drop = w.copy()
    w_res_drop = w_res.copy()
    N_drop = np.asarray(1)
    D_drop = np.asarray(1)
    del xfer
    del F_Hz
    del w
    del sF_Hz

    def add_point(idx):
        z = sF_Hz_drop[idx].copy()
        f = xfer_drop[idx].copy()
        fvals.append(f)
        zvals.append(z)

        _drop_inplace(idx, sF_Hz_drop)
        _drop_inplace(idx, xfer_drop)
        if w_drop.shape != ():
            _drop_inplace(idx, w_drop)
        if w_res_drop.shape != ():
            _drop_inplace(idx, w_res_drop)
        if N_drop.shape != ():
            _drop_inplace(idx, N_drop)
        if D_drop.shape != ():
            _drop_inplace(idx, D_drop)
        for v in Vn_list:
            _drop_inplace(idx, v)
        for v in Vd_list:
            _drop_inplace(idx, v)

        if z == 0:
            assert abs(f.imag / f.real) < 1e-13
            bary_D = 1 / (sF_Hz_drop - z)
            with np.errstate(divide="ignore", invalid="ignore"):
                Vn_list.append(f * bary_D)
                Vd_list.append(bary_D)
        else:
            bary_D = 1 / (sF_Hz_drop - z)
            bary_Dc = 1 / (sF_Hz_drop - z.conjugate())

            # this is the TF-symmetric version with real weights
            Vn_list.append(f * bary_D + f.conjugate() * bary_Dc)
            Vd_list.append(bary_D + bary_Dc)
            Vn_list.append(-1j * (f * bary_D - f.conjugate() * bary_Dc))
            Vd_list.append(-1j * (bary_D - bary_Dc))
        # print(z, f, bary_D)
        return

    if exact:

        def res_max_heuristic(res):
            return abs(res)

    else:

        def res_max_heuristic(res):
            rSup = np.cumsum(res)
            res_max = 0 * abs(res)
            for b in [4, 8, 16, 32, 64]:
                ravg = (rSup[b:] - rSup[:-b]) / b ** 0.5
                res_max[b // 2 : -b // 2] = np.maximum(
                    abs(ravg), res_max[b // 2 : -b // 2]
                )
            return res_max

    # adds the lowest frequency point to ensure good DC fitting
    if supports:
        for f in supports:
            idx = np.searchsorted((sF_Hz_drop / 1j).real, f)
            add_point(idx)
        skip_add = True
    else:
        if lf_eager:
            add_point(np.argmin((sF_Hz_drop / 1j).real))
            skip_add = True
        else:
            skip_add = False

    if not skip_add:
        fit_drop = np.median(abs(xfer_drop))
        res = residuals(xfer=xfer_drop, fit=fit_drop, w=w_res_drop, rtype=rtype)
    else:
        res = None

    wvals = []
    while True:
        if len(wvals) > degree_max:
            break

        if res is not None:
            idx_max = np.argmax(res_max_heuristic(res))
            add_point(idx_max)

        Vn = np.asarray(Vn_list).T
        Vd = np.asarray(Vd_list).T

        for _i in range(nconv):
            Na = np.mean(abs(N_drop) ** 2) ** 0.5 / nrel
            Hd1 = Vd * xfer_drop.reshape(-1, 1)
            Hn1 = Vn
            Hs1 = (Hd1 - Hn1) * (w_drop / (abs(N_drop) + Na)).reshape(-1, 1)

            Da = np.mean(abs(D_drop) ** 2) ** 0.5 / nrel
            Hd2 = Vd
            Hn2 = Vn * (1 / xfer_drop).reshape(-1, 1)
            Hs2 = (Hd2 - Hn2) * (w_drop / (abs(D_drop) + Da)).reshape(-1, 1)

            Hblock = [
                [Hs1.real],
                [Hs1.imag],
                [Hs2.real],
                [Hs2.imag],
            ]

            SX1 = np.block(Hblock)
            u, s, v = np.linalg.svd(SX1)
            wvals = v[-1, :].conjugate()

            N_drop = Vn @ wvals
            D_drop = Vd @ wvals

            fit_drop = N_drop / D_drop

        srel = s[-1] / s[0]

        res = residuals(xfer=xfer_drop, fit=fit_drop, w=w_res_drop, rtype=rtype)
        res_asq = res.real ** 2 + res.imag ** 2
        res_rms = np.mean(res_asq) ** 0.5
        res_max = np.max(res_asq) ** 0.5

        fit_list.append(
            dict(
                order=len(wvals),
                p_order=len(fvals),
                wvals=wvals,
                srel=srel,
                s=s,
                res_asq=res_asq,
                res_rms=res_rms,
                res_max=res_max,
            )
        )

        if (res_max < res_tol) or (srel < s_tol):
            break

    res_max_asq = res_max_heuristic(res) ** 2

    def interp(F_Hz, p_order):
        return tf_bary_interp(
            F_Hz,
            zvals=zvals[:p_order],
            fvals=fvals[:p_order],
            # p_order doesn't directly correspond to wvals, but this is OK since
            # only the ones matched to zvals and fvals are used
            wvals=wvals,
        )

    results = rtAAAResults(
        zvals_full=zvals,
        fvals_full=fvals,
        fit_list=fit_list,
        debug=Structish(locals()),
        minreal_cutoff=minreal_cutoff,
    )
    return results


class rtAAAResults(object):
    def __init__(
        self,
        zvals_full,
        fvals_full,
        fit_list,
        minreal_cutoff=1e-2,
        debug=None,
    ):
        self.zvals_full = np.asarray(zvals_full)
        self.fvals_full = np.asarray(fvals_full)
        self.fit_list = fit_list
        self.fit_idx = len(fit_list) - 1
        self.fit_dict = self.fit_list[self.fit_idx]
        self.p_order = self.fit_dict["p_order"]
        self.order = self.fit_dict["order"]
        self.wvals = self.fit_dict["wvals"]
        self.zvals = self.zvals_full[: self.p_order]
        self.fvals = self.fvals_full[: self.p_order]
        self.minreal_cutoff = minreal_cutoff
        self.zpks_by_fit_idx = dict()
        if debug is not None:
            self.debug = debug
        return

    def choose(self, order):
        """ Select which order to return.

        This method selects this or a lesser order to return the results for.
        """
        # go down in index
        for idx in range(len(self.fit_list) - 1, -1, -1):
            if self.fit_list[idx]["order"] < order:
                break
        else:
            # TODO: warn user
            pass
        self.fit_idx = idx
        self.fit_dict = self.fit_list[self.fit_idx]
        self.p_order = self.fit_dict["p_order"]
        self.order = self.fit_dict["order"]
        self.wvals = self.fit_dict["wvals"]
        self.zvals = self.zvals_full[: self.p_order]
        self.fvals = self.fvals_full[: self.p_order]
        return

    def __call__(self, F_Hz):
        return tf_bary_interp(
            F_Hz,
            zvals=self.zvals,
            fvals=self.fvals,
            wvals=self.wvals,
        )

    def _zpk_compute(self):
        zpk = self.zpks_by_fit_idx.get(self.fit_idx, None)
        if zpk is None:
            zpk = tf_bary_zpk(
                fvals=self.fvals,
                zvals=self.zvals,
                wvals=self.wvals,
                minreal_cutoff=self.minreal_cutoff,
            )
            self.zpks_by_fit_idx[self.fit_idx] = zpk
        return zpk

    @property
    def supports(self):
        return self.zvals.imag

    @property
    def zpk(self):
        return self._zpk_compute()

    @property
    def poles(self):
        zeros, poles, gain = self._zpk_compute()
        return poles

    @property
    def zeros(self):
        zeros, poles, gain = self._zpk_compute()
        return zeros

    @property
    def gain(self):
        zeros, poles, gain = self._zpk_compute()
        return gain


def _drop_inplace(idx, arr):
    arr[idx:-1] = arr[idx + 1 :]
    arr.resize((len(arr) - 1,), refcheck=False)


def domain_sort(X, *Y):
    X = np.asarray(X)
    if not np.all(X[:-1] <= X[1:]):
        sort_idxs = np.argsort(X)
        X = X[sort_idxs]
        output = [X]
        for y in Y:
            if y is None:
                output.append(None)
            else:
                y = np.asarray(y)
                if len(y) == 1:
                    output.append(y)
                else:
                    output.append(y[sort_idxs])
    else:
        output = [X]
        output.extend(Y)
    return output


class Structish(object):
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            self.__dict__.update(args[0])
        elif len(args) > 1:
            raise RuntimeError(
                "Structish only takes one argument (a dictionary) and kwargs"
            )
        self.__dict__.update(kwargs)


def Q_rank_calc(z, p):
    if p.real == 0 or z.real == 0:
        if p.real == z.real:
            Q_rank = 0
        else:
            # TODO
            # should use the data spacing to regularize this case
            Q_rank = 1e3
    else:
        res_ratio = z.real / p.real
        Q_rank = abs(p - z) * (1 / (p.real) ** 2 + 1 / (z.real) ** 2) ** 0.5 + abs(
            res_ratio - 1 / res_ratio
        )
    return Q_rank


def order_reduce_zp(
    zeros,
    poles,
    Q_rank_cutoff=1e-5,
):
    rpB = nearest_pairs(zeros, poles)
    Zl = list(rpB.l1_remain)
    Pl = list(rpB.l2_remain)

    for z, p in rpB.r12_list:
        Q_rank = Q_rank_calc(p, z)
        # print("rank: ", p, z, Q_rank)
        # print(z, p, Q_rank)
        if Q_rank < Q_rank_cutoff:
            continue
        Zl.append(z)
        Pl.append(p)

    Zl = np.asarray(Zl)
    Pl = np.asarray(Pl)
    return Zl, Pl


def nearest_pairs(
    l1,
    l2,
    metric_pair_dist=None,
):
    # TODO, allow other rankings than distance

    rpB = nearest_unique_pairs(l1, l2, metric_pair_dist)
    # not going to maintain these lists
    del rpB.idx_list
    del rpB.l1
    del rpB.l2

    while True:
        pair_lists = []
        l1_nearest, l1_dist = nearest_idx(
            rpB.l1_remain,
            rpB.l2_remain,
            metric_pair_dist=metric_pair_dist,
            return_distances=True,
        )
        for idx_1, idx_2 in enumerate(l1_nearest):
            if idx_2 is None:
                continue
            dist = l1_dist[idx_1]
            pair_lists.append((dist, idx_1, idx_2))
        l2_nearest, l2_dist = nearest_idx(
            rpB.l2_remain,
            rpB.l1_remain,
            metric_pair_dist=metric_pair_dist,
            return_distances=True,
        )
        for idx_2, idx_1 in enumerate(l2_nearest):
            if idx_1 is None:
                continue
            dist = l2_dist[idx_2]
            pair_lists.append((dist, idx_1, idx_2))
        if not pair_lists:
            break
        pair_lists.sort()
        dist, idx_1, idx_2 = pair_lists[0]
        rpB.r12_list.append((rpB.l1_remain[idx_1], rpB.l2_remain[idx_2]))
        del rpB.l1_remain[idx_1]
        del rpB.l2_remain[idx_2]
    return rpB


def nearest_idx(
    lst_1,
    lst_2=None,
    metric_pair_dist=None,
    return_distances=False,
):
    """
    If lst_2 is given, this returns all of the nearest items in lst_2 to lst_1.
    If not given, this returns all of the nearest elements of lst_1 to itself,
    ignoring self elements.

    if metric_pair_dist is None, use the standard distance on complex plane.
    This is the fastest.
    """
    dists = []
    if lst_2 is not None:
        # TODO, this could be much more efficient with sorting..
        if metric_pair_dist is None:

            def metric_pair_dist(r1, r2):
                return abs(r1 - r2)

        nearest_lst = []
        for r1 in lst_1:
            if r1 is None:
                nearest_lst.append(None)
                continue
            dist_nearest = float("inf")
            idx_nearest = None
            for idx_2, r2 in enumerate(lst_2):
                if r2 is None:
                    continue
                dist = metric_pair_dist(r1, r2)
                if dist < dist_nearest:
                    idx_nearest = idx_2
                    dist_nearest = dist
            nearest_lst.append(idx_nearest)
            dists.append(dist_nearest)
    else:
        # TODO, this could be much more efficient with sorting..
        if metric_pair_dist is None:

            def metric_pair_dist(r1, r2):
                return abs(r1 - r2)

        nearest_lst = []
        for idx_1, r1 in enumerate(lst_1):
            if r1 is None:
                nearest_lst.append(None)
                continue
            dist_nearest = float("inf")
            idx_nearest = None
            for idx_2, r2 in enumerate(lst_1):
                if idx_2 == idx_1:
                    continue
                if r2 is None:
                    continue
                dist = metric_pair_dist(r1, r2)
                if dist < dist_nearest:
                    idx_nearest = idx_2
                    dist_nearest = dist
            nearest_lst.append(idx_nearest)
            dists.append(dist_nearest)
    if return_distances:
        return nearest_lst, dists
    else:
        return nearest_lst


def nearest_unique_pairs(
    l1,
    l2,
    metric_pair_dist=None,
):
    r12_list = []
    idx_list = []
    l1 = list(l1)
    l2 = list(l2)
    l1_nearest = nearest_idx(l1, l2, metric_pair_dist=metric_pair_dist)
    l2_nearest = nearest_idx(l2, l1, metric_pair_dist=metric_pair_dist)

    l1_remain = []
    l2_remain = []
    idx_2_used = []

    for idx_1, idx_2 in enumerate(l1_nearest):
        if idx_2 is None:
            l1_remain.append(l1[idx_1])
            continue
        # coding_z = aid.fitter.num_codings[idx_1]
        # coding_p = aid.fitter.den_codings[idx_2]
        # TODO annotate about stability
        p = l2[idx_2]
        z = l1[idx_1]
        if idx_1 == l2_nearest[idx_2]:
            idx_2_used.append(idx_2)
            r12_list.append((z, p))
            idx_list.append((idx_1, idx_2))
        else:
            l1_remain.append(l1[idx_1])
            l1_nearest[idx_1] = None
    idx_2_used = set(idx_2_used)
    for idx_2, p in enumerate(l2):
        if idx_2 not in idx_2_used:
            l2_remain.append(p)
            l2_nearest[idx_2] = None
    assert len(r12_list) + len(l1_remain) == len(l1)
    assert len(r12_list) + len(l2_remain) == len(l2)
    return Structish(
        r12_list=r12_list,
        l1_remain=l1_remain,
        l2_remain=l2_remain,
        idx_list=idx_list,
        l1=l1_nearest,
        l2=l2_nearest,
    )
