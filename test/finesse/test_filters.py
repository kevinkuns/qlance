"""
Unit tests for control systems
"""

import numpy as np
import pytickle.controls as ctrl
import scipy.signal as sig
import h5py
import os
import pytest


def check_zpk_equality(zpk1, zpk2):
    z1, p1, k1 = zpk1
    z2, p2, k2 = zpk2
    c1 = np.all(np.isclose(np.sort(z1), np.sort(z2)))
    c2 = np.all(np.isclose(np.sort(p1), np.sort(p2)))
    c3 = np.isclose(k1, k2)
    return c1, c2, c3


def check_filter_equality(filt1, filt2):
    zpk1 = filt1.get_zpk()
    zpk2 = filt2.get_zpk()
    return np.all(check_zpk_equality(zpk1, zpk2))


class TestFilters:

    z1 = np.array([1, 2 + 3j, 2 - 3j])
    p1 = np.array([8, 3 + 2j, 3 - 2j])
    k1 = 4
    z2 = []
    p2 = ctrl.resRoots(42, 238)
    k2 = 6
    f4 = 10  # reference frequency
    g4 = 3  # gain at reference frequency
    ff = np.logspace(0, 4, 500)

    filt1a = ctrl.Filter(z1, p1, k1)
    filt1b = ctrl.Filter(-2*np.pi*z1, -2*np.pi*p1, k1, Hz=False)
    filt1c = ctrl.Filter(dict(zs=z1, ps=p1, k=k1))
    filt1d = ctrl.Filter(dict(zs=-2*np.pi*z1, ps=-2*np.pi*p1, k=k1), Hz=False)
    filt2a = ctrl.Filter(z2, p2, k2)
    # filt2b = ctrl.Filter(
    #     lambda ss: k2/((ss + 2*np.pi*p2[0])*(ss + 2*np.pi*p2[1])))
    filt3a = ctrl.Filter(
        ctrl.catzp(z1, z2), ctrl.catzp(p1, p2), k1*k2)
    filt3b = ctrl.catfilt(filt1a, filt2a)
    filt3c = ctrl.catfilt(filt1b, filt2a)
    # filt3d = ctrl.catfilt(filt1a, filt2b)
    filt4 = ctrl.Filter(z2, p2, g4, f4)

    data1 = h5py.File('test_filters.hdf5', 'w')
    filt1a.to_hdf5('filt1', data1)
    filt2a.to_hdf5('filt2', data1)
    data1.close()
    data2 = h5py.File('test_filters.hdf5', 'r')
    filt1r = ctrl.filt_from_hdf5('filt1', data2)
    filt2r = ctrl.filt_from_hdf5('filt2', data2)
    data2.close()
    os.remove('test_filters.hdf5')

    def test_1a(self):
        zpk1 = self.filt1a.get_zpk(Hz=True)
        assert np.all(check_zpk_equality(zpk1, (self.z1, self.p1, self.k1)))

    def test_1freq(self):
        zpk1 = self.filt1a.get_zpk(Hz=True)
        zpk2 = self.filt1b.get_zpk(Hz=True)
        assert np.all(check_zpk_equality(zpk1, zpk2))

    def test_1s(self):
        zpk1 = self.filt1a.get_zpk()
        zpk2 = self.filt1b.get_zpk()
        assert np.all(check_zpk_equality(zpk1, zpk2))

    def test_1c(self):
        zpk1 = self.filt1a.get_zpk()
        zpk2 = self.filt1c.get_zpk()
        assert np.all(check_zpk_equality(zpk1, zpk2))

    def test_1d(self):
        zpk1 = self.filt1a.get_zpk()
        zpk2 = self.filt1d.get_zpk()
        assert np.all(check_zpk_equality(zpk1, zpk2))

    def test_2(self):
        k2 = self.k2
        p2 = self.p2
        filt2b = ctrl.Filter(
            lambda ss: k2/((ss + 2*np.pi*p2[0])*(ss + 2*np.pi*p2[1])))
        data1 = self.filt2a.computeFilter(self.ff)
        # data2 = self.filt2b.computeFilter(self.ff)
        data2 = filt2b.computeFilter(self.ff)
        assert np.allclose(data1, data2)

    def test_cat1(self):
        assert check_filter_equality(self.filt3a, self.filt3b)

    def test_cat2(self):
        assert check_filter_equality(self.filt3a, self.filt3c)

    def test_cat3(self):
        k2 = self.k2
        p2 = self.p2
        filt2b = ctrl.Filter(
            lambda ss: k2/((ss + 2*np.pi*p2[0])*(ss + 2*np.pi*p2[1])))
        filt3d = ctrl.catfilt(self.filt1a, filt2b)
        data1 = self.filt3a.computeFilter(self.ff)
        # data2 = self.filt3d.computeFilter(self.ff)
        data2 = filt3d.computeFilter(self.ff)
        assert np.allclose(data1, data2)

    def test_gain(self):
        mag = np.abs(self.filt4.computeFilter(self.f4))
        assert np.isclose(mag, self.g4)

    def test_ss1(self):
        ss = self.filt1a.get_state_space()
        data1 = self.filt1a.computeFilter(self.ff)
        _, data2 = sig.freqresp(ss, 2*np.pi*self.ff)
        assert np.allclose(data1, data2)

    def test_ss2(self):
        ss = self.filt2a.get_state_space()
        data1 = self.filt2a.computeFilter(self.ff)
        _, data2 = sig.freqresp(ss, 2*np.pi*self.ff)
        assert np.allclose(data1, data2)

    def test_ss3(self):
        ss = self.filt3a.get_state_space()
        data1 = self.filt3a.computeFilter(self.ff)
        _, data2 = sig.freqresp(ss, 2*np.pi*self.ff)
        assert np.allclose(data1, data2)

    def test_1r(self):
        zpk1 = self.filt1a.get_zpk()
        zpk2 = self.filt1r.get_zpk()
        assert np.all(check_zpk_equality(zpk1, zpk2))

    def test_2r(self):
        zpk1 = self.filt2a.get_zpk()
        zpk2 = self.filt2r.get_zpk()
        assert np.all(check_zpk_equality(zpk1, zpk2))
