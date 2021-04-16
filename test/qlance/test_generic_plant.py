"""
Unit tests for generic plants
"""

import numpy as np
import qlance.plant as qplant
import qlance.controls as ctrl
import pytest


ff = np.logspace(0, 3, 100)
wx = -100
wy = -110
xfilter = ctrl.Filter([], wx, 1, Hz=False)
yfilter = ctrl.Filter([], wy, 1, Hz=False)
xdata = xfilter(ff)
ydata = yfilter(ff)
CARM = {'EX': 1, 'EY': 1}
DARM = {'EX': 1, 'EY': -1}


class TestPlant:

    plant = qplant.Plant()
    plant.addPlant('REFL', 'EX', xdata)
    plant.addPlant('REFL', 'EY', lambda ff: -1/wy / (1 - 2j*np.pi*ff/wy))
    plant.addPlant('AS', 'EX', xfilter)
    plant.addPlant('AS', 'EY', -ydata)
    plant.ff = ff

    def test_xfit(self):
        tf = self.plant.getTF('REFL', 'EX', fit=True)
        z, p, k = tf.get_zpk(Hz=False)
        zclose = np.allclose(z, np.array([]))
        pclose = np.allclose(p, np.array([wx]))
        kclose = np.isclose(k, 1)
        assert np.all([zclose, pclose, kclose])

    def test_sum_fit(self):
        tf = self.plant.getTF('REFL', CARM, fit=True)
        z, p, k = tf.get_zpk(Hz=False)
        zclose = np.allclose(z, np.array([(wx + wy)/2]))
        pclose = np.allclose(p, np.array([wx, wy]))
        kclose = np.isclose(k, 2)
        assert np.all([zclose, pclose, kclose])

    def test_AS_DARM(self):
        tf = self.plant.getTF('AS', DARM)
        assert np.allclose(tf, xdata + ydata)

    def test_AS_CARM(self):
        tf = self.plant.getTF('AS', CARM)
        assert np.allclose(tf, xdata - ydata)


@pytest.fixture
def base_plant():
    plant = qplant.Plant()
    plant.addPlant('REFL', 'EX', xfilter)
    plant.addPlant('REFL', 'EY', lambda ff: -1/wy / (1 - 2j*np.pi*ff/wy))
    return plant


def test_update_freq(base_plant):
    base_plant.ff = ff
    x1 = np.allclose(base_plant.getTF('REFL', 'EX'), xdata)
    y1 = np.allclose(base_plant.getTF('REFL', 'EY'), ydata)
    ff2 = np.logspace(1, 2, 50)
    base_plant.ff = ff2
    x2 = np.allclose(base_plant.getTF('REFL', 'EX'), xfilter(ff2))
    y2 = np.allclose(base_plant.getTF('REFL', 'EY'), yfilter(ff2))
    assert np.all([x1, y1, x2, y2])


def test_add_after_freq(base_plant):
    base_plant.ff = ff
    base_plant.addPlant('AS', 'EX', xdata)
    tf = base_plant.getTF('AS', 'EX')
    assert np.allclose(tf, xdata)


def test_update_freq_data(base_plant):
    base_plant.ff = ff
    base_plant.addPlant('AS', 'EX', xdata)
    msg = 'Can\'t specify a new frequency vector if any plants are ' \
        + 'defined by data.'
    with pytest.raises(ValueError, match=msg):
        base_plant.ff = np.linspace(1, 10, 10)


def test_add_redundant(base_plant):
    with pytest.raises(
            ValueError, match='There is already a plant from EX to REFL'):
        base_plant.addPlant('REFL', 'EX', yfilter)


def test_add_incompatible_data(base_plant):
    base_plant.addPlant('AS', 'EX', xdata)
    msg = 'This data has a different length than existing data'
    with pytest.raises(ValueError, match=msg):
        base_plant.addPlant('AS', 'EY', -ydata[:10])


def test_incompatible_freq(base_plant):
    base_plant.addPlant('AS', 'EX', xdata)
    msg = 'Frequency vector does not have the same length as existing data'
    with pytest.raises(ValueError, match=msg):
        base_plant.ff = np.linspace(1, 10, 10)


class TestControls:

    plant = qplant.Plant()
    plant.addPlant('REFL', 'EX', xdata)
    plant.addPlant('REFL', 'EY', lambda ff: -1/wy / (1 - 2j*np.pi*ff/wy))
    plant.addPlant('AS', 'EX', xfilter)
    plant.addPlant('AS', 'EY', -ydata)
    plant.ff = ff

    filtDARM = ctrl.Filter([], 1, 100, 1, Hz=True)
    filtCARM = ctrl.Filter([], 5, 80, 1, Hz=True)

    cs = ctrl.ControlSystem()
    cs.addDOF('CARM', 'REFL', CARM)
    cs.addDOF('DARM', 'AS', DARM)
    cs.addFilter('CARM', 'CARM', filtCARM)
    cs.addFilter('DARM', 'DARM', filtDARM)
    cs.setOptomechanicalPlant(plant)
    cs.run(mechmod=False)

    def test_oltf(self):
        oltf1 = self.cs.getOLTF('DARM', 'DARM', 'err')
        oltf2 = self.plant.getTF('AS', DARM) * self.filtDARM(ff)
        assert np.allclose(oltf1, oltf2, rtol=1e-2)

    def test_cltf(self):
        cltf1 = self.cs.getCLTF('CARM', 'CARM', 'err')
        cltf2 = 1 / (1 - self.plant.getTF('REFL', CARM) * self.filtCARM(ff))
        assert np.allclose(cltf1, cltf2, rtol=1e-2)
