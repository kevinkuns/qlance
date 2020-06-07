"""
Unit tests for Finesse MIMO control of FPMI
"""

import numpy as np
import pytickle.finesse as fin
import pytickle.controls as ctrl
import scipy.signal as sig
import pytickle.noise as pytnoise
import parFPMI
import katFPMI
import pytest


data_ref = np.load(
    'data/finesse_control_FPMI_data.npz', allow_pickle=True)['data'][()]


def check_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        raise ValueError('dictionaries are definitely not the same')

    equal = []
    for key, val1 in dict1.items():
        equal.append(np.allclose(val1, dict2[key]))
    return equal


data = {}

##############################################################################
# suspension plant
##############################################################################


def define_filt(zpk):
    return ctrl.Filter(zpk.zeros, zpk.poles, zpk.gain, Hz=False)


# rough starting guesses for resonances and Q's
w1 = 2*np.pi*0.1
w2 = 2*np.pi*0.1
w3 = 2*np.pi*1
Q1 = 100
Q2 = 100
Q3 = 100
m1 = 600
m2 = 400
m3 = 100

k1 = w1**2 * m1
k2 = w2**2 * m2
k3 = w3**2 * m3

KK = np.array([[-(k1 + k2)/m1, k2/m1, 0],
               [k2/m2, -(k2 + k3)/m2, k3/m2],
               [0, k3/m3, -k3/m3]])

A = np.block([[np.zeros((3, 3)), np.eye(3)],
               [KK, np.diag([-w1/Q1, -w2/Q2, -w3/Q3])]])

B_tst = np.zeros((6, 1))
B_pum = np.zeros((6, 1))
B_sus = np.zeros((6, 1))
C_tst = np.zeros((1, 6))
B_tst[5, 0] = 1/m3   # force on test mass
B_pum[4, 0] = 1/m2   # force on PUM
B_sus[3, 0] = k1/m1  # suspension point motion
C_tst[0, 2] = 1      # observe test mass motion
D = np.array([[0]])

# compute state space representation
ss_tst = sig.StateSpace(A, B_tst, C_tst, D)
ss_pum = sig.StateSpace(A, B_pum, C_tst, D)
ss_sus = sig.StateSpace(A, B_sus, C_tst, D)

# convert to zpk
zpk_tst = ss_tst.to_zpk()
zpk_pum = ss_pum.to_zpk()
zpk_sus = ss_sus.to_zpk()

# convert to PyTickle Filters
tst2tst = define_filt(zpk_tst)  # test mass force to position
pum2tst = define_filt(zpk_pum)  # PUM force to test mass position
sus2tst = define_filt(zpk_sus)  # sus point position to test mass position


##############################################################################
# compute the optomechanical plant
##############################################################################

par = parFPMI.parFPMI(zpk_tst.zeros, zpk_tst.poles, zpk_tst.gain)
kat = katFPMI.katFPMI(par)
katFR = fin.KatFR(kat, all_drives=False)
katFR.addDrives(['EX', 'EY', 'BS'])

# Run the model
fmin = 0.3
fmax = 1e3
npts = 400
katFR.run(fmin, fmax, npts)
ff = katFR.ff

# Define the probes that will be used to sense each DOF.
DARM = dict(EX=1/2, EY=-1/2)
CARM = dict(EX=1/2, EY=1/2)

probesDARM = {'AS_Q': 1/np.abs(katFR.getTF('AS_Q', DARM)[0])}
probesCARM = {'REFL_I': 1/np.abs(katFR.getTF('REFL_I', CARM)[0])}
probesBS   = {'REFL_Q': 1/np.abs(katFR.getTF('REFL_Q', 'BS')[0])}


##############################################################################
# define control system
##############################################################################

# define the DOF filters
filtDARM = ctrl.Filter(
    ctrl.catzp(ctrl.resRoots(2, 1), ctrl.resRoots(5, 1)),
    ctrl.catzp(0, 0, 0, ctrl.resRoots(0.5, 1), ctrl.resRoots(110, 1)),
    -1, 20)

filtCARM = filtDARM

filtBS   = ctrl.Filter(
    ctrl.catzp(ctrl.resRoots(1, 1), ctrl.resRoots(3, 1)),
    ctrl.catzp(0, 0, 0, ctrl.resRoots(0.3, 1), ctrl.resRoots(90, 1)),
    -1, 15)

# simple constant feedforward filter
filtFF   = ctrl.Filter([], [], -4.98e-3)

# Define control system
cs = ctrl.ControlSystem()

# define degrees of freedom
cs.addDOF('DARM', probesDARM, DARM)
cs.addDOF('CARM', probesCARM, CARM)
cs.addDOF('BS', probesBS, 'BS')

# define control filters
cs.addFilter('DARM', 'DARM', filtDARM)
cs.addFilter('CARM', 'CARM', filtCARM)
cs.addFilter('BS', 'BS', filtBS)

# add the feedforward
cs.addFilter('DARM', 'BS', ctrl.catfilt(filtFF, filtBS))

# set the optomechanical plant
cs.setOptomechanicalPlant(katFR)

# compensation
gainf0 = 1/np.abs(pum2tst.computeFilter(10))
pum2tst_comp = ctrl.Filter(
    ctrl.catzp(
        ctrl.resRoots(63e-3, 10),
        ctrl.resRoots(0.145, 10), ctrl.resRoots(1.12, 15)),
    ctrl.resRoots(0.13, 10),
    gainf0, 10)

for drive in ['EX', 'EY', 'BS']:
    cs.setActuator(drive, 'pos', pum2tst)
    cs.addCompensator(drive, 'pos', pum2tst_comp)


##############################################################################
# run the model
##############################################################################

cs.run()


##############################################################################
# compute some loops
##############################################################################

data['oltfs'] = dict(DARM_DARM_err=cs.getOLTF('DARM', 'DARM', 'err'),
                     CARM_CARM_err=cs.getOLTF('CARM', 'CARM', 'err'),
                     BS_BS_err=cs.getOLTF('BS', 'BS', 'err'),
                     EX_EX_drive=cs.getOLTF('EX.pos', 'EX.pos', 'drive'),
                     EY_EY_act=cs.getOLTF('EY.pos', 'EY.pos', 'act'),
                     DARM_DARM_ctrl=cs.getOLTF('DARM', 'DARM', 'ctrl'),
                     REFLQ_REFLQ_sens=cs.getOLTF('REFL_Q', 'REFL_Q', 'sens'))

data['cltfs'] = dict(DARM_DARM_err=cs.getCLTF('DARM', 'DARM', 'err'),
                     CARM_CARM_err=cs.getCLTF('CARM', 'CARM', 'err'),
                     BS_BS_err=cs.getCLTF('BS', 'BS', 'err'),
                     EX_EX_drive=cs.getCLTF('EX.pos', 'EX.pos', 'drive'),
                     EY_EY_act=cs.getCLTF('EY.pos', 'EY.pos', 'act'),
                     DARM_DARM_ctrl=cs.getCLTF('DARM', 'DARM', 'ctrl'),
                     REFLQ_REFLQ_sens=cs.getCLTF('REFL_Q', 'REFL_Q', 'sens'))


data['cross_couplings'] = dict(
    DARM_CARM_err=cs.getOLTF('DARM', 'CARM', 'err'),
    DARM_BS_err=cs.getOLTF('DARM', 'BS', 'err'),
    CARM_BS_ctrl=cs.getOLTF('CARM', 'BS', 'ctrl'),
    EX_BS_drive=cs.getOLTF('EX.pos', 'BS.pos', 'drive'),
    BS_EY_act=cs.getOLTF('BS.pos', 'EY.pos', 'act'),
    DARM_BS_ctrl=cs.getCLTF('DARM', 'BS', 'ctrl'),
    REFLQ_REFLI_sens=cs.getCLTF('REFL_Q', 'REFL_I', 'sens'))

data['cal'] = dict(
    DARM_DARM_closed_err=cs.getTF('DARM', 'err', 'DARM', 'cal'),
    DARM_DARM_open_err=cs.getTF('DARM', 'err', 'DARM', 'cal', closed=False),
    BS_BS_closed_act=cs.getTF('BS.pos', 'act', 'BS', 'cal'),
    DARM_CARM_closed_err=cs.getTF('DARM', 'err', 'CARM', 'cal'),
    DARM_BS_open_err=cs.getTF('DARM', 'err', 'BS', 'cal', closed=False),
    CARM_BS_open_ctrl=cs.getTF('CARM', 'ctrl', 'BS', 'cal', closed=False),
    EX_BS_closed_drive=cs.getTF('EX.pos', 'drive', 'BS', 'cal'),
    BS_DARM_closed_act=cs.getTF('BS.pos', 'act', 'DARM', 'cal'),
    DARM_BS_open_ctrl=cs.getTF('DARM', 'ctrl', 'BS', 'cal', closed=False),
    REFLQ_BS_closed_sens=cs.getTF('REFL_Q', 'sens', 'BS', 'cal'))


##############################################################################
# compute noise
##############################################################################

# compute the closed loop DARM calibration
DARM_cal_closed = np.abs(cs.getTF('DARM', 'err', 'DARM', 'cal'))

# dictionary of quantum noise for each probe in W/rtHz
shot_noise = {probe: katFR.getQuantumNoise(probe) for probe in cs.probes}

# compute the signal referred sensing noise
sensing_noise = {}
for probe, sn in shot_noise.items():
    # closed loop propagation from probe to error signal
    tf = np.abs(cs.getTF('DARM', 'err', probe, 'sens'))
    sensing_noise[probe] = tf * sn / DARM_cal_closed

# filter the seismic noise through the suspensions
seismic_sus = 1e-8 / ff**2
seismic_tst = np.abs(sus2tst.computeFilter(ff)) * seismic_sus

# Seismic noise at the test masss and beamsplitter in m/rtHz
seismic_noise = {drive: seismic_tst for drive in cs.drives}

# compute the signal referred displacement noise
displacement_noise = {}
for drive, sn in seismic_noise.items():
    # closed loop propagation from drive to error signal
    tf = np.abs(cs.getTF('DARM', 'err', drive, 'drive'))
    displacement_noise[drive] = tf * sn / DARM_cal_closed

# add all the noise sources in quadrature
total_noise = np.sum([sn**2 for sn in sensing_noise.values()], axis=0)
total_noise += np.sum([dn**2 for dn in displacement_noise.values()], axis=0)
total_noise = np.sqrt(total_noise)

data['noise'] = dict(
    AS_Q=sensing_noise['AS_Q'],
    REFL_Q=sensing_noise['REFL_Q'],
    REFL_I=sensing_noise['REFL_I'],
    EX=displacement_noise['EX.pos'],
    EY=displacement_noise['EY.pos'],
    BS=displacement_noise['BS.pos'],
    total=total_noise)

##############################################################################
# residual motion
##############################################################################

# Residual motion due to seismic noise
residual_seismic = {
    drive: cs.getTotalNoiseTo(drive, 'pos', 'drive', seismic_noise)
    for drive in cs.drives}
rms_seismic = {drive: pytnoise.computeRMS(ff, asd)
               for drive, asd in residual_seismic.items()}

# Residual motion due to sensing noise
residual_sensing = {drive: cs.getTotalNoiseTo(drive, 'pos', 'sens', shot_noise)
                    for drive in cs.drives}
rms_sensing = {drive: pytnoise.computeRMS(ff, asd)
               for drive, asd in residual_sensing.items()}

# Total residual motion
# Add the motion due to seismic noise and shot noise in quadrature
residual_total = {
    drive: np.sqrt(residual_seismic[drive]**2 + residual_sensing[drive]**2)
    for drive in cs.drives}
rms_total = {
    drive: pytnoise.computeRMS(ff, asd) for drive, asd in residual_total.items()}

data['residuals'] = dict(
    EX_seis=residual_seismic['EX.pos'],
    BS_seis=residual_seismic['BS.pos'],
    EX_sens=residual_sensing['EX.pos'],
    BS_sens=residual_sensing['BS.pos'],
    EX_tot=rms_total['EX.pos'],
    BS_tot=rms_total['BS.pos'],
    free=seismic_tst)


##############################################################################
# do the tests
##############################################################################


def test_oltfs():
    assert check_dicts(data_ref['oltfs'], data['oltfs'])


def test_cltfs():
    assert check_dicts(data_ref['cltfs'], data['cltfs'])


def test_cross_couplings():
    assert check_dicts(data_ref['cross_couplings'], data['cross_couplings'])


def test_cal():
    assert check_dicts(data_ref['cal'], data['cal'])


def test_noise():
    assert check_dicts(data_ref['noise'], data['noise'])


def test_residuals():
    assert check_dicts(data_ref['residuals'], data['residuals'])
