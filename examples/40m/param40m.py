# Parameters for 40m interferometer

import numpy as np


def param40m(Larm=37.795, Tprm=0.056, Tsrm=0.01, dTitm=0, dLetm=0,
             dmu=0, green=False):
    par = {}

    if green:
        par['lambda'] = np.array([1064e-9, 532e-9])
        par['lambda0'] = par['lambda'][0]
    else:
        par['lambda'] = 1064e-9
        par['lambda0'] = par['lambda']

    # Frequencies
    # 40m eLOG 11745
    par['Mod'] = {}
    par['Mod']['f1'] = 11.066209e6  # [Hz]
    par['Mod']['f2'] = 5 * par['Mod']['f1']  # [Hz]
    par['Mod']['g1'] = 0.179
    par['Mod']['g2'] = 0.226

    # Probe demodulation phases [deg]
    par['DemodPhase'] = {}
    par['DemodPhase']['REFL11'] = 179
    par['DemodPhase']['REFL55'] = -9.5
    par['DemodPhase']['REFL33'] = 0
    par['DemodPhase']['REFL165'] = 0
    par['DemodPhase']['AS55'] = 86
    par['DemodPhase']['AS110'] = 0
    par['DemodPhase']['AS165'] = 0
    par['DemodPhase']['POP55'] = 0
    par['DemodPhase']['POP22'] = 0
    par['DemodPhase']['POP110'] = 0

    par['qe'] = 0.95  # PD quantum efficiency

    # Lengths [m]
    par['Length'] = {}
    par['Length']['PR1'] = 1.9348
    par['Length']['PR2'] = 2.0808
    par['Length']['PR3'] = 0.401161 + 0.0843
    par['Length']['SR1'] = 0.5887
    par['Length']['SR2'] = 2.2794 - 1.2707
    par['Length']['SR3'] = 0.1947
    par['Length']['BS_X'] = 2.2403
    par['Length']['BS_Y'] = 2.2403 + 0.02319
    par['Length']['Xarm'] = Larm
    par['Length']['Yarm'] = Larm
    par['Length']['OMCS'] = 0.2815  # short leg of a bow-tie cavities
    par['Length']['OMCL'] = 0.2842  # long leg of a bow-tie cavities

    par['IX'] = {}
    par['IY'] = {}
    par['EX'] = {}
    par['EY'] = {}
    par['BS'] = {}
    par['PRM'] = {}
    par['SRM'] = {}
    par['PR2'] = {}
    par['PR3'] = {}
    par['SR2'] = {}
    par['SR3'] = {}
    par['IC'] = {}
    par['OC'] = {}
    par['CM1'] = {}
    par['CM2'] = {}

    # tuning length offsets [m]
    DARM_asym = 0
    par['IX']['pos'] = 0
    par['IY']['pos'] = 0
    par['EX']['pos'] = DARM_asym
    par['EY']['pos'] = -DARM_asym
    par['BS']['pos'] = 0
    par['PRM']['pos'] = 0
    par['SRM']['pos'] = 0
    par['PR2']['pos'] = 0
    par['PR3']['pos'] = 0
    par['SR2']['pos'] = 0
    par['SR3']['pos'] = 0

    # HR power transmissions
    if green:
        T_ITM = [[0.01384, par['lambda'][0]],
                 [0.015, par['lambda'][1]]]
        T_ETM = [[13.7e-6, par['lambda'][0]],
                 [0.045, par['lambda'][1]]]
    else:
        T_ITM = 0.01384
        T_ETM = 13.7e-6
    T_FOLDING = 20e-6
    par['IX']['Thr'] = T_ITM + dTitm
    par['IY']['Thr'] = T_ITM - dTitm
    par['EX']['Thr'] = T_ETM
    par['EY']['Thr'] = T_ETM
    par['BS']['Thr'] = 0.5
    par['PRM']['Thr'] = Tprm  # 0.05637  # 0.01
    par['SRM']['Thr'] = Tsrm
    par['PR2']['Thr'] = 20e-6  # 0.01  # 50e-6
    par['PR3']['Thr'] = T_FOLDING  # 50e-6  # 100e-6  # 0.01  # 50e-6
    par['SR2']['Thr'] = T_FOLDING  # 20e-6  # 100e-6  # 0.01  # 1e-3
    par['SR3']['Thr'] = T_FOLDING  # 50e-6  # 100e-6  # 0.01  # 1e-3
    par['IC']['Thr'] = 7600e-6
    par['OC']['Thr'] = 7600e-6
    par['CM1']['Thr'] = 36e-6
    par['CM2']['Thr'] = 36e-6

    # AR power reflectivities
    if green:
        R_ITM = [[417e-6, par['lambda'][0]],
                 [0, par['lambda'][1]]]
        R_ETM = [[93.3e-6, par['lambda'][0]],
                 [0, par['lambda'][1]]]
    else:
        R_ITM = 417e-6
        R_ETM = 93.3e-6
    R_Steer = 0
    par['IX']['Rar'] = R_ITM
    par['IY']['Rar'] = R_ITM
    par['EX']['Rar'] = R_ETM
    par['EY']['Rar'] = R_ETM
    par['BS']['Rar'] = 0
    par['PRM']['Rar'] = 160e-6
    par['SRM']['Rar'] = 0
    par['PR2']['Rar'] = R_Steer
    par['PR3']['Rar'] = R_Steer
    par['SR2']['Rar'] = R_Steer
    par['SR3']['Rar'] = R_Steer
    par['IC']['Rar'] = 0
    par['OC']['Rar'] = 0
    par['CM1']['Rar'] = 0
    par['CM2']['Rar'] = 0

    # 1 / ROC for HR side [1/m]
    C_ITM = 0
    C_Steer = 0
    par['IX']['Chr'] = C_ITM
    par['IY']['Chr'] = C_ITM
    par['EX']['Chr'] = 1/59.5
    par['EY']['Chr'] = 1/60.2
    par['BS']['Chr'] = 0
    par['PRM']['Chr'] = 1 / 115.5
    par['SRM']['Chr'] = 1 / 142
    par['PR2']['Chr'] = C_Steer
    par['PR3']['Chr'] = C_Steer
    par['SR2']['Chr'] = C_Steer
    par['SR3']['Chr'] = C_Steer
    par['IC']['Chr'] = 0
    par['OC']['Chr'] = 0
    par['CM1']['Chr'] = 1/2.57
    par['CM2']['Chr'] = 1/2.57

    # HR losses
    HR_ETM = 10e-6
    HR_ITM = 15e-6
    HR_loss = 50e-6
    HR_OMC = 20e-6
    par['IX']['Lhr'] = HR_ITM
    par['IY']['Lhr'] = HR_ITM
    par['EX']['Lhr'] = HR_ETM + dLetm
    par['EY']['Lhr'] = HR_ETM - dLetm
    par['BS']['Lhr'] = HR_loss
    par['PRM']['Lhr'] = HR_loss
    par['SRM']['Lhr'] = HR_loss
    par['PR2']['Lhr'] = HR_loss
    par['PR3']['Lhr'] = HR_loss
    par['SR2']['Lhr'] = HR_loss
    par['SR3']['Lhr'] = HR_loss
    par['IC']['Lhr'] = HR_OMC
    par['OC']['Lhr'] = HR_OMC
    par['CM1']['Lhr'] = HR_OMC
    par['CM2']['Lhr'] = HR_OMC

    # losses through one pass of the medium
    MD_loss = 0
    par['IX']['Lmd'] = MD_loss
    par['IY']['Lmd'] = MD_loss
    par['EX']['Lmd'] = MD_loss
    par['EY']['Lmd'] = MD_loss
    par['BS']['Lmd'] = MD_loss
    par['PRM']['Lmd'] = MD_loss
    par['SRM']['Lmd'] = MD_loss
    par['PR2']['Lmd'] = MD_loss
    par['PR3']['Lmd'] = MD_loss
    par['SR2']['Lmd'] = MD_loss
    par['SR3']['Lmd'] = MD_loss
    par['IC']['Lmd'] = MD_loss
    par['OC']['Lmd'] = MD_loss
    par['CM1']['Lmd'] = MD_loss
    par['CM2']['Lmd'] = MD_loss

    # Masses [kg]
    M_ITM = 0.2642
    M_ETM = 0.2642
    M = 0.2642
    par['IX']['mass'] = M_ITM
    par['IY']['mass'] = M_ITM
    par['EX']['mass'] = M_ETM + 2*dmu
    par['EY']['mass'] = M_ETM - 2*dmu
    par['BS']['mass'] = M
    par['PRM']['mass'] = M
    par['SRM']['mass'] = M
    par['PR2']['mass'] = M
    par['PR3']['mass'] = M
    par['SR2']['mass'] = M
    par['SR3']['mass'] = M

    # angles of incidences
    par['IX']['aoi'] = 0
    par['IY']['aoi'] = 0
    par['EX']['aoi'] = 0
    par['EY']['aoi'] = 0
    par['BS']['aoi'] = 45
    par['PRM']['aoi'] = 0
    par['SRM']['aoi'] = 0
    par['PR2']['aoi'] = 0
    par['PR3']['aoi'] = 0
    par['SR2']['aoi'] = 0
    par['SR3']['aoi'] = 0
    par['IC']['aoi'] = 4
    par['OC']['aoi'] = 4
    par['CM1']['aoi'] = 4
    par['CM2']['aoi'] = 4

    # Resonance frequency of mirror suspensions [Hz]
    par['w'] = 2*np.pi*0.45

    return par
