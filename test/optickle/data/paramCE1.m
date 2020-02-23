%---------------- p a r a m H 1 . m ---------------------
% parameters shown in LIGO-T0900043, non-folded
% Originally written by Kiwamu, and modified for L1 by KK
%
% need to update with H1 params
%--------------------------------------------------------
% In the current setting PR3, SR2 and SR3 are omitted for simplicity.
% However PR2 is included as a high reflective beam splitter so that
% the POP2 (light coming from BS to PRM) signal can be obtained.
% 

function par = paramCE1(par)

% basic constants
lambda = 1064e-9;   % Can't we get inherit these somehow?
c = 299792458;

% Laser
par.Laser.Pin = 150;
par.Laser.Wavelength = lambda;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Detector Geometry (distances in meters)
%
% 

% lasy = 12.7e-3 for Tprm = 0.03
% lasy = 8.86e-3 for Tprm = 0.01

larm = 40e3;
lPRC  = 107.0619;   % PRCL: lPRC = lPR + (lIX + lIY) / 2
lSRC  = 105.4149;   % SRCL: lSRC = lSR + (lIX + lIY) / 2  
lasy  = 8.86e-3;      % Schnupp asymmetry: lasy = lIX - lIY
lmean = 4.8298;     % (lIX + lIY) / 2
Lasy = 0e-12;         % DARM asymmetry; could be useful for sensing CARM at FSR
%--------------

% Mirror curvatures (all dimensions in meters)
Ri = 28560;         % radius of curvature of input mirrors (IX and IY)
Re = 29440;         % radius of curvature of end mirrors (EX and EY)
Rpr = -25;          % radius of curvature of power recycling mirrors
Rpr2 = 225;
Rsr = -25;         	% radius of curvature of signal recycling mirrors
Rsr2 = 225;

% Put together all the length parameters into 'par'
par.Length.IX = lmean + lasy / 2;  % distance [m] from BS to IX
par.Length.IY = lmean - lasy / 2;  % distance [m] from BS to IY
par.Length.EX = larm + Lasy / 2;  % length [m] of the X arm
par.Length.EY = larm - Lasy / 2;  % length [m] of the Y armlplp
par.Length.PR = lPRC - lmean;      % distance from PR to BS
par.Length.SR = lSRC - lmean;      % distance from SR to BS
par.Length.PR_PR2 = 16.6037;           % distance from PR to PR2
par.Length.PR2_BS = par.Length.PR - par.Length.PR_PR2;           % distance from PR2 to BS
par.Length.SR_SR2 = 16.6037;
par.Length.SR2_BS = par.Length.SR - par.Length.SR_SR2;
par.Length.OMC = 1.2;

% Put together all the Radius of Curvature [1/m] 
par.IX.Chr = 1 / Ri;
par.IY.Chr = 1 / Ri;
par.EX.Chr = 1 / Re;
par.EY.Chr = 1 / Re;
par.BS.Chr = 0;
par.PR.Chr = 1 / Rpr;
par.SR.Chr = 1 / Rsr;
par.PR2.Chr = 1/Rpr2;
par.SR2.Chr = 1/Rpr2;
par.OMCa.Chr = 1;
par.OMCb.Chr = 1;
par.PO_AS.Chr = 0;
par.FCI.Chr = 0;
par.FCE.Chr = 0;

% Microscopic length offsets
dETM = 0;            % DARM offset, for DC readout - leave this as zero
par.IX.pos = 0;
par.IY.pos = 0;
par.EX.pos = Lasy/2;      % Set DARMoffset in your own scripts, not here.
par.EY.pos = -Lasy/2;
par.BS.pos = 0;
par.PR.pos = 0;
par.SR.pos = lambda/4; % pos = lambda/4 for signal recycling
par.PR2.pos = 0;
par.SR2.pos = 0;
par.OMCa.pos = 0;
par.OMCb.pos = 0;
par.PO_AS.pos = 0;
par.FCI.pos = 0;
par.FCE.pos = 0;

% Mass [kg]
Mi = 320;
Me = 320;
dMi = 0.1;
dMe = 0.1;
par.IX.Mass = Mi + dMi/2;
par.IY.Mass = Mi - dMi/2;
par.EX.Mass = Me + dMe/2;
par.EY.Mass = Me - dMe/2;
par.BS.Mass = 320;
par.PR.Mass = 320;
par.SR.Mass = 320;
par.PR2.Mass = 320;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mirror Parameters

% % HR Transmissivities 
% dIT = 0.002;
% par.IX.T = 0.014+dIT/2;     % T for ITMX
% par.IY.T = 0.014-dIT/2;     % T for ITMY
% par.BS.T = 0.5;       % T = 50 % for BS, may have +- 0.003 RT imbalance

% par.EX.T = 5e-6;     % T = 1 for DRMI (T = 5ppm for ALIGO)
% par.EY.T = 5e-6;     % T = 1 for DRMI (T = 5ppm for ALIGO)

% par.PR.T = 0.03;       % T for PRM
% par.SR.T = 0.02;      % T for SRM

% par.PR2.T = 250e-6;  % 250ppm
% par.SR2.T = 250e-6;  % 250ppm

% par.OMCa.T = 0.01;
% par.OMCb.T = 0.01;
% par.PO_AS.T = 0.999;

% par.FCI.T = 0.0017;
% par.FCE.T = 5e-6;

% HR transmisivity
Ti = 0.014;  % ITM transmisivity
Te = 5e-6;  % ETM transmisivity
dTi = 0.05 * Ti;  % difference in ITM transmisivities
par.IX.Thr = Ti + dTi/2;
par.IY.Thr = Ti - dTi/2;
par.EX.Thr = Te;
par.EY.Thr = Te;
par.BS.Thr = 0.5;
par.PR.Thr = 0.01;
par.SR.Thr = 0.02;
par.PR2.Thr = 250e-6;
par.SR2.Thr = 250e-6;
par.OMCa.Thr = 0.01;
par.OMCb.Thr = 0.01;
par.PO_AS.Thr = 0.999;  % transmisivity of AS port pickoff
par.FCI.Thr = 0.0017;
par.FCE.Thr = 5e-6;

% Power reflectivity on AR Surfaces
% Using 40m parameters ...
%
par.IX.Rar = 0;  % designed value is 500 ppm
par.IY.Rar = 0;  % designed value is 500 ppm
par.EX.Rar = 0;  % designed value is less than 300 ppm
par.EY.Rar = 0;  % designed value is less than 300 ppm
par.BS.Rar = 0;       % designed value is less than 600 ppm 
par.PR.Rar = 0;       % designed value is less than 300 ppm
par.SR.Rar = 0;       % designed value is less than 300 ppm
par.PR2.Rar = 0;
par.SR2.Rar = 0;
par.OMCa.Rar = 0;
par.OMCb.Rar = 0;
par.PO_AS.Rar = 0;
par.FCI.Rar = 0;
par.FCE.Rar = 0;

% % HR Losses (75 ppm round trip, 50 ppm assumed in 40m)
% dIL = 5e-6;
% dEL = 5e-6;
% fcLrt = 150e-6;  % filter cavity round trip loss
% par.IX.L = 30e-6+dIL/2;
% par.IY.L = 30e-6-dIL/2;
% par.EX.L = 30e-6+dEL/2;
% par.EY.L = 30e-6-dEL/2;
% par.BS.L = 100e-6;
% par.PR.L = 1000e-6;
% par.SR.L = 1000e-6;
% par.PR2.L = 37.5e-6;
% par.SR2.L = 37.5e-6;
% par.OMCa.L = 10e-6;
% par.OMCb.L = 10e-6;
% par.PO_AS.L = 0;
% par.FCI.L = fcLrt/2;
% par.FCE.L = fcLrt/2;

% HR losses
Li = 20e-6;
Le = 20e-6;
dLi = 5e-6;  % difference in ITM losses
dLe = 5e-6;  % difference in ETM losses
fcLrt = 150e-6;  % filter cavity round trip loss
par.IX.Lhr = Li + dLi/2;
par.IY.Lhr = Li - dLi/2;
par.EX.Lhr = Le + dLe/2;
par.EY.Lhr = Li - dLi/2;
par.BS.Lhr = 100e-6;  % 37.5e-6  % GWINC
par.PR.Lhr = 0;  % 37.5e-6
par.SR.Lhr = 1000e-6;  % 37.5e-6
par.PR2.Lhr = 37.5e-6;
par.SR2.Lhr = 37.5e-6;
par.OMCa.Lhr = 10e-6;
par.OMCb.Lhr = 10e-6;
par.PO_AS.Lhr = 0;
par.FCI.Lhr = fcLrt/2;
par.FCE.Lhr = fcLrt/2;

% angles of incidence
par.IX.aoi = 0;
par.IY.aoi = 0;
par.EX.aoi = 0;
par.EY.aoi = 0;
par.BS.aoi = 45;
par.PR.aoi = 0;
par.SR.aoi = 0;
par.PR2.aoi = 10;
par.SR2.aoi = 10;
par.OMCa.aoi = 0;
par.OMCb.aoi = 0;
par.PO_AS.aoi = 0;
par.FCI.aoi = 0;
par.FCE.aoi = 0;

% Losses through medium
par.IX.Lmd = 0;
par.IY.Lmd = 0;
par.EX.Lmd = 0;
par.EY.Lmd = 0;
par.BS.Lmd = 0;
par.PR.Lmd = 0;
par.SR.Lmd = 0;
par.PR2.Lmd = 0;
par.SR2.Lmd = 0;
par.OMCa.Lmd = 0;
par.OMCb.Lmd = 0;
par.PO_AS.Lmd = 0;
par.FCI.Lmd = 0;
par.FCE.Lmd = 0;

% mechanical parameters
par.w = 2 * pi * 0.1;   % resonance frequency of the mirror (rad/s)
par.w_pit = 2 * pi * 0.3;   % pitch mode resonance frequency

% Squeezer
par.Sqz.sqAng = 89.535;  % squeeze angle [deg]
par.Sqz.sqdB = 6;  % squeezing [dB]
par.Sqz.antidB = 6;  % antisqueezing [dB]
par.Sqz.fcDetuning = -4.98;  % filter cavity detuning [Hz]
par.Sqz.fcLength = 4e3;  % filter cavity length [m]
par.Sqz.fcFSR = c/(2*par.Sqz.fcLength);

% Homodyne readout
par.Homodyne.angle = 90;
par.Homodyne.LOpower = 1;
par.Homodyne.qe = 0.96;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Beam Parameters
par.Mod.f1 = 9100574.803;          % first modulation frequency
par.Mod.f2 = 5*par.Mod.f1;                % second modulation frequency
par.Mod.g1 = 0.1; %  first modulation depth (radians)
par.Mod.g2 = 0.1; % second modulation depth (radians)
par.Mod.nMod1 = 2;              % first modulation order
par.Mod.nMod2 = 2;              % second modulation order

par.Mod.AM1 = 0;
par.Mod.AM2 = 0;
par.Mod.a1 = 0;
par.Mod.a2 = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Adjustment of demodulation phase 
% Demodulation Phases -- tuned with getDemodPhases

par.phi.REFLf1  = 0;          % f1 : adjusted for CARM, I-phase
par.phi.REFLf2  = 5;             % f2 : adjusted for CARM, I-phase
par.phi.REFL3f1 = 0;            % 3*f1: adjusted for PRCL, I-phase
par.phi.REFL3f2 = 0;         % 3*f2: adjusted for SRCL, I-phase
par.phi.REFLM = 0;
par.phi.REFLP = 0;

par.phi.ASf1    = 0;       % f1 : adjusted for DARM, Q-pjase
par.phi.ASf2    = 0;      % f2 : adjusted for DARM, Q-phase
par.phi.AS3f1   = 0;        % 3f1: adjusted for MICH, Q-phase
par.phi.AS3f2   = 0;
par.phi.ASM = 0;
par.phi.ASP = 0;

par.phi.POPf1 = 0;             % f1 : adjusted for PRCL, I-phase
par.phi.POPf2 = 0;            % f2 : adjusted for SRCL, I-phase
par.phi.POP3f1 = 0;           % 3f1: adjusted for PRCL, I-phase
par.phi.POP3f2 = 0;            % 3f2: adjusted for SRCL, I-phase
par.phi.POPM = 0;
par.phi.POPP = 0;

par.phi.POXf1 = 0;            % f1 : adjusted for PRCL, I-phase
par.phi.POXf2 = 0;        % f2 : adjusted for MICH, Q-phase
par.phi.POX3f1 = 0;           % 3f1: adjusted for PRCL, I-phase
par.phi.POX3f2 = 0;      % 3f2: adjusted for MICH, Q-phase
par.phi.POXM = 0;
par.phi.POXP = 0;

par.phi.POYf1 = 0;
par.phi.POYf2 = 0;
par.phi.POY3f1 = 0;
par.phi.POY3f2 = 0;
par.phi.POYM = 0;
par.phi.POYP = 0;

par.phi.POSf1 = 0;
par.phi.POSf2 = 0;
par.phi.POS3f1 = 0;
par.phi.POS3f2 = 0;
par.phi.POSM = 0;
par.phi.POSP = 0;
