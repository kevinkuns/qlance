%---------------- p r o b e s H 1. m -------------
% 
% Kiwamu's probesC1_00.m for L1 by KK
%
%--------------------------------------------------------
%
%[Description]
%   This function adds the neccessary RFPDs and DCPDs on
%  an interferomter model. In the name of this file, '00' means
% the TEM00 mode and hence these are the probes dedicated only for
% the length sensing and not for angular sensing.
% Example usage : 
%              par = paramC1;
%              opt = optC1(par);
%              opt = probesC1_00(opt, par);
%--------------------------------------------------------
%
% [Notes]
%  This file is a modified version of the eLIGO opticle file
% called probesH1_00.m.
%


function probesCE(opt, par, includeRF)
    % Add homodyne readout

    addHomodyneReadoutPO(opt, 'OMC', par.Homodyne.angle, par.Homodyne.qe);
    opt.addLink('OMCb', 'bk', 'OMC_BS', 'fr', 0);
    dL = par.Homodyne.angle/2 * par.Laser.Wavelength/360;
    opt.setPosOffset('OMC_LOphase', dL);
    % opt.addLink('PR2', 'bkB', 'OMC_LOphase', 'fr', 0);
    opt.addMirror('PR2PO', 0, 0, 0.5);
    opt.addLink('PR2', 'bkB', 'PR2PO', 'fr', 0);
    opt.addLink('PR2PO', 'bk', 'OMC_LOphase', 'fr', 0);

    if includeRF > 0
        addAllReadout(opt, par, 'PO_AS', 'fr', 'AS', includeRF);
        addAllReadout(opt, par, 'PR', 'bk', 'REFL', includeRF);
        % addAllReadout(opt, par, 'PR2', 'bkB', 'POP', includeRF);
        addAllReadout(opt, par, 'PR2PO', 'fr', 'POP', includeRF);
        addAllReadout(opt, par, 'IX', 'po', 'POX', includeRF);
        addAllReadout(opt, par, 'SR2', 'bkB', 'POS', includeRF);

        opt.addSink('TRX');
        opt.addSink('TRY');
        opt.addLink('EX', 'bk', 'TRX', 'in', 0);
        opt.addLink('EY', 'bk', 'TRY', 'in', 0);
        opt.addProbeIn('TRX_DC', 'TRX', 'in', 0, 0);
        opt.addProbeIn('TRY_DC', 'TRX', 'in', 0, 0);
    end

    % Set output matrix for homodyne readout
    nA = opt.getProbeNum('OMC_SUM');
    nB = opt.getProbeNum('OMC_DIFF');

    opt.mProbeOut = eye(opt.Nprobe);      % start with identity matrix
    opt.mProbeOut(nA, nA) = 1;
    opt.mProbeOut(nB, nA) = -1;  % add B to A
    opt.mProbeOut(nA, nB) = 1; % subtract A from B
    opt.mProbeOut(nB, nB) = 1;
end


function addAllReadout(opt, par, evalName, evalPort, probeName, includeRF)
    f1 = par.Mod.f1;
    f2 = par.Mod.f2;
    opt.addSink(probeName);
    opt.addLink(evalName, evalPort, probeName, 'in', 0);
    if includeRF == 1
        freqs = [f1, f2];
        names = {'f1', 'f2'};
    elseif includeRF == 2
        freqs = [f1, f2, 3*f1, 3*f2, f2 - f1, f2 + f1];
        names = {'f1', 'f2', '3f1', '3f2', 'M', 'P'};
    end
    for ii=1:length(names)
        phases(ii) = par.phi.([probeName, names{ii}]);
    end
    probeData = [freqs; phases]';
    opt.addReadout(probeName, probeData, names);
end
