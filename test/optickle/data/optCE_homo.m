% Create a CE optickl model
% opt = optCE(par, rfOption)
% par a struct with parameters
% includeRF: how many sidebands to include
%    0: Only include the carrier
%    1: Include the carrier and f1 and f2 sidebands
%    2: Include the carrier and the intermodulation sidebands as well


function opt = optCE_homo(par, includeRF)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up RF frequencies
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if includeRF == 0
        % Only the carrier
        vFrf = 0;

    elseif (includeRF == 1) || (includeRF == 2)
        % Include RF sidebands
        f1 = par.Mod.f1;
        f2 = par.Mod.f2;
        nMod1 = par.Mod.nMod1;
        nMod2 = par.Mod.nMod2;

        if includeRF == 1
            % Only the f1 and f2 sidebands
            vFrf = [-f2, -f1, 0, f1, f2]';

        elseif includeRF == 2
            % Include the intermodulation sidebands as well
            % frequencies of the RF sidebands:
            vFrf1 = f1 * (-nMod1:nMod1');
            vFrf2 = f2 * (-nMod2:nMod2');

            % frequencies of the intermodulation sidebands:
            for ii=1:length(vFrf1)
                 for jj=1:length(vFrf2)
                     index = jj + (ii - 1)*length(vFrf1);
                     vFrf(index) = vFrf1(ii) + vFrf1(jj);
                 end
             end
         end
        vFrf = sort(vFrf);
    else
        error(['Unrecognized option for sideband number', int2str(includeRF)]);
    end

    opt = Optickle(vFrf, par.Laser.Wavelength);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Add Optics
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    listBS = {'BS', 'PR2', 'SR2'};
    listMirror = {'PR','SR', 'IX', 'IY', 'EX', 'EY', 'OMCa', 'OMCb', ...
                  'FCI', 'FCE', 'PO_AS', 'SR2'};

    for n = 1:length(listBS)
        name = listBS{n};
        p = par.(name);
        opt.addBeamSplitter(name, p.aoi, p.Chr, p.Thr, p.Lhr, p.Rar, p.Lmd);
        opt = setPosOffset(opt, name, p.pos);
    end

    for n = 1:length(listMirror)
        name = listMirror{n};
        p = par.(name);
        opt = addMirror(opt, name, p.aoi, p.Chr, p.Thr, p.Lhr, p.Rar, p.Lmd);
        opt = setPosOffset(opt, name, p.pos);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Mechanical
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Set transfer functions
    dampRes = [-0.1 + 1i, -0.1 - 1i];
    poles = par.w * dampRes;
    listOptics = {'IX', 'EX', 'IY', 'EY', 'PR', 'SR', 'BS'};
    for n=1:length(listOptics)
        name = listOptics{n};
        p = par.(name);
        opt.setMechTF(name, zpk([], poles, 1/p.Mass));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Main laser
    vArf = zeros(size(vFrf));
    vArf(find(vFrf == 0, 1)) = sqrt(par.Laser.Pin);
    opt.addSource('Laser', vArf);

    % Modulators for amplitude and phase noise
    opt.addModulator('AM', 1);
    opt.addModulator('PM', 1j);

    % RF modulators
    if includeRF > 0
        opt.addRFmodulator('Mod1', par.Mod.f1, par.Mod.g1*1j);
        opt.addRFmodulator('Mod2', par.Mod.f2, par.Mod.g2*1j);
    end

    % Links
    opt.addLink('Laser', 'out', 'AM', 'in', 0);
    opt.addLink('AM', 'out', 'PM', 'in', 0);
    if includeRF > 0
        opt.addLink('PM', 'out', 'Mod1', 'in', 5);
        opt.addLink('Mod1', 'out', 'Mod2', 'in', 0);
        opt.addLink('Mod2', 'out', 'PR', 'bk', 0.2);
    elseif includeRF == 0
        opt.addLink('PM', 'out', 'PR', 'bk', 0.2);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Arms
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % X Arm
    opt.addLink('BS', 'bkA', 'IX', 'bk', par.Length.IX);
    opt.addLink('IX', 'fr', 'EX', 'fr', par.Length.EX);
    opt.addLink('EX', 'fr', 'IX', 'fr', par.Length.EX);
    opt.addLink('IX', 'bk', 'BS', 'bkB', par.Length.IX);
    opt.setCavityBasis('IX', 'EX');

    % Y arm
    opt.addLink('BS', 'frA', 'IY', 'bk', par.Length.IY);
    opt.addLink('IY', 'fr', 'EY', 'fr', par.Length.EY);
    opt.addLink('EY', 'fr', 'IY', 'fr', par.Length.EY);
    opt.addLink('IY', 'bk', 'BS', 'frB', par.Length.IY);
    opt.setCavityBasis('IY', 'EY');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Recycling cavities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % PRC
    opt.addLink('PR', 'fr', 'PR2', 'frA', par.Length.PR_PR2);
    opt.addLink('PR2', 'frA', 'BS', 'frA', par.Length.PR2_BS);
    opt.addLink('BS', 'frB', 'PR2', 'frB', par.Length.PR2_BS);
    opt.addLink('PR2', 'frB', 'PR', 'fr', par.Length.PR_PR2);

    % SRC
    opt.addLink('BS', 'bkB', 'SR2', 'frA', par.Length.SR2_BS);
    opt.addLink('SR2', 'frA', 'SR', 'fr', par.Length.SR_SR2);
    opt.addLink('SR', 'fr', 'SR2', 'frB', par.Length.SR_SR2);
    opt.addLink('SR2', 'frB', 'BS', 'bkA', par.Length.SR2_BS);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % AS port
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % AS pickoff
    opt.addLink('SR', 'bk', 'PO_AS', 'fr', 10);
    opt.addLink('PO_AS', 'bk', 'OMCa', 'bk', 4);

    % OMC
    opt.addLink('OMCa', 'fr', 'OMCb', 'fr', par.Length.OMC);
    opt.addLink('OMCb', 'fr', 'OMCa', 'fr', par.Length.OMC);
    opt.setCavityBasis('OMCa', 'OMCb');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Squeezer
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    p = par.('Sqz');
    opt.addSqueezer('SQZ', par.Laser.Wavelength, 0, Optickle.polS, ...
                    p.sqAng, p.sqdB, p.antidB, 0);

    % Filter cavity
    opt.addLink('FCI', 'fr', 'FCE', 'fr', p.fcLength);
    opt.addLink('FCE', 'fr', 'FCI', 'fr', p.fcLength);
    opt.setCavityBasis('FCI', 'FCE');

    % Set FC detuning
    % dL/lambda = df/(2fsr)
    % fsr = scc.c / (2*p.fcLength);
    dLfc = p.fcDetuning/(2*p.fcFSR) * par.Laser.Wavelength;
    opt.setPosOffset('FCE', dLfc/2);

    % Links
    % FIXME: add injection loss
    opt.addLink('SQZ', 'out', 'FCI', 'bk', 0.1);
    opt.addLink('FCI', 'bk', 'SR', 'bk', 0.1);
end
