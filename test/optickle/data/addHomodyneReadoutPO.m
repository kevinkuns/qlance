function addHomodyneReadoutPO(opt, name, phase, qe)
  opt.addMirror([name, '_BS'], 45, 0, 0.5);
  opt.addSink([name, '_attnA'], 1 - qe);
  opt.addSink([name, '_attnB'], 1 - qe);
  opt.addSink([name, '_A'], 1);
  opt.addSink([name, '_B'], 1);

  opt.addMirror([name, '_LOphase'], 0, 0, 0);
  opt.addLink([name, '_LOphase'], 'fr', [name, '_BS'], 'bk', 0);
  opt.addLink([name, '_BS'], 'fr', [name, '_attnA'], 'in', 0);
  opt.addLink([name, '_attnA'], 'out', [name, '_A'], 'in', 0);
  opt.addLink([name, '_BS'], 'bk', [name, '_attnB'], 'in', 0);
  opt.addLink([name, '_attnB'], 'out', [name, '_B'], 'in', 0);
  opt.addProbeIn([name, '_SUM'], [name, '_A'], 'in', 0, 0);
  opt.addProbeIn([name, '_DIFF'], [name, '_B'], 'in', 0, 0);
