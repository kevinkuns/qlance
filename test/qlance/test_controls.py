import qlance.controls as ctrl


def test_composite_dof():
    dof = ctrl.DegreeOfFreedom({'MC2.pos': 1, 'Laser': -1}, doftype='freq')
    assert dof.drives['MC2.pos'] == 1
    assert dof.drives['Laser.freq'] == -1
    assert dof.doftype == 'freq'
    component_dofs = {}
    coeffs = {}
    for cdof, cc in dof.dofs():
        component_dofs[cdof.name] = cdof
        coeffs[cdof.name] = cc
    assert component_dofs['MC2'].name == 'MC2'
    assert component_dofs['MC2'].doftype == 'pos'
    assert component_dofs['MC2'].drives['MC2.pos'] == 1
    assert coeffs['MC2'] == 1
    assert component_dofs['Laser'].name == 'Laser'
    assert component_dofs['Laser'].doftype == 'freq'
    assert component_dofs['Laser'].drives['Laser.freq'] == 1
    assert coeffs['Laser'] == -1
