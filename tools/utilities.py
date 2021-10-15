#  Adapted from https://github.com/libAtoms/silicon-testing-framework


import os
from io import StringIO

from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter, voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress


from ase import Atoms
import numpy as np


def read_bulk_reference(model_name, test_name):
    log_file = 'model-{0}-test-{1}.txt'.format(model_name, test_name)
    log_lines = open(os.path.join('..', log_file), 'r').readlines()
    start_idx = log_lines.index('relaxed bulk\n')
    n_atoms = int(log_lines[start_idx+1])
    xyz_lines = log_lines[start_idx+1:start_idx+n_atoms+3]
    fh = StringIO.StringIO(''.join(xyz_lines))
    bulk = read(fh, format='extxyz')
    return bulk

def relax_atoms(atoms, tol=1e-3, method='lbfgs_precon', max_steps=1000, traj_file=None, **kwargs):
    if method.startswith('lbfgs') or method == 'fire' or method == 'cg_n':
        if method == 'lbfgs_ASE':
            from ase.optimize import LBFGS
            opt = LBFGS(atoms, **kwargs)
        from ase.optimize.precon.precon import Exp
        from ase.optimize.precon.lbfgs import PreconLBFGS
        precon = None
        if method.endswith('precon'):
            precon = Exp(3.0, reinitialize=True)

        if method.startswith('lbfgs'):
            opt = PreconLBFGS(atoms, precon=precon, **kwargs)
        else:
            opt = FIRE(atoms, **kwargs)
        if traj_file is not None and method != 'cg_n':
            traj = open(traj_file, 'w')
            def write_trajectory():
                write(traj, atoms, format='extxyz')
            opt.attach(write_trajectory)
        opt.run(tol, max_steps)
        try:
            traj.close()
        except:
            pass
    else:
        raise ValueError('unknown method %s!' % method)

    return atoms

from .symmetrize import prep,forces, stress
from ase.calculators.calculator import Calculator
class SymmetrizedCalculator(Calculator):
   implemented_properties = ['energy','forces','stress','free_energy']
   def __init__(self, calc, atoms, *args, **kwargs):
      Calculator.__init__(self, *args, **kwargs)
      self.calc = calc
      (self.rotations, self.translations, self.symm_map) = prep(atoms, symprec=0.01)

   def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            self.results = {}
        if 'energy' in properties and 'energy' not in self.results:
            self.results['energy'] = self.calc.get_potential_energy(atoms)
            self.results['free_energy'] = self.results['energy']
        if 'forces' in properties and 'forces' not in self.results:
            raw_forces = self.calc.get_forces(atoms)
            self.results['forces'] = forces(atoms.get_cell(), atoms.get_reciprocal_cell().T, raw_forces,
                self.rotations, self.translations, self.symm_map)
        if 'stress' in properties and 'stress' not in self.results:
            raw_stress = self.calc.get_stress(atoms)
            if len(raw_stress) == 6: # Voigt
                raw_stress = voigt_6_to_full_3x3_stress(raw_stress)
            symmetrized_stress = stress(atoms.get_cell(), atoms.get_reciprocal_cell().T, raw_stress, self.rotations)
            self.results['stress'] = full_3x3_to_voigt_6_stress(symmetrized_stress)


def relax_atoms_cell(atoms, tol=1e-3, stol=None, method='lbfgs_precon', max_steps=100, mask=None, traj_file=None,
                     hydrostatic_strain=False, constant_volume=False, precon_apply_positions=True,
                     precon_apply_cell=True, symmetrize = False, **kwargs):
    if symmetrize:
        calc = atoms.get_calculator()
        scalc = SymmetrizedCalculator(calc, atoms)
        atoms.set_calculator(scalc)

    if method != 'cg_n':
        atoms = UnitCellFilter(atoms, mask=mask,
                               hydrostatic_strain=hydrostatic_strain,
                               constant_volume=constant_volume)
    if method.startswith('lbfgs') or method == 'fire' or method == 'cg_n':
        from ase.optimize.precon.precon import Exp
        from ase.optimize.precon.lbfgs import PreconLBFGS
        precon = None
        if method.endswith('precon'):
            precon = Exp(3.0, apply_positions=precon_apply_positions,
                            apply_cell=precon_apply_cell, reinitialize=True)
        if method.startswith('lbfgs'):
            opt = PreconLBFGS(atoms, precon=precon, **kwargs)
        else:
            opt = FIRE(atoms, **kwargs)
        if traj_file is not None:
            traj = open(traj_file, 'w')
            def write_trajectory():
                try:
                    write(traj, atoms.atoms, format='extxyz')
                except:
                    write(traj, atoms, format='extxyz')
            opt.attach(write_trajectory)
        if method != 'cg_n' and isinstance(opt, PreconLBFGS):
            opt.run(tol, max_steps, smax=stol)
        else:
            opt.run(tol, max_steps)
        if traj_file is not None:
            traj.close()
    else:
        raise ValueError('unknown method %s!' % method)

    if isinstance(atoms, UnitCellFilter):
        return atoms.atoms
    else:
        return atoms


def evaluate(atoms, do_energy=True, do_forces=True, do_stress=True):
    energy = None
    if do_energy:
        energy = atoms.get_potential_energy()

    forces = None
    if do_forces:
        forces = atoms.get_forces()

    stress = None
    if do_stress:
        stress = atoms.get_stress()

    spc = SinglePointCalculator(atoms,
                                energy=energy,
                                forces=forces,
                                stress=stress)
    atoms.set_calculator(spc)
    return atoms

