from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga import set_raw_score
from chgnet.model import StructOptimizer


def finalize(atoms, energy, forces, stress):
    # Finalizes the atoms by attaching a SinglePointCalculator
    # and setting the raw score as the negative of the total energy
    atoms.wrap()
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                 stress=stress)
    atoms.calc = calc
    raw_score = -energy
    set_raw_score(atoms, raw_score)
    return atoms


def relax(atoms, relaxer:StructOptimizer, cellbounds=None, verbose=False):
    # Performs a variable-cell relaxation of the structure with chgnet
    result = relaxer.relax(atoms, verbose=verbose)
    atoms = result['final_structure'].to_ase_atoms()
    e = result['trajectory'].energies[-1].astype(float).item()
    f = result['trajectory'].forces[-1].astype(float)
    s = result['trajectory'].stresses[-1].astype(float)
    return finalize(atoms, energy=e, forces=f, stress=s)
