import numpy as np
import pandas as pd
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.ga.data import PrepareDB, DataConnection
from ase.ga import get_raw_score
from ase.ga.startgenerator import StartGenerator
from ga_bulk_relax import relax
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.ga.offspring_creator import OperationSelector
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.population import Population
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import StrainMutation
from ase.ga.utilities import CellBounds, closest_distances_generator
from ase.io import write
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase import Atoms
from ase.data import atomic_numbers

from chgnet.model import StructOptimizer

from AlphaCrystal.cryspnet.cryspnet.utils import FeatureGenerator, load_input, dump_output, group_outputs, topkacc
from AlphaCrystal.cryspnet.cryspnet.models import load_Bravais_models, load_Lattice_models, load_SpaceGroup_models
from AlphaCrystal.cryspnet.cryspnet.config import *

import warnings
import os
warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from pymatgen.core.composition import Composition, Element
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-i", "--input", default=None, type=str, required=True,
                        help="The input formula")
    parser.add_argument("-o", "--output", default=None, type=str, required=True,
                        help="The output directory where predictions  will be written.")
    parser.add_argument("--default_v", default=300, type=int,
                        help="default target unit cell volume")
    parser.add_argument("--topn_bravais", default=1, type=int,
                        help="The top-n Bravais Lattice the user want to pre \
                        serve. The space group and lattice parameter would \
                        be predicted for each top-n Bravais Lattice"
            )
    parser.add_argument("--topn_spacegroup", default=1, type=int,
                        help="The top-n Space Group the user want to pre \
                        serve."
            )
    parser.add_argument("--population_size", default=20, type=int,
                        help="population size for genetic algorithm")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Batch size per GPU/CPU for prediction.")
    parser.add_argument("--use_cpu", action='store_true', default=True,
                        help="Avoid using CUDA when available")
    parser.add_argument("--n_ensembler", default=5, type=int,
                        help="number of ensembler for Bravais Lattice Prediction.")

    args = parser.parse_args()

    #define parameters
    which = "oxide"
    batch_size = args.batch_size
    use_cpu = args.use_cpu
    formula = args.input
    topn_bravais = args.topn_bravais
    topn_spacegroup= args.topn_spacegroup
    N = 20
    default_bounds={'phi': [20, 160], 'chi': [20, 160],'psi': [20, 160], 
                    'a': [2, 60], 'b': [2, 60], 'c': [2, 60]}
    default_v = args.default_v
    db_file_name = formula+"_ga.db"

    #make directory for output if not exist
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    #load Cryspnet models and predict UC
    BE = load_Bravais_models(
        n_ensembler = args.n_ensembler,
        which = which,
        batch_size = batch_size,
        cpu=use_cpu
    )
    LPB = load_Lattice_models(batch_size = batch_size, cpu=use_cpu)
    SGB = load_SpaceGroup_models(batch_size = batch_size, cpu=use_cpu)
    featurizer = FeatureGenerator()
    data = pd.DataFrame({"formula": [formula]})
    ext_magpie = featurizer.generate(data)

    bravais_probs, bravais = BE.predicts(ext_magpie, topn_bravais=topn_bravais)
    lattices = []
    spacegroups = []
    spacegroups_probs = []

    for i in range(topn_bravais):
        ext_magpie["Bravais"] = bravais[:, i]
        lattices.append(LPB.predicts(ext_magpie))
        sg_prob, sg = SGB.predicts(ext_magpie, topn_spacegroup=topn_spacegroup)
        spacegroups.append(sg)
        spacegroups_probs.append(sg_prob)

    out = group_outputs(bravais, bravais_probs, spacegroups, spacegroups_probs, lattices, data)


    #prepare starting generator for GA
    def formula_to_blocks(formula):
        comp = Composition(formula).as_dict()
        block = []
        for elem in comp:
            block += [elem] * int(comp[elem])
        return block

    blocks = formula_to_blocks(formula)

    def lattice_to_sg(lattice, blocks, pm=10):
        if lattice['Bravais prob'].values[0] >= 0.5:
            a = [lattice.a.values[0]-pm, lattice.a.values[0]+pm]
            b = [lattice.b.values[0]-pm, lattice.b.values[0]+pm]
            c = [lattice.c.values[0]-pm, lattice.c.values[0]+pm]
            phi = [lattice.alpha.values[0]-pm, lattice.alpha.values[0]+pm]
            chi = [lattice.beta.values[0]-pm, lattice.beta.values[0]+pm]
            psi = [lattice.gamma.values[0]-pm, lattice.gamma.values[0]+pm]
            v = lattice.v.values[0]
            cellbounds = CellBounds(bounds={'phi': phi, 'chi': chi, 'psi': psi, 
                                    'a': a, 'b': b, 'c': c})
        else:
            cellbounds = CellBounds(bounds=default_bounds)
            v = default_v
        slab = Atoms('', pbc=True)
        blmin = closest_distances_generator(atom_numbers=[atomic_numbers[block] for block in blocks],ratio_of_covalent_radii=0.5)
        sg = StartGenerator(slab, blocks, blmin, box_volume=v,
                        number_of_variable_cell_vectors=3,
                        cellbounds=cellbounds)
        return sg, cellbounds, blmin

    sg, cellbounds, blmin = lattice_to_sg(out['Top-1 Bravais'], blocks)

    #prepare database
    if not os.path.exists(db_file_name):
        d = PrepareDB(db_file_name=db_file_name,
                    stoichiometry=blocks)
        for i in tqdm(range(N)):
            a = sg.get_new_candidate()
            d.add_unrelaxed_candidate(a)

    da = DataConnection(db_file_name)
    while da.get_number_of_unrelaxed_candidates() < N:
        a = sg.get_new_candidate()
        da.add_unrelaxed_candidate(a)
    slab = da.get_slab()
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_top = len(atom_numbers_to_optimize)

    #prepare operators
    comp = OFPComparator(n_top=n_top, dE=1.0,
                        cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                        pbc=[True, True, True], sigma=0.05, nsigma=4,
                        recalculate=False)

    pairing = CutAndSplicePairing(slab, n_top, blmin, p1=1., p2=0., minfrac=0.15,
                                number_of_variable_cell_vectors=3,
                                cellbounds=cellbounds, use_tags=False)

    strainmut = StrainMutation(blmin, stddev=0.7, cellbounds=cellbounds,
                            number_of_variable_cell_vectors=3,
                            use_tags=False)

    blmin_soft = closest_distances_generator([atomic_numbers[block] for block in blocks], 0.1)
    softmut = SoftMutation(blmin_soft, bounds=[2., 5.], use_tags=False)

    operators = OperationSelector([4., 3., 3.],
                                [pairing, softmut, strainmut])

    #CHGNet relaxer
    relaxer = StructOptimizer()

    #relax initial population
    i = 0
    while da.get_number_of_unrelaxed_candidates() > 0:
        print("relaxing candidate no." + str(i))
        a = da.get_an_unrelaxed_candidate()

        relax(a, relaxer, cellbounds=cellbounds, verbose=True)
        da.add_relaxed_step(a)

        cell = a.get_cell()
        if not cellbounds.is_within_bounds(cell):
            print("Killed" + str(a.info['confid']))
            da.kill_candidate(a.info['confid'])
        
        i += 1

    # Initialize the population
    population_size = 20
    population = Population(data_connection=da,
                            population_size=population_size,
                            comparator=comp,
                            logfile='log.txt',
                            use_extinct=True)

    # Update the scaling volume used in some operators
    # based on a number of the best candidates
    current_pop = population.get_current_population()
    strainmut.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)
    pairing.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)

    # Test n_to_test new candidates; in this example we need
    # only few GA iterations as the global minimum (FCC Ag)
    # is very easily found (typically already after relaxation
    # of the initial random structures).
    n_to_test = 50

    for step in range(n_to_test):
        print(f'Now starting configuration number {step}')

        # Create a new candidate
        a3 = None
        while a3 is None:
            a1, a2 = population.get_two_candidates()
            a3, desc = operators.get_new_individual([a1, a2])

        # Save the unrelaxed candidate
        da.add_unrelaxed_candidate(a3, description=desc)

        # Relax the new candidate and save it
        print("relaxing")
        relax(a3, relaxer, cellbounds=cellbounds, verbose=True)
        da.add_relaxed_step(a3)

        # If the relaxation has changed the cell parameters
        # beyond the bounds we disregard it in the population
        cell = a3.get_cell()
        if not cellbounds.is_within_bounds(cell):
            da.kill_candidate(a3.info['confid'])

        # Update the population
        population.update()

        if step % 10 == 0:
            # Update the scaling volumes of the strain mutation
            # and the pairing operator based on the current
            # best structures contained in the population
            current_pop = population.get_current_population()
            strainmut.update_scaling_volume(current_pop, w_adapt=0.5,
                                            n_adapt=4)
            pairing.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)
            write(out_dir+'/current_population.traj', current_pop)

    print('GA finished after step %d' % step)
    hiscore = get_raw_score(current_pop[0])
    print('Highest raw score = %8.4f eV' % hiscore)

    all_candidates = da.get_all_relaxed_candidates()
    write(out_dir+'/all_candidates.traj', all_candidates)

    current_pop = population.get_current_population()
    write(out_dir+'/current_population.traj', current_pop)

if __name__ == "__main__":
    main()
