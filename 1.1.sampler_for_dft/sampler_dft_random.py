import os
import random
#os.environ['PATH'] = '/home/phy_cmp/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/phy_cmp/program/enumlib/src:/home/phy_cmp/program/enumlib/aux_src'
import numpy as np
import pandas as pd
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure, Composition
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator
import pymatgen.io.vasp.inputs as inputs
from pymatgen.core.periodic_table import Element
import pickle
from matminer.featurizers.composition import ElementProperty

paws = {
    'H': 'H', 'He': 'He', 'Li': 'Li_sv', 'Be': 'Be', 'B': 'B', 'C': 'C', 'N': 'N', 'O': 'O', 'F': 'F', 'Ne': 'Ne',
    'Na': 'Na_pv', 'Mg': 'Mg', 'Al': 'Al', 'Si': 'Si', 'P': 'P', 'S': 'S', 'Cl': 'Cl', 'Ar': 'Ar', 'K': 'K_sv',
    'Ca': 'Ca_sv', 'Sc': 'Sc_sv', 'Ti': 'Ti_sv', 'V': 'V_sv', 'Cr': 'Cr_pv', 'Mn': 'Mn_pv', 'Fe': 'Fe', 'Co': 'Co',
    'Ni': 'Ni', 'Cu': 'Cu', 'Zn': 'Zn', 'Ga': 'Ga_d', 'Ge': 'Ge_d', 'As': 'As', 'Se': 'Se', 'Br': 'Br', 'Kr': 'Kr',
    'Rb': 'Rb_sv', 'Sr': 'Sr_sv', 'Y': 'Y_sv', 'Zr': 'Zr_sv', 'Nb': 'Nb_sv', 'Mo': 'Mo_sv', 'Tc': 'Tc_pv',
    'Ru': 'Ru_pv',
    'Rh': 'Rh_pv', 'Pd': 'Pd', 'Ag': 'Ag', 'Cd': 'Cd', 'In': 'In_d', 'Sn': 'Sn_d', 'Sb': 'Sb', 'Te': 'Te', 'I': 'I',
    'Xe': 'Xe', 'Cs': 'Cs_sv', 'Ba': 'Ba_sv', 'La': 'La', 'Ce': 'Ce', 'Pr': 'Pr_3', 'Nd': 'Nd_3', 'Pm': 'Pm_3',
    'Sm': 'Sm_3', 'Eu': 'Eu_2', 'Gd': 'Gd_3', 'Tb': 'Tb_3', 'Dy': 'Dy_3', 'Ho': 'Ho_3', 'Er': 'Er_3', 'Tm': 'Tm_3',
    'Yb': 'Yb_2', 'Lu': 'Lu_3', 'Hf': 'Hf_pv', 'Ta': 'Ta_pv', 'W': 'W_sv', 'Re': 'Re', 'Os': 'Os', 'Ir': 'Ir',
    'Pt': 'Pt', 'Au': 'Au', 'Hg': 'Hg', 'Tl': 'Tl_d', 'Pb': 'Pb_d', 'Bi': 'Bi_d', 'Po': 'Po_d', 'At': 'At',
    'Rn': 'Rn', 'Fr': 'Fr_sv', 'Ra': 'Ra_sv', 'Ac': 'Ac', 'Th': 'Th', 'Pa': 'Pa', 'U': 'U', 'Np': 'Np', 'Pu': 'Pu',
    'Am': 'Am', 'Cm': 'Cm'
}


def generate_random_compositions(element_list, num_compositions, num_atoms=16):
    compositions = []

    for _ in range(num_compositions):
        # Randomly select 3 or 4 elements from the element_list
        num_elements = random.choice([3, 4])
        selected_elements = random.sample(element_list, num_elements)

        # Generate random stoichiometric coefficients that sum up to num_atoms
        stoichiometric_coeffs = np.zeros(num_elements, dtype=int)

        # Randomly assign at least one atom to each element
        stoichiometric_coeffs[:num_elements] = 1

        remaining_atoms = num_atoms - num_elements

        # Distribute the remaining atoms randomly among the elements
        for _ in range(remaining_atoms):
            idx = np.random.randint(num_elements)
            stoichiometric_coeffs[idx] += 1

        composition = {Element(selected_elements[i]): stoichiometric_coeffs[i] for i in range(num_elements)}
        compositions.append(Composition(composition))

    return compositions

def replace_species(structure, composition):
    total_sites = len(structure.sites)
    site_counts = {el: int(frac) for el, frac in composition.element_composition.items()}

    while sum(site_counts.values()) < total_sites:
        for el in site_counts:
            if sum(site_counts.values()) < total_sites:
                site_counts[el] += 1
            else:
                break

    species_list = []
    for el, count in site_counts.items():
        species_list.extend([el] * count)

    for i in range(total_sites):
        structure.replace(i, species_list[i])

    return structure

def shannon_entropy(comp):
    frac_comp = comp.element_composition.get_el_amt_dict()
    entropy = -np.sum([frac * np.log(frac) for frac in frac_comp.values()])
    return entropy

def get_mag_ordering(structure):
    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    symmetrized_structure = sga.get_symmetrized_structure()

    kwargs = {'check_ordered_symmetry': False}
    try:
        enumerator = MagneticStructureEnumerator(symmetrized_structure, default_magmoms={"Mn": 3.0, "Co": 3.0},
                                                 transformation_kwargs=kwargs)
        magnetic_structures = list(enumerator.ordered_structures)
        return magnetic_structures
    except ValueError:
        return [symmetrized_structure]



def get_magmom_str(structure):
    magmom = ""
    for site in structure:
        spin = site.species.elements[0].spin if hasattr(site.species.elements[0], 'spin') else 0
        magmom += "{} ".format(spin)
    return magmom

def write_input_vasp(composition, folder_path):
    for symmetry_group in [216, 225]:
        symm_folder_path = os.path.join('samples', str(symmetry_group), folder_path)
        os.makedirs(symm_folder_path, exist_ok=True)

        try:
            structure = Structure.from_file(f'initial_poscars/POSCAR_{symmetry_group}')
        except FileNotFoundError:
            print(f"File not found: initial_poscars/POSCAR_{symmetry_group}")
            continue

        structure = replace_species(structure, composition)
        volume, tetragonal_ratio = calculate_fitness([composition])

        structure.scale_lattice(volume)
        magnetic_orderings = get_mag_ordering(structure)
        for i, ordering in enumerate(magnetic_orderings):
            for j, ca in enumerate([1, tetragonal_ratio]):
                D = np.array([
                    [ca ** (-1 / 3), 0, 0],
                    [0, ca ** (-1 / 3), 0],
                    [0, 0, ca ** (2 / 3)]
                ], dtype=object)

                ordering.lattice = np.dot(ordering.lattice.matrix, D)

                poscar = inputs.Poscar(ordering)
                potentials = [paws[s.symbol] for s in structure.types_of_species]
                # potcar = inputs.Potcar(symbols=potentials, functional="PBE_54")
                kpoints = inputs.Kpoints.automatic_density(structure, 5000)
                incar = inputs.Incar.from_file('INCAR')
                incar['MAGMOM'] = get_magmom_str(ordering)

                vasp_input = inputs.VaspInput(incar, kpoints, poscar, potcar=None)

                if j == 0:
                    save_path = os.path.join(symm_folder_path, 'austenite', str(i))
                else:
                    save_path = os.path.join(symm_folder_path, 'martensite', str(i))
                vasp_input.write_input(save_path)


def calculate_fitness(population):
    df = pd.DataFrame({'composition': population})

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")

    df['shannon_entropy'] = df['composition'].map(lambda x: shannon_entropy(x))

    volume_model = pickle.load(open("model_volume.pickle", "rb"))
    df['volume'] = volume_model.predict(df.drop(['composition', 'shannon_entropy'], axis=1))

    tetragonal_ratio_model = pickle.load(open("model_tetr.pickle", "rb"))
    df['tetragonal_ratio'] = tetragonal_ratio_model.predict(df.drop(['composition'], axis=1))

    return df['volume'].values[0], df['tetragonal_ratio'].values[0]


if __name__ == '__main__':
    element_list = [
    # 3d Transition Metals
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    # 4d Transition Metals
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    # 5d Transition Metals
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
    'Ir', 'Pt', 'Au', 'Hg']

    random_population = generate_random_compositions(element_list, 10)
    for i, element in enumerate(random_population):
        print(f'{element} {i+1}/{len(random_population)}')
        write_input_vasp(element, element.formula.replace(' ', ''))