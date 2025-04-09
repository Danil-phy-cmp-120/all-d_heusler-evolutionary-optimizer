import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

import matplotlib.pyplot as plt
import mpltern
import numpy as np
import random
import pandas as pd
import pickle
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from scipy.spatial.distance import euclidean
from matplotlib.colors import ListedColormap
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
import json


def generate_random_composition(element_list):
    nums = [random.random() for _ in range(len(element_list) - 1)]
    nums.append(0)
    nums.append(1)
    nums.sort()
    composition = {el: nums[i + 1] - nums[i] for i, el in enumerate(element_list)}
    return Composition(composition)


def generate_uniform_composition(element_list, num_points):
    compositions = []
    for i in range(num_points + 1):
        for j in range(num_points - i + 1):
            k = num_points - i - j
            composition = {
                element_list[0]: round(i / num_points, 2),
                element_list[1]: round(j / num_points, 2),
                element_list[2]: round(k / num_points, 2)
            }
            compositions.append(Composition(composition))
    return compositions


def generate_subpopulations_within_range(subpopulation_count, subpopulation_size, range_limit, element_list):
    subpopulations = []

    for _ in range(subpopulation_count):
        # Generate a random center composition
        center = generate_random_composition(element_list)
        center_values = np.array(list(center.values()))

        subpopulation = []
        for _ in range(subpopulation_size):
            while True:
                # Generate random perturbations within the range limit
                perturbations = np.random.uniform(-range_limit, range_limit, len(element_list))
                individual_values = center_values + perturbations

                # Ensure values are non-negative
                individual_values = np.clip(individual_values, 1e-6, None)

                # Normalize to ensure the sum of the concentrations equals 1
                if sum(individual_values) > 0:
                    individual_values /= sum(individual_values)

                # Check if the generated individual is within valid range
                if np.all(individual_values >= 1e-6) and np.all(individual_values <= 1):
                    break

            individual = {el: individual_values[i] for i, el in enumerate(element_list)}
            subpopulation.append(Composition(individual))

        subpopulations.append(subpopulation)

    return subpopulations

def shannon_entropy(comp):
    frac_comp = comp.fractional_composition.get_el_amt_dict()
    entropy = -np.sum([frac * np.log(frac) for frac in frac_comp.values()])
    return entropy


def calculate_fitness(population):
    df = pd.DataFrame({'composition': population})

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")

    df['shannon_entropy'] = df['composition'].map(lambda x: shannon_entropy(x))

    volume_model = pickle.load(open("model_volume.pickle", "rb"))
    df['volume'] = volume_model.predict(df.drop(['composition', 'shannon_entropy'], axis=1))

    tetragonal_ratio_model = pickle.load(open("model_tetr.pickle", "rb"))
    df['tetragonal_ratio'] = tetragonal_ratio_model.predict(df.drop(['composition'], axis=1))

    magnetization_aust_model = pickle.load(open("model_mag_aust.pickle", "rb"))
    df['magnetization_aust'] = magnetization_aust_model.predict(df.drop(['composition', 'tetragonal_ratio'], axis=1))

    magnetization_mart_model = pickle.load(open("model_mag_mart.pickle", "rb"))
    df['magnetization_mart'] = magnetization_mart_model.predict(df.drop(['composition', 'magnetization_aust'], axis=1))

    return abs(df['magnetization_aust'].values - df['magnetization_mart'].values)


def fitness_sharing(fitness_scores, subpopulation, sharing_radius):
    shared_fitness = []
    for i, fitness in enumerate(fitness_scores):
        niche_count = 1
        for j in range(len(subpopulation)):
            if i != j and np.linalg.norm(np.array(list(subpopulation[i].values())) - np.array(list(subpopulation[j].values()))) < sharing_radius:
                niche_count += 1
        shared_fitness.append(fitness / niche_count)
    return shared_fitness



def arithmetic_crossover(parent1, parent2):
    child = {}
    for element in parent1.elements:
        child[element] = (parent1[element] + parent2[element]) / 2
    return Composition(child)


def mutation(composition, mutation_rate=0.1):
    elements = list(composition.keys())
    values = list(composition.values())

    if np.random.rand() < mutation_rate:
        idx1, idx2 = random.sample(range(len(elements)), 2)

        # Decrease the range for mutation_amount to reduce mutation impact
        mutation_range = min(values[idx1], values[idx2], 0.02)
        mutation_amount = np.random.uniform(-mutation_range, mutation_range)

        values[idx1] += mutation_amount
        values[idx2] -= mutation_amount

        # Ensure values stay within the bounds [0, 1]
        values[idx1] = max(0, min(1, values[idx1]))
        values[idx2] = max(0, min(1, values[idx2]))

        # Normalize values so they sum to 1
        total = sum(values)
        values = [v / total for v in values]

    mutated_composition = dict(zip(elements, values))
    return Composition(mutated_composition)


def plot_population_distribution(subpopulations, generation, comp_dict_uniform, fitness_uniform, generation_number):
    elements = list(subpopulations[0][0].elements)

    # Initialize the ternary plot
    fig = plt.figure(figsize=(10, 8))
    tax = fig.add_subplot(projection='ternary')

    # Define colors for each subpopulation
    colormap = plt.get_cmap('tab10')
    colors = [mcolors.to_hex(colormap(i / len(subpopulations))) for i in range(len(subpopulations))]

    # Define arrow style
    arrowstyle = ArrowStyle('simple', head_length=10, head_width=5)
    kwargs_arrow = {
        'transform': tax.transAxes,  # Used with ``ax.transAxesProjection``
        'arrowstyle': arrowstyle,
        'linewidth': 1,
        'clip_on': False,  # To plot arrows outside triangle
        'zorder': -10,
    }

    # Start of arrows in barycentric coordinates.
    ta = np.array([0.0, -0.1, 1.1])
    la = np.array([1.1, 0.0, -0.1])
    ra = np.array([-0.1, 1.1, 0.0])

    # End of arrows in barycentric coordinates.
    tb = np.array([1.0, -0.1, 0.1])
    lb = np.array([0.1, 1.0, -0.1])
    rb = np.array([-0.1, 0.1, 1.0])

    # This transforms the above barycentric coordinates to the original Axes
    # coordinates. In combination with ``ax.transAxes``, we can plot arrows fixed
    # to the Axes coordinates.
    f = tax.transAxesProjection.transform

    tarrow = FancyArrowPatch(f(ta), f(tb), ec='C0', fc='C0', **kwargs_arrow)
    larrow = FancyArrowPatch(f(la), f(lb), ec='C1', fc='C1', **kwargs_arrow)
    rarrow = FancyArrowPatch(f(ra), f(rb), ec='C2', fc='C2', **kwargs_arrow)
    tax.add_patch(tarrow)
    tax.add_patch(larrow)
    tax.add_patch(rarrow)

    # To put the axis-labels at the positions consistent with the arrows above
    kwargs_label = {
        'transform': tax.transTernaryAxes,
        'backgroundcolor': 'w',
        'ha': 'center',
        'va': 'center',
        'rotation_mode': 'anchor',
        'zorder': -9,
    }

    # Put axis-labels on the midpoints of arrows
    tpos = (ta + tb) * 0.5
    lpos = (la + lb) * 0.5
    rpos = (ra + rb) * 0.5

    tax.text(*tpos, elements[0], color='C0', rotation=-60, **kwargs_label, size=16)
    tax.text(*lpos, elements[1], color='C1', rotation=60, **kwargs_label, size=16)
    tax.text(*rpos, elements[2], color='C2', rotation=0, **kwargs_label, size=16)

    tax.grid(True, lw=1)
    tax.set_axisbelow(True)

    cs = tax.tricontourf(comp_dict_uniform[elements[0]], comp_dict_uniform[elements[1]],
                         comp_dict_uniform[elements[2]], fitness_uniform, cmap='plasma', alpha=0.5)

    # Add colorbar
    cax = tax.inset_axes([0.05, -0.2, 0.85, 0.05], transform=tax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax, orientation="horizontal")
    colorbar.mappable.set_clim(0, 1)
    colorbar.set_label(r'Magnetization ($\mu_B$/atom)', fontsize=16)
    colorbar.ax.tick_params(labelsize=14)

    # Plot the population distribution with different colors for each subpopulation
    for i, subpop in enumerate(subpopulations):
        comp_dict = {element: [comp[element] for comp in subpop] for element in elements}
        tax.scatter(comp_dict[elements[0]], comp_dict[elements[1]], comp_dict[elements[2]],
                    c=colors[i], edgecolor='black', label=f'Subpopulation {i + 1}', s=50, alpha=0.7)

    # Add legend
    #tax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    tax.text(0.25, 0.95, f'Generation: {generation_number}',
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes, fontdict={'size': 20})

    # Set ticks and labels
    ticks = np.linspace(0, 1, 11)
    formatted_ticklabels = [f'{label*100:.0f}' for label in ticks]

    tax.taxis.set_ticks(ticks)
    tax.taxis.set_ticklabels(formatted_ticklabels, fontsize=14)
    tax.laxis.set_ticks(ticks)
    tax.laxis.set_ticklabels(formatted_ticklabels, fontsize=14)
    tax.raxis.set_ticks(ticks)
    tax.raxis.set_ticklabels(formatted_ticklabels, fontsize=14)

    fig.savefig(f'graphs/{generation}.png', transparent=False, bbox_inches='tight', dpi=300)
    plt.close(fig)


def tournament_selection(population, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament_candidates = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = max(tournament_candidates, key=lambda x: x[1])[0]
        selected_parents.append(winner)
    return selected_parents


def stochastic_universal_sampling_selection(population, fitness_scores, num_parents):
    total_fitness = sum(fitness_scores)
    spacing = total_fitness / num_parents
    start = random.uniform(0, spacing)
    parents = []
    cumulative_fitness = 0
    idx = 0
    for _ in range(num_parents):
        target = start + spacing * idx
        while cumulative_fitness < target:
            cumulative_fitness += fitness_scores[idx]
            idx += 1
            if idx >= len(fitness_scores):
                idx = 0
        parents.append(population[idx - 1])
    return parents


def evolve_subpopulation(subpopulation, mutation_rate, crossover_rate, tournament_size, sharing_radius):
    fitness_scores_real = calculate_fitness(subpopulation)
    fitness_scores = fitness_sharing(fitness_scores_real, subpopulation, sharing_radius)

    selected_parents = tournament_selection(subpopulation, fitness_scores, tournament_size)

    new_subpopulation = []
    while len(new_subpopulation) < len(subpopulation):
        parents = random.sample(selected_parents, 2)
        if np.random.rand() < crossover_rate:
            child1 = arithmetic_crossover(parents[0], parents[1])
            child2 = arithmetic_crossover(parents[1], parents[0])
        else:
            child1, child2 = parents[0], parents[1]

        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)

        new_subpopulation.append(child1)
        new_subpopulation.append(child2)

    return new_subpopulation[:len(subpopulation)], max(fitness_scores), subpopulation[np.argmax(fitness_scores)], max(fitness_scores_real)



def migrate_population(subpopulations, migration_rate=0.1):
    for i in range(len(subpopulations)):
        num_migrants = int(len(subpopulations[i]) * migration_rate)
        migrants = random.sample(subpopulations[i], num_migrants)
        for migrant in migrants:
            target_subpop = (i + random.randint(1, len(subpopulations) - 1)) % len(subpopulations)
            subpopulations[target_subpop].append(migrant)
            subpopulations[i].remove(migrant)
    return subpopulations


def save_best_individuals(generation, best_individuals, best_fitnesses):
    data = []
    for i, (ind, fitness) in enumerate(zip(best_individuals, best_fitnesses)):
        data.append({
            'generation': generation,
            'subpopulation': i + 1,
            'best_individual': ind.as_dict(),
            'best_fitness': fitness
        })
    with open('best_individuals_all2.json', 'a') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')  # Write a newline for each JSON object


def calculate_population_diversity(population):
    compositions = np.array([list(comp.values()) for comp in population])
    return np.std(compositions)

def adaptive_mutation_rate(base_rate, diversity, diversity_threshold, improvement, improvement_threshold):
    if diversity < diversity_threshold:
        return base_rate * 1.0
    if improvement < improvement_threshold:
        return base_rate * 1.5
    return base_rate

def adaptive_crossover_rate(base_rate, diversity, diversity_threshold):
    if diversity < diversity_threshold:
        return base_rate * 1
    return base_rate


if __name__ == "__main__":
    population_size = 450
    subpopulation_count = 15
    subpopulation_size = population_size // subpopulation_count
    crossover_rate = 1.0
    max_generations = 100
    initial_mutation_rate = 0.07
    initial_crossover_rate = 1.0
    diversity_threshold = 0.02
    improvement_threshold = 0.001
    migration_rate = 0.1
    migration_interval = 5
    diversity_convergence_threshold = 0.01
    diversity_convergence_generations = 10
    n_jobs = -1  # Use all available CPU cores

    element_list = ['Ni', 'Co', 'Mn', 'Ti']
    #radius = 0.15
    #subpopulations = generate_subpopulations_within_range(subpopulation_count, subpopulation_size, radius, element_list)
    population = [generate_random_composition(element_list) for _ in range(population_size)]
    subpopulations = [population[i * subpopulation_size:(i + 1) * subpopulation_size] for i in
                      range(subpopulation_count)]

    best_fitness = float('-inf')
    best_individual = None
    prev_best_fitness = best_fitness
    consecutive_generations = 0
    fitness_history = []
    consecutive_low_diversity_generations = 0

    for generation in range(max_generations):

        '''if generation == 0:
            uniform_population = generate_uniform_composition(subpopulations[0][0].elements, 50)
            comp_dict_uniform = {element: [comp[element] for comp in uniform_population] for element in subpopulations[0][0].elements}
            fitness_uniform = calculate_fitness(uniform_population)
            plot_population_distribution(subpopulations, generation, comp_dict_uniform, fitness_uniform, generation)
        else:
            plot_population_distribution(subpopulations, generation, comp_dict_uniform, fitness_uniform, generation)'''


        diversity = np.mean([calculate_population_diversity(subpop) for subpop in subpopulations])
        improvement = best_fitness - prev_best_fitness
        mutation_rate = adaptive_mutation_rate(initial_mutation_rate, diversity, diversity_threshold, improvement,
                                               improvement_threshold)
        crossover_rate = adaptive_crossover_rate(initial_crossover_rate, diversity, diversity_threshold)

        results = Parallel(n_jobs=n_jobs)(delayed(evolve_subpopulation)(
            subpop, mutation_rate, crossover_rate, tournament_size=3, sharing_radius=0.1) for subpop in subpopulations)

        subpopulations, subpop_fitnesses, best_individuals, subpop_fitnesses_real = zip(*results)

        best_subpop_fitness = max(subpop_fitnesses)
        best_subpop_best_individual = best_individuals[subpop_fitnesses.index(best_subpop_fitness)]

        fitness_history.append(best_subpop_fitness)

        if best_subpop_fitness > best_fitness:
            best_fitness = best_subpop_fitness
            best_individual = best_subpop_best_individual

        save_best_individuals(generation, best_individuals, subpop_fitnesses_real)

        if generation % migration_interval == 0:
            subpopulations = migrate_population(subpopulations, migration_rate)

        if diversity < diversity_convergence_threshold:
            consecutive_low_diversity_generations += 1
            if consecutive_low_diversity_generations >= diversity_convergence_generations:
                print(f"Stopping early due to low diversity at generation {generation}.")
                break
        else:
            consecutive_low_diversity_generations = 0

        prev_best_fitness = best_fitness

    np.savetxt('fitness_history.dat', fitness_history)