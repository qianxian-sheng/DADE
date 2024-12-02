from population import *
from auxiliary import *
import random
import copy
from scipy.integrate import quad

NP = 0  # number of individuals
F = 0.5  # scale factor
CR = 0.9  # crossover rate
accuracy_level = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]  # accuracy levels
accuracy = accuracy_level[3]

FES = 0  # current fitness evaluations
MAX_FES = 0  # maximum of fitness evaluations
sub_populations = []  # each subpopulation(niches)
S = []  # success archive
Archive = []  # taboo archive

gbest_f = -math.inf  # the fitness of global best individual


def create_individuals(problem):  # create NP individuals
    pop = []
    for i in range(NP):
        indi = Individual(problem)
        indi.eval_fitness(problem)
        pop.append(indi)
    global FES
    FES += NP

    return pop


def diversity_based_niche_division(pop, problem):  # divide one niche
    global FES, MAX_FES, NP
    division = []
    best_i = find_best_individual_in_pop(pop)  # find the best individual
    best = pop[best_i]
    division.append(best)

    farthest_i = find_farthest_individual(pop, best.position)
    farthest = pop[farthest_i]
    d_max = np.linalg.norm(best.position - farthest.position)  # d_max

    remain = [x for x in pop if x != best]  # the remaining individuals
    A = get_diagonal_length(problem)
    d_min = A / NP  # d_min
    if FES > MAX_FES:
        FES = MAX_FES
    da = (d_max - d_min) * (1 - FES / MAX_FES) + d_min  # da

    distances = [np.linalg.norm(x.position - best.position) for x in remain]
    in_range = []
    for i in range(len(remain)):  # find individuals within range da
        indi = remain[i]
        d = distances[i]
        if d <= da:
            in_range.append(indi)

    initial_distances = []
    if len(in_range) >= 5:  # enough individuals within range
        indices = [x for x in range(len(in_range))]  # randomly select 5 individuals
        selected = random.sample(indices, 5)
        for i in range(len(selected)):
            indi = in_range[selected[i]]
            division.append(indi)

        for i in range(1, len(division)):
            initial_distances.append(np.linalg.norm(division[i].position - division[0].position))
    else:  # not enough individuals within range
        pairs = [(i, distances[i]) for i in range(len(remain))]
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        top = sorted_pairs[:5]
        indices = [item[0] for item in top]
        for i in range(len(indices)):
            indi = remain[indices[i]]
            division.append(indi)

    old_Do = cal_Do(division, problem)
    while True:  # add new individual according to the optimal based diversity
        remain = [x for x in remain if x not in division]
        if not remain:
            break
        sorted_remain = sorted(remain, key=lambda x: np.linalg.norm(x.position - best.position))
        division.append(sorted_remain[0])
        new_Do = cal_Do(division, problem)
        if new_Do > old_Do:
            division.pop()
            break
        else:
            old_Do = new_Do

    if len(in_range) >= 5:  # compare the initial 5 individuals and the last individual
        last_distance = np.linalg.norm(division[0].position - division[-1].position)
        indices_to_remove = []
        for i in range(len(initial_distances)):
            di = initial_distances[i]
            if di > last_distance:
                indices_to_remove.append(i + 1)

        for index in sorted(indices_to_remove, reverse=True):
            if 0 <= index < len(division):
                e = division.pop(index)
                remain.append(e)

        remain = [x for x in remain if x not in division]
        if len(division) < 6:
            needed = 6 - len(division)
            sorted_remain = sorted(remain, key=lambda x: np.linalg.norm(x.position - best.position))
            elements_to_add = sorted_remain[:needed]
            division.extend(elements_to_add)
            sorted_remain = sorted_remain[needed:]
            remain = copy.deepcopy(sorted_remain)

    return division, remain


def niches_divisions(pop, problem):  # divide all niches
    global NP
    divisions = []
    remain_pop = copy.deepcopy(pop)
    while len(remain_pop) >= 6:
        division, remain = diversity_based_niche_division(remain_pop, problem)
        divisions.append(division)
        remain_pop = copy.deepcopy(remain)
        if len(divisions) >= math.sqrt(NP):  # the maximum number of niches
            break
    bests = [x[0] for x in divisions]
    for i in range(len(remain_pop)):
        indi = remain_pop[i]
        nearest_i = find_nearest_individual(bests, indi.position)
        divisions[nearest_i].append(indi)

    return divisions


def divide_population(pop, problem, rho):  # divide all the niches and place them in the array "sub_populations"
    global gbest_f, accuracy
    divisions = niches_divisions(pop, problem)
    sub_populations.clear()
    for i in range(len(divisions)):
        p = Population(divisions[i])
        p.update(gbest_f, accuracy, problem, rho)
        sub_populations.append(p)


def get_best_individual():  # find the best individual in the current population
    best_i = 0
    best_f = sub_populations[0].xbest.fitness
    for i in range(len(sub_populations)):
        if sub_populations[i].xbest.fitness > best_f:
            best_i = i
            best_f = sub_populations[i].xbest.fitness

    return sub_populations[best_i].xbest


def distribution_of_diversity(Du, D):  # the distribution of diversity
    if Du > 0.25:
        return 0
    return (1 - Du / 0.25) ** D


def diversity_based_mutation(p, problem, j):  # the diversity based mutation
    D = problem.get_dimension()
    Du = p.Dc
    boundary = distribution_of_diversity(Du, D)
    r = np.random.rand()
    if r <= boundary:
        indices = [x for x in range(p.size) if x != p.xbest_index]
        selected = random.sample(indices, 4)
        v = p.xbest.position + F * \
            (p.individuals[selected[0]].position - p.individuals[selected[1]].position) + F * \
            (p.individuals[selected[2]].position - p.individuals[selected[3]].position)
    else:
        indices = [x for x in range(p.size) if x != j]
        selected = random.sample(indices, 5)
        v = p.individuals[selected[0]].position + F * \
            (p.individuals[selected[1]].position - p.individuals[selected[2]].position) + F * \
            (p.individuals[selected[3]].position - p.individuals[selected[4]].position)

    return v


def crossover(individual, v):  # crossover
    j = random.randint(0, len(v) - 1)
    u = np.zeros(len(v))
    for i in range(len(v)):
        n = random.random()
        if n <= CR or j == i:
            u[i] = v[i]
        else:
            u[i] = individual.position[i]

    return u


def selection(p, j, u, problem):  # selection
    U = Individual(problem)
    U.position = u
    U.eval_fitness(problem)
    old = p.individuals[j]
    if U.fitness > old.fitness:
        p.individuals[j] = U


def common_reinitialization(problem):  # randomly reinitialize the individuals
    new = Individual(problem)
    new.eval_fitness(problem)

    return new


def taboo_reinitialization(problem):  # reinitialize the individuals with the taboo archive
    while True:
        new = Individual(problem)
        flag = True
        for i in range(len(Archive)):
            pair = Archive[i]
            d = np.linalg.norm(pair[0].position - new.position)
            if d < pair[1]:
                flag = False
                break
        if flag:
            new.eval_fitness(problem)
            return new


def append_archive(new_individual, new_radius):  # add one pair of data into the taboo archive
    global gbest_f, Archive, accuracy
    if new_individual.fitness < accuracy - gbest_f:  # not excellent enough
        return
    if_append = False
    add = True
    for i in range(len(Archive)):
        pair = Archive[i]
        d = np.linalg.norm(new_individual.position - pair[0].position)
        if d <= pair[1]:
            if new_individual.fitness > pair[0].fitness:
                new_pair = (new_individual, new_radius)
                Archive[i] = new_pair
                if_append = True
            add = False
            break
    if add:
        Archive.append((new_individual, new_radius))
        if_append = True

    i = 0
    while i < len(Archive):  # delete the elements which are not excellent enough
        pair = Archive[i]
        if abs(pair[0].fitness-gbest_f) > accuracy:
            del Archive[i]
        else:
            i += 1

    return if_append


def update_archive():  # add all the data into the taboo archive
    global accuracy
    for i in range(len(sub_populations)):
        p = sub_populations[i]
        xbest = copy.deepcopy(p.xbest)
        p.added = append_archive(xbest, p.radius)


def stagnant_check_and_processing(problem, rho, lambdas):  # check and process the stagnant niches
    global accuracy, gbest_f, FES
    for i in range(len(sub_populations)):
        p = sub_populations[i]
        if p.gap_counter > lambdas:
            for j in range(p.size):
                p.individuals[j] = taboo_reinitialization(problem)
                FES += 1
            p.update(gbest_f, accuracy, problem, rho)
            p.trapped = True  # the niche has been trapped into local optima


# Check if any niches that once fell into local optima converged to the taboo regions
def stagnant_in_taboo_regions_check(problem, rho):
    global FES, gbest_f, accuracy
    for i in range(len(sub_populations)):
        p = sub_populations[i]
        if p.added:  # the niche has been added in the taboo archive
            continue
        D = problem.get_dimension()
        boundary, _ = quad(distribution_of_diversity, 0, 0.25, args=(D,))
        # the niche has been trapped into local optima and its diversity is low enough
        if p.trapped and p.Dc < boundary:
            in_range = False
            for j in range(p.size):
                indi = p.individuals[j]
                for k in range(len(Archive)):
                    pair = Archive[k]
                    d = np.linalg.norm(pair[0].position - indi.position)
                    if d < pair[1]:
                        in_range = True
                        p.individuals[j] = taboo_reinitialization(problem)
                        FES += 1
                        break
            p.update(gbest_f, accuracy, problem, rho)
            if not in_range:
                p.trapped = False


# Check if any niche successfully converges to the global optimal solution
def converge_check(problem):
    global accuracy, gbest_f, FES
    re = False
    for i in range(len(sub_populations)):
        p = sub_populations[i]
        flag = True
        for j in range(p.size):
            indi = p.individuals[j]
            if abs(indi.fitness - gbest_f) > accuracy:
                flag = False
        if flag:
            for j in range(p.size):
                indi = p.individuals[j]
                S.append(indi)
                p.individuals[j] = common_reinitialization(problem)
                FES += 1
                re = True

    return re


# Check if any niche successfully converges to the global optimal solution (for high dimensional problems)
def high_dimensional_converge_check(problem, lambdas):
    global FES
    re = False
    for i in range(len(sub_populations)):
        p = sub_populations[i]
        flag = False
        if p.near_counter > lambdas:
            flag = True
        if flag:
            for j in range(p.size):
                indi = p.individuals[j]
                S.append(indi)
                p.individuals[j] = common_reinitialization(problem)
                FES += 1
                re = True

    return re


def redivide_population(problem, rho):  # redividing the population into several niches
    pop = []
    for i in range(len(sub_populations)):
        p = sub_populations[i]
        for j in range(p.size):
            pop.append(p.individuals[j])

    divide_population(pop, problem, rho)


def DADE(problem, rho, lambdas):  # the main DADE algorithm
    global FES, MAX_FES, gbest_f, accuracy
    dim = problem.get_dimension()
    MAX_FES = problem.get_maxfes()
    pop = create_individuals(problem)
    divide_population(pop, problem, rho)
    while FES < MAX_FES:  # while the end condition is not reached
        for i in range(len(sub_populations)):
            p = sub_populations[i]
            for j in range(p.size):
                indi = p.individuals[j]
                v = diversity_based_mutation(p, problem, j)  # diversity based mutation
                u = crossover(indi, v)  # crossover
                selection(p, j, u, problem)  # selection
                FES += 1
            p.update(gbest_f, accuracy, problem, rho)  # update the information of each niche

        best = get_best_individual()
        best_f = best.fitness
        if best_f > gbest_f:  # update the global optima
            gbest_f = best_f

        update_archive()  # update the taboo archive
        stagnant_check_and_processing(problem, lambdas, lambdas)  # local optima processing
        stagnant_in_taboo_regions_check(problem, rho)  # local optima processing

        # Check if any niche successfully converges to the global optimal solution
        if dim < 10:
            if converge_check(problem):
                redivide_population(problem, rho)
        else:
            if high_dimensional_converge_check(problem, lambdas):
                redivide_population(problem, rho)


result = []


def calculate_data(problem):  # calculate the result of one run
    positions = []
    for i in range(len(sub_populations)):
        p = sub_populations[i]
        for j in range(p.size):
            indi = p.individuals[j]
            positions.append(indi.position)

    for i in range(len(S)):
        positions.append(S[i].position)

    positions = np.array(positions)
    count, _ = how_many_goptima(positions, problem, accuracy)

    return count


def reset():  # Reset some base variables
    sub_populations.clear()
    global FES, gbest_f
    FES = 0
    S.clear()
    Archive.clear()
    gbest_f = -math.inf


def set_NP(n_function):  # set the number of individuals according to the problem
    global NP
    if 1 <= n_function <= 5:
        NP = 80
    elif n_function == 6 or n_function == 10:
        NP = 100
    elif 7 <= n_function <= 9:
        NP = 300
    elif 11 <= n_function <= 20:
        NP = 200


def run_multiple_times(time, func_no, rho, lambdas):  # run DADE multiple times on one function
    count = 0
    problem = CEC2013(func_no)
    success = 0
    set_NP(func_no)
    for i in range(time):
        reset()
        DADE(problem, rho, lambdas)
        peaks = calculate_data(problem)
        count += peaks
        if peaks == problem.get_no_goptima():
            success += 1
    PR = count/(problem.get_no_goptima()*time)
    SR = success / time

    return PR, SR








