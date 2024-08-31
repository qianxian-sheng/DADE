from cec2013.cec2013 import *


def boundary_correction(vector, problem):  # boundary treatment
    up, low = get_boundary(problem)
    for i in range(len(vector)):
        if vector[i] > up[i]:
            vector[i] = 2 * up[i] - vector[i]
        elif vector[i] < low[i]:
            vector[i] = 2 * low[i] - vector[i]


def get_boundary(problem):  # obtain the upper and lower bounds of the problem in each dimension
    dim = problem.get_dimension()
    up = np.zeros(dim)
    low = np.zeros(dim)
    for i in range(dim):
        up[i] = problem.get_ubound(i)
        low[i] = problem.get_lbound(i)

    return up, low


def get_diagonal_length(problem):  # get the diagonal length of the problem
    up, low = get_boundary(problem)
    diagonal_length = sum([(up[i] - low[i]) ** 2 for i in range(len(up))])
    diagonal_length = math.sqrt(diagonal_length)

    return diagonal_length


def find_best_individual_in_pop(pop):  # find the best individual
    best_i = 0
    best_f = pop[0].fitness
    for j in range(len(pop)):
        indi = pop[j]
        if indi.fitness > best_f:
            best_f = indi.fitness
            best_i = j

    return best_i


def cal_Do(pop, problem):  # calculate the optimal based diversity
    best_i = find_best_individual_in_pop(pop)
    best = pop[best_i]

    sum_d = 0
    for i in range(len(pop)):
        if i == best_i:
            continue
        indi = pop[i]
        sum_d += np.linalg.norm(indi.position - best.position)
    A = get_diagonal_length(problem)
    M = len(pop) - 1
    Do = sum_d / (A * M)

    return Do


def find_nearest_individual(pop, point):  # find the nearest individual form "point" in pop
    nearest_x = 0
    nearest_d = math.inf
    for i in range(len(pop)):
        indi = pop[i]
        distance = np.linalg.norm(point - indi.position)
        if distance < nearest_d:
            nearest_x = i
            nearest_d = distance

    return nearest_x


def find_farthest_individual(pop, point):  # find the farthest individual form "point" in pop
    farthest_x = 0
    farthest_d = -math.inf
    for i in range(len(pop)):
        indi = pop[i]
        distance = np.linalg.norm(point - indi.position)
        if distance > farthest_d:
            farthest_x = i
            farthest_d = distance

    return farthest_x





