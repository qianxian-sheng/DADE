from auxiliary import *


class Individual:  # individual
    def __init__(self, problem):
        dim = problem.get_dimension()
        self.position = np.zeros(dim)
        up, low = get_boundary(problem)
        for i in range(dim):
            self.position[i] = low[i] + np.random.rand() * (up[i] - low[i])
        self.fitness = None

    def eval_fitness(self, problem):
        boundary_correction(self.position, problem)  # boundary treatment
        self.fitness = problem.evaluate(self.position)


class Population:  # subpopulation(niche)
    def __init__(self, division):
        self.individuals = []  # individuals within the niche
        for i in range(len(division)):
            self.individuals.append(division[i])

        self.size = len(division)  # size of the niche
        self.xbest = self.individuals[0]  # best individual of the niche
        self.xbest_index = 0  # index of the best individual
        self.xbest_f = -math.inf  # fitness of the best individual

        self.gap_counter = 0  # the continuous iterations that its best individual is worse than global optima

        self.Dc = None  # centroid based diversity

        # distance between the best individual and its nearest individual
        self.radius = np.zeros(len(self.individuals[0].position))

        self.cal_radius()  # 最小值

        self.trapped = False  # whether the niche has been trapped into local optima
        self.added = False  # whether the niche has been added in the taboo archive

    def cal_Dc(self, problem):  # calculate the centroid based diversity
        S = []
        for i in range(self.size):
            S.append(self.individuals[i].position)
        S = np.array(S)
        mean_position = np.mean(S, axis=0)

        sum_d = 0
        for i in range(self.size):
            sum_d += np.linalg.norm(mean_position - self.individuals[i].position)
        A = get_diagonal_length(problem)
        Dc = sum_d / (self.size * A)

        return Dc

    def update(self, gbest, accuracy, problem):  # update the information of the niche
        best_i = 0
        best_f = self.individuals[0].fitness
        for i in range(self.size):
            indi = self.individuals[i]
            if indi.fitness > best_f:
                best_f = indi.fitness
                best_i = i
        self.xbest = self.individuals[best_i]
        self.xbest_index = best_i

        self.Dc = self.cal_Dc(problem)  # calculate the diversity

        if self.Dc < 1e-5:
            if abs(self.xbest.fitness - gbest) > accuracy:
                self.gap_counter += 1
            else:
                self.gap_counter = 0
        else:
            self.gap_counter = 0

    def cal_radius(self):  # calculate distance between the best individual and its nearest individual
        nearest_i = 0
        nearest_d = math.inf
        for i in range(self.size):
            indi = self.individuals[i]
            d = np.linalg.norm(indi.position - self.xbest.position)
            if d == 0:
                continue
            if d < nearest_d:
                nearest_d = d
                nearest_i = i
        if nearest_d == math.inf:
            self.radius = 0
            return
        nearest = self.individuals[nearest_i]
        self.radius = np.linalg.norm(nearest.position - self.xbest.position)

