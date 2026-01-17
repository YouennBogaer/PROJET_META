import numpy as np
import random as rd

from src.python.script import params
from src.python.script.MOP import MOP
from src.python.script.metrics import pf, hypervolume


class MOEAD():
    def __init__(self, mop: MOP, stop_criterion, weights, len_neighborhood, params=params, init_population=None,
                 percentage_mutation=0.2):
        """
        Args:
            mop (MOP Class) : multi objective problem to treat.
            stop_criterion : indicates what should stop the research of Pareto front solutions
                (see self.is_criterion_met() and self.execute()).
            weights (List[Array]): List of Weight vectors corresponding to each subproblems.
                Each Weight vector is an array of length equals to the problem dimension i.e. number of objectives.
            len_neighborhood (int): number of closest weight vector to find in the neighborhood of each weight vector.
            init_population (List[array]): if a preferred initial population is needed.
        """
        self.listnbEP = []
        self.listHP = []
        self.mop = mop
        # dimension of the multi objective problem (number of objectives)
        self.dim = mop.getDim()
        self.params = params
        self.criterion = stop_criterion
        self.weights = weights
        self.T_ = len_neighborhood

        # There are as many subproblems as there are weight vectors
        self.N_ = len(self.weights)

        # Step 1.1 : Initialize EP (External population) as empty
        # Will be the Pareto Front at the end
        self.ex_pop = []

        # Step 1.2 : Find T closest neighbors for each of the N weight vectors
        # Stores the neighborhood for each subproblem as an index of self.weights, and self.population
        self.B_ = self.neighborhood()

        # Step 1.3 : Generate an initial population
        if init_population is None:
            self.population, self.F_pop = self.init_pop()
        else:
            if len(init_population) != self.N_:
                raise ValueError(
                    f"Initial population should be of same length as weights.\nPopulation length : {len(self.population)}\nExpected length :{self.N_}")
            self.population = init_population
            self.F_pop = [self.mop.evaluate(x) for x in self.population]

        # Step 1.4 : Reference solution for each objective
        self.z_ = self.init_z()

        # Nadir point initialization for normalization
        # Find the worst point (Max) for each objective in the initial population
        self.z_nad = [float('-inf')] * self.dim
        for sol_eval in self.F_pop:
            for m in range(self.dim):
                if sol_eval[m] > self.z_nad[m]:
                    self.z_nad[m] = sol_eval[m]

        # Generation counter
        self.gen = 0
        self.percentage_mutation = percentage_mutation

    # Step 2
    def update(self):
        for i in range(self.N_):
            # Step 2.1 : Reproduction

            # Selection of 2 parents
            k, l = self.select_mating(i)
            x_k, x_l = self.population[k], self.population[l]  # 2 chromosomes of the parents
            y = self.generate_solution(x_k, x_l)  # create a child

            # Step 2.2 : Improvement
            # Solution repair is delegated to the problem (MOP)
            y_clean = self.mop.repair(y)

            # Step 2.3 : Update of z AND z_nad
            F_y = self.mop.evaluate(y_clean)
            for m in range(self.dim):
                if self.z_[m] > F_y[m]:
                    self.z_[m] = F_y[m]

                # Update the Nadir point
                if self.z_nad[m] < F_y[m]:
                    self.z_nad[m] = F_y[m]

            # Step 2.4 : Update of neighboring solutions
            count = 0
            max_count = 2
            for j in self.B_[i]:
                if count >= max_count: break
                # Pass z_nad for normalization
                g_y = self.tcheb_aggFunc(j, self.weights, F_y, self.z_, self.z_nad)
                g_old_j = self.tcheb_aggFunc(j, self.weights, self.F_pop[j], self.z_, self.z_nad)

                if g_y <= g_old_j:
                    self.population[j] = y_clean
                    self.F_pop[j] = F_y
                    count += 1

            # Step 2.5 Update of EP (External Population)
            # Adds y_clean to the External Population if non dominated
            if self.non_dominated_in_expop(F_y):
                self.ex_pop = [pair for pair in self.ex_pop if not self.is_dominated_by(pair[1], F_y)]
                # Add the new pair
                self.ex_pop.append((y_clean, F_y))

        # Update generation counter
        self.gen += 1

    # Step 3
    def execute(self):
        """
        Execute the MOEAD algorithm.
        """
        # If we don't pass the limit we continue
        while not self.is_criterion_met():
            self.update()
            self.listnbEP.append(len(self.ex_pop))
            self.listHP.append(hypervolume(pf(self), [20.0, 65000.0, 20.0]))
        return self.listnbEP, self.listHP

    def tcheb_aggFunc(self, index, lambdas, F, z, z_nad):
        """
        Tchebycheff with STRICT Normalization.
        Forces all objectives onto a fair 0.0 - 1.0 scale.
        """
        max_val = float('-inf')

        for m in range(self.dim):
            # 1. Dynamic scale calculation (Nadir - Ideal)
            scale = z_nad[m] - z[m]

            # Safety: If scale is zero (all points are identical), set to 1.0
            if scale < 1e-9:
                scale = 1.0

            # 2. Normalization: (Value - Min) / (Max - Min)
            # Provides a value between 0 and 1 for ALL objectives
            normalized_diff = abs(F[m] - z[m]) / scale

            # 3. Weight application
            # If lambda is 0, keep a tiny trace (1e-6) to avoid ignoring the objective
            weight = lambdas[index][m]
            if weight < 1e-6: weight = 1e-6

            val = weight * normalized_diff

            if val > max_val:
                max_val = val

        return max_val

    def select_mating(self, index):
        """
        Returns the indices k,l of two randomly chosen neighbors of solution 'index' picked in self.B_[index].
        See # Step 2.1 in self.update()
        """
        indices = list(range(self.T_))
        i = rd.randint(0, self.T_ - 1)
        rest_indices = indices[:i] + indices[i + 1:]
        j_id = rd.randint(0, len(rest_indices) - 1)
        j = rest_indices[j_id]

        return self.B_[index][i], self.B_[index][j]

    def is_dominated_by(self, F_x, F_y):
        """
        Determines if solution x is dominated by solution y.

        Returns:
            Bool
        """
        y_wins = all(F_y[m] <= F_x[m] for m in range(self.dim))
        y_strictly_wins = any(F_y[m] < F_x[m] for m in range(self.dim))
        return y_wins and y_strictly_wins

    def non_dominated_in_expop(self, F_y):
        """
        Checks if solution y is non dominated
        in the population self.ex_pop

        Returns:
            Bool
        """
        for _, F_x in self.ex_pop:
            if self.is_dominated_by(F_x, F_y):
                return False
        return True

    def generate_solution(self, x1, x2):
        # Create a child via uniform crossover
        y = np.array(x1).copy()
        for i in range(len(y)):
            if rd.random() < 0.5:
                y[i] = x2[i]

        # Mutation: Change a random task
        if rd.random() < self.percentage_mutation:
            idx_tache = rd.randint(0, (len(y) // 2) - 1)
            # Server mutation
            y[2 * idx_tache] = rd.randint(0, self.params['M'])
            # Resource mutation
            y[2 * idx_tache + 1] = rd.randint(1, 7)

        return y

    def neighborhood(self):
        """
        Returns :
            List(Array) : The list that keep the T closest neighbors
        """
        B = []
        for i in range(self.N_):
            distances = []
            for j in range(self.N_):
                # Distance between vectors with weights i and j (Euclidean distance)
                dist = np.linalg.norm(self.weights[i] - self.weights[j])
                # Add distance to the list
                distances.append((dist, j))

            # Sort the list
            distances.sort()
            # indices of the closest neighbors
            indices = [d[1] for d in distances[:self.T_]]
            B.append(indices)
        return B

    def init_pop(self):
        """
        Generates a population and stores each element by the MOP objective functions F.
        Returns:
            List[Array], List[Array]: population of solutions, evaluations by objective functions
        """
        population = []
        for _ in range(self.N_):
            chromosome = []
            for _ in range(self.params['N']):
                srv = rd.randint(0, self.params['M'])
                cpu = rd.randint(1, 7)
                chromosome.extend([srv, cpu])
            population.append(chromosome)

        f_pop = [self.mop.evaluate(ind) for ind in population]
        return population, f_pop

    def init_z(self):
        """
        Initiates as infinite.
        """
        return [float('inf')] * self.dim

    def is_criterion_met(self):
        """
        Compares criterion with gen attribute.
        Returns:
            Bool
        """
        return self.gen >= self.criterion