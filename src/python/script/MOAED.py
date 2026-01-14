import numpy as np
import random as rd

from src.python.script.MOP import MOP


class MOEAD():
    def __init__(self, mop: MOP, stop_criterion, weights, len_neighborhood, params, init_population=None):
        """
        Args:
            mop (MOP Class) : multi objectif problem to treat.
            stop_criterion : indicates what should stop the research of Pareto front solutions 
                (see self.is_criterion_met() and self.execute()).
            weights (List[Array]): List of Weight vectors corresponding to each subproblems. 
                Each Weight vector is an array of length equals to the problem dimension i.e. number of objectives.
            len_neighborhood (int): number of closest weight vector to find in the neighborhood of each weight vector.
            init_population (List[array]): if a preferred initial population is needed.
        """
        self.mop = mop
        # dimension of the multi objective problem (number of objectives)
        self.dim = mop.getDim()

        self.criterion = stop_criterion
        self.weights = weights
        self.T_ = len_neighborhood

        # There are as much subproblems as there are weight vectors
        self.N_ = len(self.weights)

        # Step 1.1 : Initialize EP (External population) as empty
        # Will be the Pareto Front at the end
        self.ex_pop = []

        # Step 1.2 : Find T closest neighbors for each of the N weight vectors
        # Stores the neighborrhood for each subproblems as an index of self.weights, and self.population
        self.B_ = self.neighborhood()

        # Step 1.3 : Generate an initial population
        if init_population is None:
            self.population, self.F_pop = self.init_pop()
        else:
            if len(init_population) != self.N_:
                raise ValueError(f"Initial population should be of same length as weights.\nPopulation length : {len(self.population)}\nExpected length :{self.N_}")
            self.population = init_population
            self.F_pop = [self.mop.evaluate(x) for x in self.population]

        # Step 1.4 : Reference solution for each objectif
        self.z_ = self.init_z()

        # Competeur de genération
        self.gen = 0
        # TODO : #Peut être mettre le dictionnaire paramls ici (plus logique) et l'ajouter en paramètre de classe
        self.params = params

    # Step 2
    def update(self):
        for i in range(self.N_):
            # Step 2.1 : Reproduction

            #Selction of 2 parents
            k, l = self.select_mating(i)
            x_k, x_l = self.population[k], self.population[l] # 2 chromosomes of the parents
            y = self.generate_solution(x_k, x_l) # create a child

            # Step 2.2 : Improvement
            y_clean = self.repair(y)

            # Step 2.3 : Update of z
            F_y = self.mop.evaluate(y_clean)
            for m in range(self.dim):
                if self.z_[m] > F_y[m]:
                    self.z_[m] = F_y[m]

            # Step 2.4 : Update of neighboring solutions
            for j in self.B_[i]:

                g_y = self.tcheb_aggFunc(j, self.weights, F_y, self.z_)
                g_old_j = self.tcheb_aggFunc(j, self.weights, self.F_pop[j], self.z_)

                if g_y <= g_old_j:
                    self.population[j] = y_clean
                    self.F_pop[j] = F_y

            # Step 2.5 Update of EP (External Population)
            # Adds y_clean to the External Population if non dominated
            if self.non_dominated_in_expop(F_y):
                self.ex_pop = [pair for pair in self.ex_pop if not self.is_dominated_by(pair[1], F_y)]
                # On ajoute la nouvelle paire
                self.ex_pop.append((y_clean, F_y))

        # On met à jour le compteur de génération
        self.gen += 1

    # Step 3
    def execute(self):
        """
        Execute the MOEAD algorithm.
        """
        #If we don't pass the limit we continue
        while not self.is_criterion_met():
            self.update()
            if self.gen % 10 == 0:
                print(f"Génération {self.gen} terminée. Taille EP: {len(self.ex_pop)}")

    def tcheb_aggFunc(self, index, lambdas, F, z):
        """
        Compute the aggregation function following the Tchebycheff approach.
        
        Args:
            index (int): current subproblem index (1 to N)
            lambdas (List[Array]): weight vectors
            F (Array): values of objective functions for current index
            z (List[int]): reference

        Returns:
            int : value of the aggregation function, see Tchebycheff approach.
        """
        max_gap = 0

        for m in range(self.dim):
            g = lambdas[index][m] * abs(F[m] - z[m])
            if max_gap < g:
                max_gap = g
        return max_gap

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

    # TODO
    def generate_solution(self, x1, x2):
        # On crée un enfant par croisement uniforme
        y = np.array(x1).copy()
        for i in range(len(y)):
            if rd.random() < 0.5:
                y[i] = x2[i]

        # Mutation : On change une tâche au hasard
        if rd.random() < 0.2:  # 20% de chance de mutation
            idx_tache = rd.randint(0, (len(y) // 2) - 1)
            # Mutation du serveur
            y[2 * idx_tache] = rd.randint(0, self.params['M'])
            # Mutation des ressources
            y[2 * idx_tache + 1] = rd.randint(1, 7)

        return y

    def repair(self, y):
        # On s'assure que les valeurs restent des entiers dans les bornes
        for i in range(len(y)):
            low, high = self.mop.bounds_x[i]
            y[i] = int(np.clip(y[i], low, high))

        # Contrainte : Somme des CPU par serveur <= 8
        M = self.params['M']
        cap_max = self.params['mnv']  # 8

        usage = np.zeros(M + 1)
        for n in range(len(y) // 2):
            srv = int(y[2 * n])
            cpu = y[2 * n + 1]
            if srv > 0:  # Si c'est un MEC
                if usage[srv] + cpu > cap_max:
                    # Si ça déborde, on envoie la tâche sur le Cloud (Bricolage un peu)
                    y[2 * n] = 0
                else:
                    usage[srv] += cpu
        return y

        # TODO

    def neighborhood(self):
        """
        Returns :
            List(Array) : The list that keep the T closest neighbors
        """
        B = []
        for i in range(self.N_):
            distances = []
            for j in range(self.N_):
                # Distance between the vectors with weigth i and j (distance euclidienne)
                dist = np.linalg.norm(self.weights[i] - self.weights[j])
                # We had the distance to a list
                distances.append((dist, j))

            # We sort the list
            distances.sort()
            #T is an int to fix the number of closest neighbor that we keep

            # TODO Remarque est ce que le poids lui meme est son propre voisin pusique distance = 0 ?
            # Peut etre faire distances[1:self.T_+1]. 

            indices = [d[1] for d in distances[1:self.T_+1]]
            B.append(indices)
        return B

    # TODO
    def init_pop(self):
        """
        Generates a population and stores each element by the MOP objective functions F.
        Returns:
            List[Array], List[Array]: population of solutions, evaluations by objective functions
        """
        # Need to figure out the generation of population
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

    # TODO
    def init_z(self):
        """
        Initiates as infinite.
        """
        return [float('inf')] * self.dim

    # TODO
    def is_criterion_met(self):
        """
        Compares criterion with gen attribute.
        Returns:
            Bool
        """
        return self.gen >= self.criterion
