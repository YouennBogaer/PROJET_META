import numpy as np
import random as rd
import src.python.scripts.MOP as MOP

class MOEAD():
    def __init__(self, mop:MOP, stop_criterion, weights, len_neighborhood):
        """
        Args:
            mop (MOP Class) : multi objectif problem to treat.
            stop_criterion : indicates what should stop the research of Pareto front solutions 
                (see self.is_criterion_met() and self.execute()).
            weights (List[Array]): List of Weight vectors corresponding to each subproblems. 
                Each Weight vector is an array of length equals to the problem dimension i.e. number of objectives.
            len_neighborhood (int): number of closest weight vector to find in the neighborhood of each weight vector.
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
        self.population, self.F_pop = self.init_pop()
        
        # Step 1.4 : Reference solution
        self.z_ = self.init_z()
    
    
    # Step 2
    def update(self):
        for i in range(self.N_):
            # Step 2.1 : Reproduction
            k,l = self.select_mating(i)
            x_k, x_l = self.population[k], self.population[l]
            y = self.generate_solution(x_k,x_l)
            
            # Step 2.2 : Improvement
            y_clean = self.repair(y)
            
            # Step 2.3 : Update of z
            F_y = self.mop.evaluate(y_clean)
            for m in range(self.dim):
                if self.z_[m] < F_y[m]:
                    self.z_[m] = F_y[m]
            
            # Step 2.4 : Update of neighboring solutions
            for j in self.B_[i]:
                g_y_clean = self.tcheb_aggFunc(i,self.weights,F_y,self.z_)
                g_x_j = self.tcheb_aggFunc(i,self.weights,self.F_pop[i],self.z_)

                if g_y_clean <= g_x_j:
                    self.population[j] = y_clean
                    self.F_pop[j] = F_y
                    
            # Step 2.5 Update of EP (External Population)
            # Removes dominated solutions from External Population
            self.ex_pop = [x for x in self.ex_pop if not self.is_dominated_by(x,y_clean)]
            # Adds y_clean to the External Population if non dominated
            if self.non_dominated_in_expop(y_clean):
                self.ex_pop.append(y_clean)
        

    # TODO
    # Step 3
    def execute(self):
        while not self.is_criterion_met():
            self.update()
    
    def tcheb_aggFunc(index, lambdas, F, z):
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
        M = len(lambdas)
        for m in range(M):
            g = lambdas[index][m]*abs(F[m]-z[m])
            if max_gap < g:
                max_gap = g
        return max_gap
    
    def select_mating(self, index):
        """
        Returns the indices k,l of two randomly chosen neighbors of solution 'index' picked in self.B_[index].
        See # Step 2.1 in self.update()
        """
        indices = range(self.T_)
        i = rd.randint(0,self.T_-1)
        # We remove k so we assure that output indices are distinct but random
        rest_indices = indices[:k] + indices[k+1:]
        
        j_id = rd.randint(0,len(rest_indices))
        j = rest_indices[j_id]
        
        return self.B_[index][i], self.B_[index][j]
        
    def is_dominated_by(self, x, y):
        """
        Determines if solution x is dominated by solution y.

        Returns:
            Bool
        """
        m = 0
        F_x = self.mop.evaluate(x)
        F_y = self.mop.evaluate(y)

        while m<self.dim and F_x[m]<F_y[m]:
            m += 1
        if m==self.dim:
            return True
        else:
            return False
        
    def non_dominated_in_expop(self, y):
        """
        Checks if solution y is non dominated
        in the population self.ex_pop

        Returns:
            Bool
        """
        for x in self.ex_pop:
            if self.is_dominated_by(y, x):
                return False
        return True
        
    # TODO
    def generate_solution(self, x1, x2):
        """
        Generate a solution y based on both input solutions x1 and x2, using a genetic operation
        """
        return np.array([0 for _ in range(self.dim)]) 
    
    # TODO
    def repair(self, y):
        """
        Apply a problem-specific repair/improvement heuristic on y to produce y_clean.
        """
        y_clean = y
        return y_clean  
    
    #TODO
    def neighborhood(self):
        """
        Returns the indices of the T closest weight vectors of any weight vector based on euclidian distance.
        Output should be a list B_ of length self.N_.
        """
        neighborhood = [None for _ in range(self.N_)]
        # inverted dict to obtain index from weight vector as a key
        weight_dict = {w:i for i,w in enumerate(self.weights)}
        for index_w,w in enumerate(self.weights):
            # We sort weight vectors by distance from w
            sorted_neighbors_w = self.weights.copy().sort(key=lambda x:np.linalg.norm(x,w))
            # We keep the indexes of T_ closest weight vectors, except index 0 which is w itself
            neighborhood[index_w] = [weight_dict[neighbor] for neighbor in sorted_neighbors_w[1:self.T_+1]]

        return neighborhood
    
    # TODO
    def init_pop(self):
        """
        Generates a population and stores each element by the MOP objective functions F.
        Returns:
            List[Array], List[Array]: population of solutions, evaluations by objective functions
        """
        # Need to figure out the generation of population
        population = [np.array([0 for _ in range(self.dim)]) for _ in range(self.N_)]
        F_pop = [self.mop.evaluate(x) for x in population]
        return population, F_pop
    
    # TODO
    def init_z(self):
        return [0 for i in range(self.dim)]
    
    # TODO
    def is_criterion_met(self):
        return True

    