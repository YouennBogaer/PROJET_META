import numpy as np
import random as rd
import MOP

class MOEAD():
    def __init__(self, mop:MOP, stop_criterion, weights, len_neighborhood):
        """_summary_

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
        self.ex_pop = []
        
        # Step 1.2 : Find T closest neighbors for each of the N weight vectors
        self.B_ = self.neighborhood()
        
        # Step 1.3 : Generate an initial population
        self.population = self.init_pop()
        
        # Step 1.4 : Reference solution
        self.z_ = self.init_z()
    
    
    # TODO
    # Step 2
    def update(self):
        for i in range(self.N_):
            # Step 2.1 : Reproduction
            k,l = self.select_mating(i)
            x_k, x_l = self.population[k], self.population[l]
            y = self.generate_solution(x_k,x_l)
            
            # Step 2.2 : Improvement
            y_clean = self.repair(y)
            
            # Step 2.3 : Update of
            
            # Step 2.4 : Update of neighboring solutions
            
            # Step 2.5 Update of EP (External Population)

    # TODO
    # Step 3
    def execute(self):
        while not self.is_criterion_met():
            self.update()
    
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
        return [0 for i in range(len(self.weights))]
    
    # TODO
    def init_pop(self):
        return [np.array([0 for _ in range(self.dim)]) for _ in range(self.N_)]
    
    # TODO
    def init_z(self):
        return [0 for i in range(self.dim)]
    
    # TODO
    def is_criterion_met(self):
        return True

    