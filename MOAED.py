import numpy as np

class MOEAD():
    def __init__(self, stop_criterion, weights, T_neighborhood):
        self.criterion = stop_criterion
        self.weights = weights
        self.T_nghb = T_neighborhood
        
        self.N_subs = len(self.weights)
        
        # Initialize as empty
        self.EP = []
        
        self.B_nghb = self.neighborhood()
    
    #TODO
    def neighborhood(self):
        """
        Returns the T closest weight vectors of any weight vector based on euclidian distance.
        Output should be a list B_nghb of length self.N_subs.
        """
        return [None for i in range(len(self.weights))]