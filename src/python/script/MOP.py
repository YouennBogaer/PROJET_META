class MOP():
    """
    Represents a multi-objective problem MOP with its dimension, objective functions and chromosome.
    """
    def __init__(self, obj_functions, dim_x, bounds_x):
        """

        Args:
            obj_functions (List[Func]): List of objective functions
            dim_x (int): number of variables to the chromosome x (dimension of x)
            bounds_x (List[(min,max)]): list of bounds for each variable of x
        """
        self.functions = obj_functions
        self.dim = len(self.functions)
        self.dim_x = dim_x
        self.bounds_x = bounds_x
        
    def evaluate(self, x):
        """
        Compute the evaluation of x by the objective functions.
        Args:
            x (vector): solution vector to evaluate

        Returns:
            List[float]: Vector F of values of each objectives functions for input solution x.
        """
        return [f(x) for f in self.functions]
    
    def getFunctions(self):
        return self.functions
    
    def getDim(self):
        return self.dim