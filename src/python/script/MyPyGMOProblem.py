import pygmo as pg

class MyPyGMOProblem:
    def __init__(self, obj_functions, dim, bounds):
        self.obj_functions = obj_functions
        self.dim = dim
        self.bounds = bounds

    def fitness(self, x):
        # PyGMO attend un vecteur de toutes les fonctions d'objectifs
        return [f(x) for f in self.obj_functions]

    def get_bounds(self):
        # On sépare les bornes min et max
        low = [b[0] for b in self.bounds]
        up = [b[1] for b in self.bounds]
        return (low, up)

    def get_nobj(self):
        return len(self.obj_functions)