import numpy as np

class MOP():
    """
    Represents a multi-objective problem MOP with its dimension, objective functions and chromosome.
    """

    def __init__(self, obj_functions, dim_x, bounds_x, params):
        """
        Args:
            obj_functions (List[Func]): List of objective functions
            dim_x (int): number of variables to the chromosome x (dimension of x)
            bounds_x (List[(min,max)]): list of bounds for each variable of x
            params (dict): Parameters of the problem (M, mnv, etc.)
        """
        self.functions = obj_functions
        self.dim = len(self.functions)
        self.dim_x = dim_x
        self.bounds_x = bounds_x
        self.params = params

    def evaluate(self, x):
        """
        Compute the evaluation of x by the objective functions.
        Args:
            x (vector): solution vector to evaluate

        Returns:
            List[float]: Vector F of values of each objective function for input solution x.
        """
        return [f(x) for f in self.functions]

    def getFunctions(self):
        return self.functions

    def getDim(self):
        return self.dim

    def repair(self, y):
        """
        Repairs a chromosome to respect server capacity constraints.
        """
        # 1. Clip integers within defined bounds
        for i in range(len(y)):
            low, high = self.bounds_x[i]
            y[i] = int(np.clip(y[i], low, high))

        M = self.params['M']
        cap_max = self.params['mnv']
        usage = np.zeros(M + 1)

        # 2. Calculate current load and identify overflowing tasks
        tasks_overflowing = []
        for n in range(len(y) // 2):
            srv = int(y[2 * n])
            cpu = int(y[2 * n + 1])
            if srv > 0:  # If assigned to a MEC server
                if usage[srv] + cpu <= cap_max:
                    usage[srv] += cpu
                else:
                    # Capacity exceeded, mark task for reassignment
                    tasks_overflowing.append(n)

        # 3. Repair by reassigning tasks to available resources
        # Shuffle to avoid systematic bias toward specific tasks
        np.random.shuffle(tasks_overflowing)

        for n in tasks_overflowing:
            cpu = int(y[2 * n + 1])

            # Find the least loaded MEC server (indices 1 to M)
            # usage[1:] targets MECs. argmin gives relative index (0 = MEC 1)
            target_idx = np.argmin(usage[1:])
            target_srv = target_idx + 1

            if usage[target_srv] + cpu <= cap_max:
                # Reassign to the least loaded server if it fits
                y[2 * n] = target_srv
                usage[target_srv] += cpu
            else:
                # If even the least loaded server is full, reassign to Cloud
                y[2 * n] = 0

        return y