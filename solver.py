import math
import time
from collections import defaultdict, namedtuple
from functools import cache
from itertools import combinations, permutations

import numpy as np


class Solver:
    """
    The parent class of a Solver for finding the optimal portorder.
    The solvers take an instance of one of the models from mappings.py as input.
    The main method solve() then returns the best solution it found, the search trajectory
        and the time in seconds it needed to land on the optimum.

    Attributes:
        model (Model): An instance of one of the models from mappings.py
        random_seed (int): Random seed for any randomness used in solver.
        min_size (int):
            The minimum number of ports an instance needs to have to be solved approximately.
            For any instance smaller than that the solver goes through all possible solutions.
        max_runtime (int):
            Maximum runtime in seconds.
            When this maximum is reached, solvers finish current iteration before terminating.
    """

    def __init__(
        self,
        model,
        random_seed=1111,
        min_size=8,
        max_runtime=600,
    ):
        self.model = model
        self.min_size = min_size
        self.max_runtime = max_runtime
        self.index_combinations_2opt = list(
            combinations(range(self.model.instance.numOfPorts), 2)
        )
        self.index_combinations_3opt = list(
            combinations(range(self.model.instance.numOfPorts - 1), 3)
        )
        self.rng = np.random.default_rng(seed=random_seed)
        self.start_time = time.process_time()
        self.time_until_optimum = time.process_time() - self.start_time
        self.obj_value_timeline = list()

    @cache
    def get_solution(self, x):
        """
        Cached method for calling the generate_solution() method of the model.

        Args:
            x (tuple): Portorder, given as a tuple starting and ending at 0

        Returns:
            ModelSolution: Optimal solution for portorder x
        """
        return self.model.generate_solution(x=x)

    def generate_solution_random(self):
        """
        Randomly create any solution.

        Returns:
            ModelSolution: Randomly generated solution.
        """
        x = np.concatenate(
            [
                [0],
                self.rng.permutation(np.arange(1, self.model.instance.numOfPorts - 1)),
                [0],
            ]
        )
        return self.get_solution(tuple(x))

    def generate_initial_solution_greedy(self):
        """
        Incrementally build initial solution greedily by only considering distance of ports.

        Returns:
            ModelSolution: Constructed initial solution.
        """
        if self.model.distribution == "deterministic":
            # distances without last row and column (for readability)
            distances = np.delete(
                np.delete(self.model.distance_matrix, -1, axis=1), -1, axis=0
            )
        else:
            distances = self.model.distance_matrix
        x = np.zeros(self.model.instance.numOfPorts, dtype=int)
        ports_visited = {0}
        for i in range(1, self.model.instance.numOfPorts - 1):
            current_port = x[i - 1]
            nearest_port = min(
                [
                    (port, distances[current_port][port])
                    for port in range(1, self.model.instance.numOfPorts - 1)
                    if port not in ports_visited
                ],
                key=lambda k: k[1],
            )[0]
            x[i] = nearest_port
            ports_visited.add(nearest_port)
        return self.get_solution(tuple(x))

    def generate_2_opt_neighbor(self, sol, indices):
        """
        Generate a neighboring portorder with respect to a 2-opt neighborhood structure.

        Args:
            sol (ModelSolution): Current solution
            indices (iterable): list, array or tuple of length 2, indicating the eliminated arcs.

        Returns:
            np.ndarray: neighboring portorder
        """
        # reverse entire order from index to index
        y = sol.x[:-1].copy()
        y[indices[0] : indices[1] + 1] = np.flipud(y[indices[0] : indices[1] + 1])
        y_dupl = np.append(y, y)
        i_0 = np.argwhere(y_dupl == 0)
        y = y_dupl[
            i_0[0][0] : i_0[1][0] + 1
        ]  # order from y, but starting and ending at 0
        return y

    def generate_3opt_neighborhood(self, sol, subset_combinations):
        """
        Generate a list of all neighbors with respect to a 3-opt neighborhood.

        Args:
            sol (ModelSolution): current Solution
            subset_combinations (iterable):
                array or list containing the index-combinations of length 3 that
                indicate where to break and then reconnect the arcs.

        Returns:
            list: list of all neighboring portorders
        """
        if len(sol.x) < 5:
            raise ValueError("Instance must have at least 4 ports to apply 3-opt.")
        neighborhood = list()
        nodes = sol.x[1:-1].copy()
        for i, j, k in subset_combinations:
            segm1 = nodes[:i]
            segm2 = nodes[i:j]
            segm3 = nodes[j:k]
            segm4 = nodes[k:]
            # Apply all 7 possible 3-opt reconnections that result in a new valid portorder
            new_tours = [
                np.concatenate([[0], segm1, segm2[::-1], segm3, segm4, [0]]),
                np.concatenate([[0], segm1, segm3[::-1], segm2[::-1], segm4, [0]]),
                np.concatenate([[0], segm1, segm2, segm3[::-1], segm4, [0]]),
                np.concatenate([[0], segm1, segm3[::-1], segm2, segm4, [0]]),
                np.concatenate([[0], segm1, segm2[::-1], segm3[::-1], segm4, [0]]),
                np.concatenate([[0], segm1, segm3, segm2[::-1], segm4, [0]]),
                np.concatenate([[0], segm1, segm3, segm2, segm4, [0]]),
            ]
            for ii, tour in enumerate(new_tours):
                if (len(neighborhood) == 0) or (
                    not np.any(np.all(neighborhood == tour, axis=1))
                ):
                    neighborhood.append(tour)
        return neighborhood

    def generate_3opt_neighbor_random(self, sol):
        """
        Randomly generate a single neighboring solution with respect to a 3-opt
            neighborhood structure.

        Args:
            sol (ModelSolution): Current Solution

        Returns:
            ModelSolution: Generated neighboring solution
        """
        neighborhood = self.generate_3opt_neighborhood(
            sol=sol,
            subset_combinations=[self.rng.choice(self.index_combinations_3opt)],
        )
        return self.get_solution(tuple(self.rng.choice(neighborhood)))

    def solve_exact(self):
        """
        Go through all possible solutions and select the best one.

        Returns:
            ModelSolution: the global optimum
            list: empty list, included for uniformity with approximate solve() methods
            int: time until optimum given as negative to indicate use of exact solver
        """
        assert self.model.instance.numOfPorts <= self.min_size + 1
        solutions = []
        for perm in permutations(range(1, self.model.instance.numOfPorts - 1)):
            if (time.process_time() - self.start_time) > self.max_runtime:
                break
            x = np.concatenate([[0], perm, [0]])
            solutions.append(self.get_solution(tuple(x)))
        return min(solutions, key=lambda sol: sol.obj_value), [], -1

    def solve(self):
        """
        Placeholder method for the required method solve that subclasses have to implement.
        Implemented method in subclass should return three elements:

        Returns:
            ModelSolution: The best solution found
            list: The search trajectory (2Dlist of encountered objective values and their timestamp)
            int: time until optimum in seconds

        Raises:
            NotImplementedError: Method should be implemented in subclass.
        """
        raise NotImplementedError("A method solve() needs to be implemented")


class LSSolver(Solver):
    """
    The parent class for heuristic neighborhood search techniques.
    Includes all functionality for a one-time executing of a local search,
    excluding the method for generating a neighbor

    Args:
        Solver (_type_): _description_
    """

    def generate_neighbor(self):
        """
        Placeholder method for the required method generate_neighbor that subclasses have to implement.

        Returns:
            ModelSolution: selected neighboring solution

        Raises:
            NotImplementedError: Method should be implemented in subclass.
        """
        raise NotImplementedError(
            "A method generate_neighbor() needs to be implemented"
        )

    def new_sol_best(self):
        """
        Log timestamp and objective value whenever an improving solution is encountered.
        """
        self.time_until_optimum = time.process_time() - self.start_time
        self.obj_value_timeline.append([self.s_curr.obj_value, self.time_until_optimum])

    def solve(self):
        """
        Find a locally optimal solution with respect to some neighborhood structure.

        Returns:
            ModelSolution: The best solution found
            list: The search trajectory (2Dlist of encountered objective values and their timestamp)
            int: time until optimum in seconds
        """
        if self.model.instance.numOfPorts <= self.min_size:
            return self.solve_exact()
        self.s_curr = self.generate_initial_solution_greedy()
        self.new_sol_best()
        improvement = True
        while improvement and (
            (time.process_time() - self.start_time) <= self.max_runtime
        ):
            s_new = self.generate_neighbor()
            if (s_new.obj_value < self.s_curr.obj_value) and not (
                math.isclose(s_new.obj_value, self.s_curr.obj_value)
            ):
                self.s_curr = s_new
                self.new_sol_best()
            else:
                improvement = False
        return (self.s_curr, self.obj_value_timeline, self.time_until_optimum)


class LSSolverSteepest3opt(LSSolver):
    def generate_neighbor(self):
        """
        Generating mechanism for selecting a solution from a neighborhood.
        Uses steepest descent and a 3-opt neighborhood structure.

        Returns:
            ModelSolution: Selected neighboring solution
        """
        neighborhood = [
            self.get_solution(x=tuple(x))
            for x in self.generate_3opt_neighborhood(
                sol=self.s_curr, subset_combinations=self.index_combinations_3opt
            )
        ]
        return min(neighborhood, key=lambda sol: sol.obj_value)


class LSSolverFirst2Hopt(LSSolver):
    def generate_neighbor(self):
        """
        Generating mechanism for selecting a solution from a neighborhood.
        Uses first descent and a 2H-opt neighborhood structure.

        Returns:
            ModelSolution: Selected neighboring solution
        """
        # search through 2-opt neighborhood
        self.rng.shuffle(self.index_combinations_2opt)
        for indices in self.index_combinations_2opt:
            new_x = tuple(
                self.generate_2_opt_neighbor(sol=self.s_curr, indices=indices)
            )
            sol_new = self.get_solution(x=new_x)
            if (sol_new.obj_value < self.s_curr.obj_value) and not (
                math.isclose(sol_new.obj_value, self.s_curr.obj_value)
            ):
                return sol_new
        # search through insertion neighborhood
        x = self.s_curr.x.copy()
        for i in range(1, len(x) - 1):
            for j in range(1, len(x) - 1):
                if i != j:
                    new_x = tuple(np.insert(np.delete(x, i), j, x[i]))
                    sol_new = self.get_solution(x=new_x)
                    if (sol_new.obj_value < self.s_curr.obj_value) and not (
                        math.isclose(sol_new.obj_value, self.s_curr.obj_value)
                    ):
                        return sol_new
        return self.s_curr


# general namedtuple logging information about the history of a solution.
# includes
# rep_count: how often has solution been encountered?
# last_seen: at which iterations was it last encountered?
# Used in TabuSolver
SolutionHistory = namedtuple(
    typename="SolutionsHistory", field_names=["rep_count", "last_seen"]
)


class TabuSolver(Solver):
    """A solver using reactive tabu search for trying to find a global optimum."""

    def __init__(
        self,
        model,
        random_seed=1111,
        min_size=8,
        max_runtime=600,
        init_solution_type="greedy",
        max_iter=300,
        reactive_increase=1.2,
        reactive_decrease=0.95,
        neighborhood_structure="3-opt",
        T0=1,
        descent="steepest",
    ):
        super().__init__(
            model=model,
            random_seed=random_seed,
            min_size=min_size,
            max_runtime=max_runtime,
        )
        self.init_solution_type = init_solution_type
        self.set_sol_0()
        self.max_iter = max_iter
        self.reactive_increase = reactive_increase
        self.reactive_decrease = reactive_decrease
        self.descent = descent
        self.T0 = T0
        self.T = self.T0  # prohibition parameter (tabu list length)
        self.neighborhood_structure = neighborhood_structure
        self.moving_average = 0
        self.j = 0  # counter iterations without change of T
        assert self.sol_0.flow_valid
        self.sol_curr = self.sol_0
        self.sol_best = self.sol_0
        self.i = 0  # counter iterations without improvement
        self.t = 0  # general iteration counter
        self.sol_history = defaultdict(
            lambda: SolutionHistory(rep_count=0, last_seen=-np.inf)
        )
        self.sol_history[tuple(self.sol_0.x)] = SolutionHistory(
            rep_count=1, last_seen=self.t
        )
        self.obj_value_timeline.append(
            [self.sol_0.obj_value, time.process_time() - self.start_time]
        )

    def set_sol_0(self):
        """
        Set initial solution in specified manner.

        Raises:
            NotImplementedError: can only handle method greedy or random
        """
        if self.init_solution_type == "greedy":
            self.sol_0 = self.generate_initial_solution_greedy()
        elif self.init_solution_type == "random":
            self.sol_0 = self.generate_solution_random()
        else:
            raise NotImplementedError("only supported values: greedy, random")

    def create_neighborhood(self):
        """
        Generate neighborhood for specified neighborhood structure.

        Raises:
            NotImplementedError: can only handle 3-opt neighborhood structure.

        Returns:
            list: Neighboring portorders
        """
        if self.neighborhood_structure == "3-opt":
            return self.generate_3opt_neighborhood(
                sol=self.sol_curr, subset_combinations=self.index_combinations_3opt
            )
        else:
            raise NotImplementedError("only supported neighborhood structure: 3-opt")

    def get_allowed_neighborhood(self):
        """
        Generate allowed neighborhood, without solutions on tabu list.

        Returns:
            list: Allowed neighboring solutions
        """
        # tabu-filter
        neighborhood = [
            neighbor
            for neighbor in self.create_neighborhood()
            if self.sol_history[tuple(neighbor)].last_seen < (self.t - self.T)
        ]
        # solutions instead of portorders
        allowed_neighborhood = [self.get_solution(tuple(x)) for x in neighborhood]
        return allowed_neighborhood

    def choose_from_neighborhood(self, neighborhood):
        """
        Generating mechanism for selecting a neighbor from a given (allowed) neighborhood.
        Also contains React-mechanism, adjusting the prohibition parameter T based on the search history.

        Args:
            neighborhood (list): List of (allowed) neighbors

        Raises:
            NotImplementedError: can only handle steepest descent.
        """
        # choose best neighbor
        if self.descent == "steepest":
            self.sol_curr = min(neighborhood, key=lambda sol: sol.obj_value)
        else:
            raise NotImplementedError("descent not valid")

        self.sol_history[tuple(self.sol_curr.x)] = SolutionHistory(
            rep_count=(
                sol_hist_prev := self.sol_history[tuple(self.sol_curr.x)]
            ).rep_count
            + 1,
            last_seen=self.t,
        )
        # React
        if sol_hist_prev.rep_count > 0:
            rep_cycle_length = self.t - sol_hist_prev.last_seen
            self.moving_average = 0.1 * rep_cycle_length + 0.9 * self.moving_average

            if self.neighborhood_structure == "3-opt":
                length_limit = len(self.index_combinations_3opt) - 2
            elif self.neighborhood_structure == "2-opt":
                length_limit = len(self.index_combinations_2opt) - 2
            else:
                length_limit = length_limit = 2 * len(self.index_combinations_2opt) - 2
            self.T = min(
                max(
                    self.T * self.reactive_increase,
                    self.T + 1,
                ),
                length_limit,
            )
            self.j = 0
        if self.j > self.moving_average:
            self.T = max(self.T * self.reactive_decrease, 1)
            self.j = 0

    def new_sol_best(self):
        """
        Log solution and timestamp when a new globally best solution has been encountered.
        """
        self.sol_best = self.sol_curr
        self.time_until_optimum = time.process_time() - self.start_time
        self.i = 0

    def logging(self):
        """
        Log objective value after an iteration, regardless of improvement of global optimum.
        """
        self.obj_value_timeline.append(
            [
                self.sol_curr.obj_value,
                time.process_time() - self.start_time,
            ]
        )

    def solve(self):
        """
        Attempt to find globally optimal solution using reactive tabu search.

        Returns:
            ModelSolution: The best solution found
            list: The search trajectory (2Dlist of encountered objective values and their timestamp)
            int: Time until optimum in seconds
        """
        if self.model.instance.numOfPorts <= self.min_size:
            return self.solve_exact()
        while (self.i <= self.max_iter) and (
            (time.process_time() - self.start_time) <= self.max_runtime
        ):
            allowed_neighborhood = self.get_allowed_neighborhood()
            if len(allowed_neighborhood) < 1:
                break
            self.choose_from_neighborhood(neighborhood=allowed_neighborhood)
            if self.sol_curr.obj_value < self.sol_best.obj_value:
                self.new_sol_best()
            else:
                self.i += 1
            self.j += 1
            self.t += 1
            self.logging()
        return (
            self.sol_best,
            self.obj_value_timeline,
            self.time_until_optimum,
        )


class ILSSolver(Solver):
    """A solver using reactive iterated local search for trying to find a global optimum."""

    def __init__(
        self,
        model,
        random_seed=1111,
        min_size=8,
        max_runtime=600,
        init_solution_type="greedy",
        max_iter=30,
        descent="first",
        loc_search="2H-opt",
        acc_crit="better",
        perturb="3-opt",
        perturb_ntimes=3,
    ):
        super().__init__(
            model=model,
            random_seed=random_seed,
            min_size=min_size,
            max_runtime=max_runtime,
        )
        self.init_solution_type = init_solution_type
        self.iter = 0  # counter iterations without improvement
        self.max_iter = max_iter
        self.descent = descent
        self.loc_search = loc_search
        self.loc_search_method = self.loc_search
        self.acc_crit = acc_crit
        self.perturb = perturb
        self.perturb_ntimes = perturb_ntimes
        self.history = []
        self.loc_min_history = []
        self.obj_value_history = []
        self.counter = 0  # general iteration counter

    def get_solution(self, x):
        """
        Uncached method for calling the generate_solution() method of the model.

        Args:
            x (tuple): Portorder, given as a tuple starting and ending at 0

        Returns:
            ModelSolution: Optimal solution for portorder x
        """
        return self.model.generate_solution(x=x)

    def add_sol_to_history(self, sol):
        """
        Log an encountered solution, its objective value and timestamp

        Args:
            sol (ModelSolution): Solution encountered
        """
        self.history.append(tuple(sol.x))
        self.obj_value_history.append(
            [sol.obj_value, time.process_time() - self.start_time]
        )

    def set_sol_0(self):
        """
        Set initial solution in specified manner.

        Raises:
            NotImplementedError: can only handle method greedy or random
        """
        if self.init_solution_type == "greedy":
            self.sol_0 = self.generate_initial_solution_greedy()
        elif self.init_solution_type == "random":
            self.sol_0 = self.generate_solution_random()
        else:
            raise NotImplementedError("only supported values: greedy, random")
        self.sol_curr = self.sol_0
        self.sol_best = self.sol_0
        self.sol_temp = self.sol_0
        self.add_sol_to_history(self.sol_0)

    def local_search_2Hopt_first_descent(self):
        """
        Perform Local Search using first descent with a 2H-opt neighborhood structure
            until no improving solution in neighborhood.

        Returns:
            ModelSolution: Encountered local optimum
        """
        sol_best_local = self.sol_temp
        improvement = True
        while improvement:
            improvement = False
            # search through 2-opt neighborhood
            self.rng.shuffle(self.index_combinations_2opt)
            for indices in self.index_combinations_2opt:
                new_x = tuple(
                    self.generate_2_opt_neighbor(sol=sol_best_local, indices=indices)
                )
                if new_x not in self.history:
                    sol_new = self.get_solution(new_x)
                    if (sol_new.obj_value < sol_best_local.obj_value) and not (
                        math.isclose(sol_new.obj_value, sol_best_local.obj_value)
                    ):
                        sol_best_local = sol_new
                        self.add_sol_to_history(sol_new)
                        improvement = True
                        break
            if not improvement:
                # search through insertion neighborhood
                x = sol_best_local.x.copy()
                for i in range(1, len(x) - 1):
                    for j in range(1, len(x) - 1):
                        if i != j:
                            new_x = tuple(np.insert(np.delete(x, i), j, x[i]))
                            if new_x not in self.history:
                                sol_new = self.get_solution(new_x)
                                if (
                                    sol_new.obj_value < sol_best_local.obj_value
                                ) and not (
                                    math.isclose(
                                        sol_new.obj_value, sol_best_local.obj_value
                                    )
                                ):
                                    sol_best_local = sol_new
                                    self.add_sol_to_history(sol_new)
                                    improvement = True
                                    break
                    if improvement:
                        break
        return sol_best_local

    def local_search(self):
        """
        Select the method of local search to perform.

        Raises:
            NotImplementedError: Only first 2h-opt descent implemented.

        Returns:
            ModelSolution: Local Optimum
        """
        if (self.loc_search_method == "2H-opt") and (self.descent == "first"):
            sol_best_local = self.local_search_2Hopt_first_descent()
        else:
            raise NotImplementedError("invalid loc_search - descent combination")
        self.loc_min_history.append(sol_best_local)
        return sol_best_local

    def perturbation(self):
        """
        Perturbate current solution with the goal of leaving attraction basin.

        Raises:
            NotImplementedError: Can only handle 3-opt perturbation

        Returns:
            ModelSolution: Perturbed solution
        """
        sol_new = self.sol_curr
        # perturb n times
        for _ in range(math.ceil(self.perturb_ntimes)):
            if self.perturb == "3-opt":
                sol_new = self.generate_3opt_neighbor_random(sol=sol_new)
            else:
                raise NotImplementedError("perturb method invalid")
        self.add_sol_to_history(sol_new)
        return sol_new

    def acceptance_crit(self):
        """
        Decide if encountered solution becomes new current solution.

        Raises:
            NotImplementedError: Can only handle techniques better and random walk

        Returns:
            ModelSolution: Either new or old solution
        """
        if self.acc_crit == "better":
            sol_best_local = min(
                [self.sol_curr, self.sol_new], key=lambda sol: sol.obj_value
            )
        elif self.acc_crit == "random_walk":
            sol_best_local = self.sol_new
        else:
            raise NotImplementedError("Only accepted values: better, random_walk")
        return sol_best_local

    def new_best_sol_found(self):
        """
        Log newly encountered best solution and timestamp.
        """
        self.sol_best = self.sol_curr
        self.time_until_optimum = time.process_time() - self.start_time
        # no need to append to search trajectory, that happens elsewhere.
        self.iter = 0

    def solve(self):
        """
        Attempt to find globally optimal solution using iterative local search.

        Returns:
            ModelSolution: The best solution found
            list: The search trajectory (2Dlist of encountered objective values and their timestamp)
            int: Time until optimum in seconds
        """
        if self.model.instance.numOfPorts <= self.min_size:
            return self.solve_exact()
        self.set_sol_0()
        self.sol_curr = self.local_search()
        self.new_best_sol_found()
        while (self.iter <= self.max_iter) and (
            (time.process_time() - self.start_time) <= self.max_runtime
        ):
            self.counter += 1
            self.sol_temp = self.perturbation()
            self.sol_new = self.local_search()
            self.sol_curr = self.acceptance_crit()
            if math.isclose(self.sol_curr.obj_value, self.sol_best.obj_value):
                self.iter += 1
            elif self.sol_curr.obj_value < self.sol_best.obj_value:
                self.new_best_sol_found()
        return (
            self.sol_best,
            self.obj_value_history,
            self.time_until_optimum,
        )
