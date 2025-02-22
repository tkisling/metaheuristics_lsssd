import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Schedule:
    """
    Supplementary dataclass.
    A schedule contains only w, tau_s, tau_e, tau_a in Model 1.
    In Models 2 and 3 the sailing times also get relevant.
    There it is still possible to define a schedule without sailing times.
    """

    w: np.ndarray = field(repr=False)
    tau_s: np.ndarray = field(repr=False)
    tau_e: np.ndarray = field(repr=False)
    tau_a: np.ndarray = field(repr=False)
    sailing_times: np.ndarray = field(repr=False, default=None)


@dataclass(order=True)
class ModelSolution:
    """
    The parent class of a solution, without the specifics of any of the 3 models.
    The model solutions inherit from this class.
    An instance of this class contains choices for all relevant variables for the respective model.
    The schedule is given as input, the rest is generally calculated here.
    It is possible to choose to provide the demand flow as input to save computation time.
    """

    obj_value: float = field(init=False)
    model: "Model" = field(repr=False)
    x: np.ndarray
    schedule: "Schedule" = field(repr=False)
    flow: np.ndarray = field(repr=False, default=None)
    flow_valid: bool = field(repr=False, default=None)

    def __post_init__(self):
        # dont need the flow with every intermediate solution and
        # can sometimes skip that calculation (helpful for model 2&3)
        if self.flow is None:
            self.flow, self.flow_valid = self.generate_demand_flow()
        self.obj_value = self.objective_function()

    def generate_demand_flow(self):
        """
        The flow of demands on all arcs.
        Returned flow is given as numpy array in format [[d1,...dn],...,[d1,...,dn]] with each
        subarray representing the load when leaving the respective port.

        Returns:
            np.darray: flow of each demand leaving each port
            np.bool: whether or not the flow is valid based on the vessel's capacity
        """
        flow = np.array(
            [
                np.zeros(self.model.instance.numOfDemands)
                for _ in range(self.model.instance.numOfPorts - 1)
            ]
        )
        x_dupl = np.append(self.x[:-1], self.x[:-1])
        for demand, (source, destination, amount) in enumerate(
            zip(
                self.model.instance.demandSource,
                self.model.instance.demandDestination,
                self.model.instance.demandAmount,
            )
        ):
            x_from_source = x_dupl[np.argwhere(x_dupl == source - 1)[0][0] :]
            x_source_to_destination = x_from_source[
                : (np.argwhere(x_from_source == destination - 1)[0][0])
            ]
            for port in x_source_to_destination:
                flow[port][demand] += amount
        flow_validity = np.all(
            np.array([np.sum(port) for port in flow]) <= self.model.instance.capacity
        )
        return flow, flow_validity

    def objective_function(self):
        """
        Placeholder method for the required method objective_function that subclasses have to implement.
        Implemented method in subclass should return the value of the respective objective function as a float.
        It should also include a penalty term for the case that a demand flow is not valid.
        Raises:
            NotImplementedError: Method should be implemented in subclass.
        """
        raise NotImplementedError("No method objective_function() was defined")


@dataclass
class Model1Solution(ModelSolution):
    """The solution class for Model 1."""

    model: "Model1" = field(repr=False)

    def objective_function(self):
        return (
            self.model.instance.charterCost * self.schedule.w[-1]
            + np.sum(
                [
                    self.model.instance.fixedSailingCost[self.x[i - 1], self.x[i]]
                    for i in range(1, self.model.instance.numOfPorts)
                ]
            )
            + (not self.flow_valid) * np.nan_to_num(np.inf)
        )


@dataclass
class Model2Solution(Model1Solution):
    """The solution class for Model 2."""

    model: "Model2" = field(repr=False)

    def get_sailing_cost(self, port1, port2, sailing_time):
        """
        Calculate the bunker cost gamma_{j-1,j} for sailing from p_{j-1} to p_j
            in specified time rho_{j-1,j}.
        Use cubic formula instead of linear approximation.

        Args:
            port1 (int): origin port
            port2 (int): destination port
            sailing_time (float): sailing time rho, validity not verified in this method

        Returns:
            float: Calculated bunker cost.
        """
        return (
            (self.model.instance.sailingTime[port1, port2] / sailing_time) ** 3
            * (self.model.instance.designConsumption / 24)
            * self.model.fuel_price
            * sailing_time
        )

    def objective_function(self):
        return (
            self.model.instance.charterCost * self.schedule.w[-1]
            + np.sum(
                [
                    self.get_sailing_cost(
                        port1=self.x[i - 1],
                        port2=self.x[i],
                        sailing_time=self.schedule.sailing_times[i],
                    )
                    for i in range(1, self.model.instance.numOfPorts)
                ]
            )
            + (not self.flow_valid) * np.nan_to_num(np.inf)
        )


@dataclass
class Model3Solution(Model2Solution):
    """The solution class for Model 3."""

    model: "Model3" = field(repr=False)
    delta: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.delta = self.get_delta()
        self.y = self.get_y()
        super().__post_init__()

    def get_delta(self):
        """
        Calculate for each demand if in the portorder x the source is before the destination
        or if it is the other way around.

        Returns:
            np.darray: bool for each demand, 0 if source before dest, 1 otherwise
        """
        return np.array(
            [
                (
                    np.where(self.x[:-1] == self.model.instance.demandSource[k] - 1)
                    > np.where(
                        self.x[:-1] == self.model.instance.demandDestination[k] - 1
                    )
                )[0]
                for k in range(self.model.instance.numOfDemands)
            ]
        )

    def get_y(self):
        """
        Calculate for each demand if it is possible to transport it under the active schedule.

        Returns:
            np.darray: bool for each demand, True if demand is transported
        """
        tau_s_dest = np.array(
            [
                self.schedule.tau_s[
                    np.where(self.x == self.model.instance.demandDestination[k] - 1)
                ][0]
                for k in range(self.model.instance.numOfDemands)
            ]
        )
        tau_e_source = np.array(
            [
                self.schedule.tau_e[
                    np.where(self.x == self.model.instance.demandSource[k] - 1)
                ][0]
                for k in range(self.model.instance.numOfDemands)
            ]
        )
        transit_time = (
            tau_s_dest - tau_e_source + 168 * self.schedule.w[-1] * self.delta
        )
        return (transit_time <= self.model.instance.demandTransitTime) | np.isclose(
            transit_time, self.model.instance.demandTransitTime
        )

    def generate_demand_flow(self):
        """
        Updated demand flow with respect to y.

        Returned flow is given as numpy array in format [[d1,...dn],...,[d1,...,dn]] with each
        subarray representing the load when leaving the respective port.

        Returns:
            np.darray: flow of each demand leaving each port
            np.bool: whether or not the flow is valid based on the vessel's capacity
        """
        flow = np.array(
            [
                np.zeros(self.model.instance.numOfDemands)
                for _ in range(self.model.instance.numOfPorts - 1)
            ]
        )
        x_dupl = np.append(self.x[:-1], self.x[:-1])
        for demand, (source, destination, amount) in enumerate(
            zip(
                self.model.instance.demandSource,
                self.model.instance.demandDestination,
                self.model.instance.demandAmount,
            )
        ):
            if self.y[demand]:
                x_from_source = x_dupl[np.argwhere(x_dupl == source - 1)[0][0] :]
                x_source_to_destination = x_from_source[
                    : (np.argwhere(x_from_source == destination - 1)[0][0])
                ]
                for port in x_source_to_destination:
                    flow[port][demand] += amount
        flow_validity = np.all(
            np.array([np.sum(port) for port in flow]) <= self.model.instance.capacity
        )
        return flow, flow_validity

    def objective_function(self):
        return (-1) * (
            np.dot(
                self.model.instance.demandRevenue * self.model.instance.demandAmount,
                self.y,
            )
            - self.model.instance.charterCost * self.schedule.w[-1]
            - np.sum(
                [
                    self.get_sailing_cost(
                        port1=self.x[i - 1],
                        port2=self.x[i],
                        sailing_time=self.schedule.sailing_times[i],
                    )
                    for i in range(1, self.model.instance.numOfPorts)
                ]
            )
        ) + (not self.flow_valid) * np.nan_to_num(np.inf)


class Model:
    """
    The parent class of a Model.
    The models inherit from this class.
    An instance of this class represents one selection of distribution and service level
        for one LINERLIB-instance.
    The Model class is centered around the `generate_solution`-method, which maps
        a portorder x to its optimal ModelSolution.

    Attributes:
        instance (LsssdInstance): The LINERLIB-instance
        distribution (string): either genlog_3p, normal, or deterministic
        service_level (string): if distribution is not deterministic:
            either "0.7000", "0.9000", or "0.9500"
            otherwise not important, defaults to None
        ModelSolution (ModelSolution): corresponding solution class
        distance_matrix (np.darray): appropriate distance matrix for distribution and service level
        fuel_price (int): fuel price in dollar per ton
        min_speed (int): minimum possible speed of vessel in knots
        max_speed (int): maximum possible speed of vessel in knots
        max_sailing_time (np.darray): maximum possible sailing time on all arcs (when using min_speed)
    """

    def __init__(
        self,
        instance,
        distribution,
        service_level=None,
    ):
        self.instance = instance
        self.distribution = distribution
        self.service_level = service_level
        self.ModelSolution = ModelSolution
        self.distance_matrix = self.set_distance_matrix()
        self.fuel_price = 400
        if self.instance.vesselClass == "Post_panamax":
            self.min_speed = 12
            self.max_speed = 23
        elif self.instance.vesselClass == "Super_panamax":
            self.min_speed = 12
            self.max_speed = 22
        else:
            raise NotImplementedError("properties of given vesselClass not known")
        self.max_sailing_time = self.instance.sailingTime * (
            self.instance.designSpeed / self.min_speed
        )

    def set_distance_matrix(self):
        """
        Set the correct distance matrix for the chosen distribution and service level.

        Raises:
            NotImplementedError: if chosen distribution not genlog_3p, normal, or deterministic
            NotImplementedError:
                if not deterministic
                and chosen service level not "0.7000", "0.9000", or "0.9500"
        """

        if self.distribution == "deterministic":
            return self.instance.sailingTime
        elif self.distribution in {"genlog_3p", "normal"}:
            if self.service_level not in {"0.7000", "0.9000", "0.9500"}:
                raise NotImplementedError(
                    'only supported service levels: "0.7000", "0.9000", "0.9500"'
                )
            sailing_time_df = pd.read_csv(
                os.path.join(
                    "stochastic_data", f"{self.distribution}_{self.service_level}.csv"
                ),
                header=None,
            )
            return np.array(
                pd.crosstab(
                    index=sailing_time_df[0],
                    columns=sailing_time_df[1],
                    values=sailing_time_df[2],
                    aggfunc="mean",
                )
                .fillna(0)
                .loc[self.instance.ports[:-1], self.instance.ports[:-1]]
            )
        else:
            raise NotImplementedError(
                "only supported values: genlog_3p, normal, deterministic"
            )

    def get_sailing_time(self, port1, port2, speed):
        """
        calculate the required sailing time between two ports.

        Args:
            port1 (int): origin port
            port2 (int): destination port
            speed (float): travel speed

        Returns:
            float: sailing time rho_ij
        """
        if math.isclose(speed, self.instance.designSpeed):
            return self.distance_matrix[port1, port2]
        elif self.distribution == "deterministic":
            min_time = 0
        else:
            min_time = self.distance_matrix[port1, port2]
        return max(
            min_time,
            (
                self.instance.sailingTime[port1, port2]
                * (self.instance.designSpeed / speed)
            ),
        )

    def generate_schedule(self, x, speed=None):
        """
        Create optimal schedule for portorder x
        when traveling at constant speed on all arcs.

        Args:
            x (np.darray): portorder starting and ending at 0
            speed (float, optional): constant speed used on all arcs.
                Defaults to None. In that case, design speed of vessel is used.

        Returns:
            Schedule: instance of class Schedule
        """
        w = np.zeros(self.instance.numOfPorts, dtype=int)
        tau_s = np.zeros(self.instance.numOfPorts)
        tau_e = np.zeros(self.instance.numOfPorts)
        # tau_a[0] will stay 0 and is not important,
        # but readability is improved with same length for all tau
        tau_a = np.zeros(self.instance.numOfPorts)
        tau_s[0] = self.instance.timeWindowStart[0]
        tau_e[0] = self.instance.timeWindowEnd[0]
        for i in range(1, self.instance.numOfPorts):
            tau_a[i] = tau_e[i - 1] + self.get_sailing_time(
                port1=x[i - 1], port2=x[i], speed=speed
            )
            temp_arrival_weektime = tau_a[i] % 168
            temp_starttime = self.instance.timeWindowStart[x[i]]
            w[i] = (tau_a[i] // 168) + (
                (temp_arrival_weektime > temp_starttime)
                and not (math.isclose(temp_arrival_weektime, temp_starttime))
            )  # week of arrival at port i + (0 or 1)
            tau_s[i] = self.instance.timeWindowStart[x[i]] + 168 * w[i]
            tau_e[i] = self.instance.timeWindowEnd[x[i]] + 168 * w[i]
        return Schedule(w=w, tau_s=tau_s, tau_e=tau_e, tau_a=tau_a)

    def generate_solution(self, x):
        """
        Placeholder method for the required method generate_solution that subclasses have to implement.
        Implemented method in subclass should return an instance of self.ModelSolution

        Args:
            x (iterable): The portorder generally given as a tuple or np array starting and ending at 0

        Raises:
            NotImplementedError: Method should be implemented in subclass.
        """
        raise NotImplementedError("No method generate_solution(x) was defined")


class Model1(Model):
    """A class for mapping portorders to their optimal solution in Model 1."""

    def __init__(
        self,
        instance,
        distribution,
        service_level=None,
    ):
        super().__init__(
            instance=instance, distribution=distribution, service_level=service_level
        )
        self.ModelSolution = Model1Solution

    def generate_solution(self, x):
        """
        Map a portorder x to its optimal Model1Solution

        Args:
            x (np.ndarray): order of ports starting and ending at 0

        Returns:
            Model1Solution: solution
        """
        x = np.array(x)
        return self.ModelSolution(
            model=self,
            x=x,
            schedule=self.generate_schedule(
                x,
                speed=self.instance.designSpeed,
            ),
        )


class Model2(Model1):
    """A class for mapping portorders to their optimal solution in Model 2."""

    def __init__(
        self,
        instance,
        distribution,
        service_level=None,
    ):
        super().__init__(
            instance=instance, distribution=distribution, service_level=service_level
        )
        self.ModelSolution = Model2Solution

    def get_relaxed_sailing_time(self, x, schedule):
        """
        Given a schedule and a portorder, get the actually needed sailing time.
        That means relax the sailing time where possible without week-shift or order-change.
        The input schedule is assumed to be valid with respect to the vessels' max speed.

        Args:
            x (np.darray): order of ports starting and ending at 0
            schedule (Schedule): instance of class Schedule

        Returns:
            np.darray: sailing times for all arcs in x
            np.darray: arrival times tau_a according to new sailing times
        """
        relaxed_sailing_times = np.zeros(self.instance.numOfPorts)
        new_tau_a = np.zeros(self.instance.numOfPorts)
        # relaxed_sailing_times[0] and new_tau_a[0] not needed, but included for readability
        for i in range(1, self.instance.numOfPorts):
            relaxed_sailing_times[i] = min(
                self.max_sailing_time[x[i - 1], x[i]],
                (schedule.tau_s[i] - schedule.tau_e[i - 1]),
            )
            new_tau_a[i] = schedule.tau_e[i - 1] + relaxed_sailing_times[i]

        return relaxed_sailing_times, new_tau_a

    def generate_solution_max_speed(self, x):
        """
        Combine initial max_speed-schedule with appropriate sailing times
        and create full solution

        Args:
            x (np.darray): port order, starting and ending at 0

        Returns:
            ModelSolution: Solution of respective model
        """
        sched_0 = self.generate_schedule(x=x, speed=self.max_speed)
        ## relax traveling times where possible without weekshift
        required_sailing_times, new_tau_a = self.get_relaxed_sailing_time(
            x=x, schedule=sched_0
        )
        return self.ModelSolution(
            model=self,
            x=x,
            schedule=Schedule(
                w=sched_0.w,
                tau_s=sched_0.tau_s,
                tau_e=sched_0.tau_e,
                tau_a=new_tau_a,
                sailing_times=required_sailing_times,
            ),
        )

    def shift_weeks(self, x, current_solution, k):
        """
        Modify a given solution by shifting the schedule by one week for each port
        starting at the k'th port of the solution.

        Args:
            x (np.darray): port order, starting and ending at 0
            current_solution (ModelSolution): full solution
            k (int): port number at which the shifting starts

        Returns:
            ModelSolution: new solution
        """
        w = np.concat(
            (current_solution.schedule.w[:k], current_solution.schedule.w[k:] + 1)
        )
        tau_s = np.concat(
            (
                current_solution.schedule.tau_s[:k],
                current_solution.schedule.tau_s[k:] + 168,
            )
        )
        tau_e = np.concat(
            (
                current_solution.schedule.tau_e[:k],
                current_solution.schedule.tau_e[k:] + 168,
            )
        )
        required_sailing_times, new_tau_a = self.get_relaxed_sailing_time(
            x=x,
            schedule=Schedule(
                w=w, tau_s=tau_s, tau_e=tau_e, tau_a=current_solution.schedule.tau_a
            ),
        )
        return self.ModelSolution(
            model=self,
            x=x,
            schedule=Schedule(
                w=w,
                tau_s=tau_s,
                tau_e=tau_e,
                tau_a=new_tau_a,
                sailing_times=required_sailing_times,
            ),
            flow=current_solution.flow,
            flow_valid=current_solution.flow_valid,
        )

    def generate_solution(self, x):
        """
        Map a portorder x to its optimal Model2Solution

        Args:
            x (np.darray): order of ports starting and ending at 0

        Returns:
            Model2Solution: optimal solution for portorder x
        """
        x = np.array(x)
        current_solution = self.generate_solution_max_speed(x=x)
        k = 1
        while k <= self.instance.numOfPorts:
            new_solution = self.shift_weeks(x=x, current_solution=current_solution, k=k)
            if new_solution.obj_value < current_solution.obj_value:
                current_solution = new_solution
            else:
                k += 1
        return current_solution


class Model3(Model2):
    """A class for mapping portorders to their optimal solution in Model 3."""

    def __init__(
        self,
        instance,
        distribution,
        service_level=None,
    ):
        super().__init__(
            instance=instance, distribution=distribution, service_level=service_level
        )
        self.ModelSolution = Model3Solution

    def generate_solution(self, x):
        """
        Map a portorder x to its optimal Model3Solution

        Args:
            x (np.darray): order of ports starting and ending at 0

        Returns:
            Model3Solution: optimal solution for portorder x
        """
        x = np.array(x)
        max_speed_solution = self.generate_solution_max_speed(x=x)
        return self.shift_weeks(sol_start=max_speed_solution, k=1)

    def shift_weeks(self, sol_start, k):
        """
        Build tree-like structure to check all relevant schedules
        This is done in a recursive way.

        Args:
            sol_start (ModelSolution): solution on which the shifting is to be performed
            k (int): port number at which the shifting starts

        Returns:
            Model3Solution: shifted solution
        """
        solutions = []  # collect "finished" solutions
        stack = [(sol_start, k)]
        while stack:
            sol_current, k = stack.pop()
            if k > self.instance.numOfPorts:
                solutions.append(sol_current)
                continue
            w = np.concatenate(
                (sol_current.schedule.w[:k], sol_current.schedule.w[k:] + 1)
            )
            tau_s = np.concatenate(
                (sol_current.schedule.tau_s[:k], sol_current.schedule.tau_s[k:] + 168)
            )
            tau_e = np.concatenate(
                (sol_current.schedule.tau_e[:k], sol_current.schedule.tau_e[k:] + 168)
            )
            required_sailing_times, new_tau_a = self.get_relaxed_sailing_time(
                x=sol_start.x,
                schedule=Schedule(
                    w=w, tau_s=tau_s, tau_e=tau_e, tau_a=sol_current.schedule.tau_a
                ),
            )
            sol_new = self.ModelSolution(
                model=self,
                x=sol_start.x,
                schedule=Schedule(
                    w=w,
                    tau_s=tau_s,
                    tau_e=tau_e,
                    tau_a=new_tau_a,
                    sailing_times=required_sailing_times,
                ),
            )
            # Shift week for next port (always) and (if good) check same port for new solution
            if sol_new.obj_value < sol_current.obj_value:
                stack.append((sol_new, k))
            stack.append((sol_current, k + 1))
        # Return best solution found
        return min(solutions, key=lambda sol: sol.obj_value)
