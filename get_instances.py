import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LsssdInstance:
    """Class representing a single instance configuration"""

    name: str
    numOfPorts: int
    ports: np.ndarray
    timeWindowStart: np.ndarray
    timeWindowEnd: np.ndarray
    fixedSailingCost: np.ndarray
    sailingTime: np.ndarray
    numOfDemands: int
    demands: np.ndarray
    demandSource: np.ndarray
    demandDestination: np.ndarray
    demandAmount: np.ndarray
    demandRevenue: np.ndarray
    demandTransitTime: np.ndarray
    capacity: int
    charterCost: float
    vesselClass: str
    designSpeed: float
    designConsumption: float


def create_single_instance_df(instance_filename):
    """
    Supplementary function reading data from instance .txt files

    Args:
        instance_filename (str): Filename of instance configuration

    Returns:
        pd.DataFrame: df with only a single row, containing raw data
    """
    with open(os.path.join("instances", instance_filename), "r") as f:
        instance = f.read()
    instance_df = pd.DataFrame.from_dict(
        {
            element[0]: [element[1]]
            for element in [row.split(":") for row in instance.split("\n")]
            if len(element) == 2
        }
    )
    instance_df["name"] = instance_filename[:-4]
    return instance_df


def create_single_instance(instance_filename, relaxed=False):
    """
    Build an instance of the LsssdInstance class from a configuration .txt file.

    Args:
        instance_filename (str): Filename of instance configuration
        relaxed (bool, optional):
            Indicate, whether or not maximum demand travel times should be relaxed.
            Defaults to False.

    Returns:
        LsssdInstance: object holding all relevant data of a single configuration
    """
    instance_df = create_single_instance_df(instance_filename)
    if relaxed:
        demandTransitTime = np.array(
            [
                int(entry) * 1.5
                for entry in instance_df.loc[0, "demandTransitTime"].split(",")
            ]
        )
    else:
        demandTransitTime = np.array(
            [int(entry) for entry in instance_df.loc[0, "demandTransitTime"].split(",")]
        )

    instance = LsssdInstance(
        name=instance_df.loc[0, "name"] + relaxed * "_LL1.5",
        numOfPorts=int(instance_df.loc[0, "numOfPorts"]),
        ports=np.array(instance_df.loc[0, "ports"].split(",")),
        timeWindowStart=np.array(
            [float(entry) for entry in instance_df.loc[0, "timeWindowStart"].split(",")]
        ),
        timeWindowEnd=np.array(
            [float(entry) for entry in instance_df.loc[0, "timeWindowEnd"].split(",")]
        ),
        fixedSailingCost=np.array(
            [
                [float(entry) for entry in row.split(" ")]
                for row in instance_df.loc[0, "fixedSailingCost"].split(",")
            ]
        ),
        sailingTime=np.array(
            [
                [float(entry) for entry in row.split(" ")]
                for row in instance_df.loc[0, "sailingTime"].split(",")
            ]
        ),
        numOfDemands=int(instance_df.loc[0, "numOfDemands"]),
        demands=np.array(instance_df.loc[0, "demands"].split(",")),
        demandSource=np.array(
            [int(entry) for entry in instance_df.loc[0, "demandSource"].split(",")]
        ),
        demandDestination=np.array(
            [int(entry) for entry in instance_df.loc[0, "demandDestination"].split(",")]
        ),
        demandAmount=np.array(
            [int(entry) for entry in instance_df.loc[0, "demandAmount"].split(",")]
        ),
        demandRevenue=np.array(
            [float(entry) for entry in instance_df.loc[0, "demandRevenue"].split(",")]
        ),
        demandTransitTime=demandTransitTime,
        capacity=float(instance_df.loc[0, "capacity"]),
        charterCost=float(instance_df.loc[0, "charterCost"]),
        vesselClass=instance_df.loc[0, "vesselClass"],
        designSpeed=float(instance_df.loc[0, "designSpeed"]),
        designConsumption=float(instance_df.loc[0, "designConsumption"]),
    )
    return instance
