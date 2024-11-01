import random
from copy import deepcopy

import numpy as np

from control.agent import Agent, load_data_from_config
from control.passenger import Passenger
from control.task import Task
from env.comms_manager import CommsManager


class Mothership(Agent):

    def __init__(self, id: int, sim_data: dict, pssngr_list) -> None:
        super().__init__(id, sim_data)
        self.type = self.MOTHERSHIP




def generate_mothership_with_data(id, sim_data, pssngr_list) -> Mothership:
    """
    Create new Mothership
    """
    m = Mothership(id,
                   deepcopy(sim_data),
                   pssngr_list
                   )
    return m


def gen_mother_from_config(sim_data_fp,
                           pssngr_list,
                           rand_base=None) -> Mothership:
    sim_data = load_data_from_config(sim_data_fp, rand_base)

    return generate_mothership_with_data(sim_data["m_id"],
                                         sim_data,
                                         pssngr_list)
