
import math
from copy import deepcopy

import numpy as np
import yaml

from control.env_model import EnvironmentModel
from control.task import Task
from env.comms_manager import CommsManager, Message
from env.environment import Environment


class Agent:

    IDLE = 0
    TRAVELING = 1
    WORKING = 2
    MOTHERSHIP = 0
    PASSENGER = 1
    SUPPORT = 2

    def __init__(self, id: int, sim_data: dict) -> None:

        self.id = id
        self.sim_data = sim_data
        self.travel_remaining = 0

        self.agent_ids = []
        self.mothership_id = None
        self.passenger_list = None

        self.stored_reward_sum = 1
        self.last_msg_content = None

        self.new_states_to_eval = []

        self.event = True
        self.expected_event = True

        self.dead = False

        # Environment status
        self.env_dim_ranges = None
        self.env_model = self._initialize_model(sim_data["env_dims"])
        self.observation = []

        # Env Processing & Planning parameters
        self.position_mod_vector = None
        self.local_flow = []  # local flow
        self.energy_burn_rate = sim_data["energy_burn_rate"]

        self.base_loc = sim_data["base_loc"]
        self.location = sim_data["base_loc"]

        # Communication variables
        self.neighbors_status = (
            {}
        )  # dictionary of True/False reachable status, indexed by agent ID

        # Task-related variables
        self.task_dict = {}

        # Current action (Traveling/Working/Idle, Task ID)
        self.action = [self.IDLE, "Init"]
        
    def reset(self, env):
        self.task_dict = {}
        self._set_up_dim_ranges(env)
        self.sense_location_from_env(env)
        self.load_tasks_on_agent(env)
        
    def update_observation(self, env: Environment, step):
        """
        Get global observation of all agent locations relative to mothership?
        """
        obs_list = [] # 1D Array observation
        # Update own location
        self.sense_location_from_env(env)
        
        # Get mothership location
        m_loc = self._get_agent_loc_from_env(env, self.mothership_id)
        
        # Add agent positions (relative to mothership)
        for p in self.passenger_list:
            a_loc = self._get_agent_loc_from_env(env, p.id)
            a_rel = np.round(np.subtract(a_loc, m_loc), 1)[:2]
            if p.connected_to_M:
                a_state = np.concatenate((a_rel, [1.0]))
            else:
                a_state = np.concatenate((a_rel, [0.0]))
            obs_list.append(a_state)
        
        # Add task positions (relative to mothership)
        for task in env.task_dict.values():
            t_rel = np.round(np.subtract(task.location, m_loc),1)[:2]
            if task.complete:
                t_state = np.concatenate((t_rel, [1.0]))
            else:
                t_state = np.concatenate((t_rel, [0.0]))
            obs_list.append(t_state)
        
        # TODO (Later?) Potentially use flow observations
        # self._sense_flow_from_env(env)
        # self._apply_observations_to_model()
        
        # Store observation as numpy array
        self.observation = np.array(obs_list).flatten()
        # Add time step to observation
        self.observation = np.concatenate((self.observation, [step]))
        
    def compute_dense_reward(self, env: Environment):
      reward_list = []
      incomplete_list = []
      connected = 0
      #L_i(s) = 1/d_min * comm_i where d_min minimum distance to uncompleted reward
      #Make list of uncomplete tasks for computing
      for task in env.task_dict.values():
        if not task.complete:
          incomplete_list.append(task)
      
      #Iterate over all passengers
      for p in self.passenger_list:
        #Check if we have connected to the mothership
        if p.connected_to_M:
          connected = 1
        else:
          connected = 0
        
        #Get agent position
        a_loc = self._get_agent_loc_from_env(env, p.id)

        #Find distance to closest incomplete task
        for n,task_incomp in enumerate(incomplete_list):
          
          #Calculate Distance
          d_rel = np.linalg.norm((np.subtract(task.location, a_loc),1)[:2])
          
          #If first calc, set to min
          if n == 0:
            d_min = d_rel
          
          #if less, set new minimum
          if d_rel < d_min:
            d_min = d_rel
        
        #Now calculate reward and append to list
        L_i = 1/d_min*connected
        reward_list.append(L_i)
      
      return reward_list

    # ===============================================    
    # === TASK FUNCTIONS ===

    def _create_local_task(self, id: int, loc: tuple, work: int, reward: int, thresh: int):
        """Adds a task to this agent's task list"""
        if id not in self.task_dict.keys():
            self.task_dict[id] = Task(id, loc, work, reward, thresh)

    def load_tasks_on_agent(self, env: Environment):
        for t_id, task in env.task_dict.items():
            self._create_local_task(t_id, task.location, task.work, task.reward, task.arrival_thresh)
    
    
    # === SENSING and MODELING ===

    def _initialize_model(self, dims: tuple) -> EnvironmentModel:
        """initialize an environment model, scaled by dimensions"""

        # Dims are env coord ranges (like (100, 195))
        x_size = abs(dims[0][0] - dims[0][1])
        y_size = abs(dims[1][0] - dims[1][1])
        z_size = abs(dims[2][0] - dims[2][1])

        if z_size == 0:  # 2D environment
            model = EnvironmentModel(y_size, x_size)

        return model

    def sense_location_from_env(self, env: Environment):
        """
        Sense loc from environment (location dict with keys agent id)
        """
        self.location = env.agent_loc_dict[self.id]
        
    def _get_agent_loc_from_env(self, env: Environment, agent_id):
        """
        Sense agent loc from environment (location dict with keys agent id)
        """
        return env.agent_loc_dict[agent_id]

    def _sense_flow_from_env(self, env: Environment):
        """
        Sense flow from environment, log to observations locally
        """
        # Get flow from actual agent location
        self.local_flow = env.get_local_flow(self.location)

        # Map observation location to model coordinate
        x_model, y_model = self.env_model.convert_location_to_model_coord(
            self.env_dim_ranges, self.location
        )

        # Add to observation list with model location (if new obs)
        # y,x for row, col
        obs = ((y_model, x_model), (self.local_flow[0], self.local_flow[1]))
        if obs not in self.observations:
            self.observations.append(obs)
            # self.event = True

    def _apply_observations_to_model(self):
        """
        Applies agent's local observations to local copy of model, updates resulting task graph for planning
        """
        # Apply to environment model
        # TODO it would be nice to only apply once
        # print("Applying obs", self.observations)
        while len(self.observations) > 0:
            obs = self.observations.pop(0)
            self.env_model.apply_observation(obs)
            
    # === MOVEMENTS ===
    
    def reduce_energy(self, vel_mag=0):
        """
        Given a velocity, reduce energy for 1 timestep of holding
        that velocity

        @param vel_mag: commanded velocity m/timestep
        """
        self.sim_data["budget"] -= self.get_energy_cost_from_vel(vel_mag)

    def get_energy_cost_from_vel(self, vel_mag):
        return self.energy_burn_rate * vel_mag

    # def update_position_mod_vector(self, loc=[]):
    #     if self.sim_data["basic"]:
    #         print("No pos vector because basic sim")
    #         return

    #     if len(loc) == 0:
    #         dest_loc = self.task_dict[self.action[1]].location
    #     else:
    #         dest_loc = loc
    #     # Get modification to position assuming travel at constant velocity
    #     vector = self.env_model.generate_scaled_travel_vector(
    #         self.location, dest_loc, self.sim_data["velocity"]
    #     )
    #     self.position_mod_vector = vector
    #     # print("Current loc is", self.location, "Destination is", dest_loc)
    #     # print("Position mod vector is then", vector)

    # def get_target_location(self):
    #     if self.action[1] == "Init":
    #         return self.location
    #     else:
    #         return self.task_dict[self.action[1]].location

    # def get_command_velocity(self):
    #     """
    #     Returns velocity command required to reach waypoint given
    #     local flows
    #     """
    #     cmd_vel = tuple(
    #         self.position_mod_vector[i] - self.local_flow[i] for i in range(len(self.local_flow))
    #     )
    #     resultant_cmd_vel = np.linalg.norm(cmd_vel)
    #     return resultant_cmd_vel


    # === MISC ===
    
    def _set_up_dim_ranges(self, env: Environment):
        self.env_dim_ranges = env.get_dim_ranges()


# Setup functions

def load_data_from_config(sim_data_fp, rand_base=None):
    
    """
    Load in problem and solver data for agents to use during operations
    """
    with open(sim_data_fp, "r") as p_fp:
        sim_data = yaml.safe_load(p_fp)

        dims = (tuple(sim_data["xCoordRange"]),
                tuple(sim_data["yCoordRange"]),
                tuple(sim_data["zCoordRange"]),
                )

        sim_data = {
            "num_robots": sim_data["num_robots"],
            "budget": sim_data["budget"],
            "velocity": sim_data["velocity"],
            "energy_burn_rate": sim_data["energy_burn_rate"],
            "m_id": sim_data["m_id"],
            "env_dims": dims,
        }
        sim_data["base_loc"] = rand_base


    return sim_data
