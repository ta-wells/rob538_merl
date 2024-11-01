import math
import os
import random

import numpy as np
import xarray as xr
import yaml


class Environment:

    def __init__(
        self,
        topography_file: str,
        flow_data_files: list[str],
        env_dimensions: tuple,
        agent_loc_dict: dict,
        base_loc,
        thin_params: tuple = (1, 1, 1),
        time_step_size: int = 1,
        flow_vec_mod: int=1000,
    ):
        """
        @param topography: xarray dataset
        @param flow_data_list: list of hourly environment state xarray datasets
        @param env_dimenions: tuple ((x1,x2),(y1,y2),(z1,z2)) environment dimensions to sample from dataset.
        z = 1 for 2D environment.
        @param thin_params: tuple (x,y,z) params for thinning dataset.
        z = 1 for 2D environment.
        @param robots_list: list of all robots in environment (possibly includes mothership)
        @param time_step_size: int minutes of one time step for incrementing environment
        """
        
        # divide hour by time step size to get number of steps
        self.time_steps = 60 / time_step_size
        self.flow_multiplier = flow_vec_mod 

        self.agent_loc_dict = agent_loc_dict
        self.base_loc = base_loc
        self.task_dict = {}
        
        self.env_dimensions = env_dimensions       
        self.processed_flow_data = self._process_flow_data(thin_params,
                                                          env_dimensions,
                                                          topography_file,
                                                          flow_data_files
                                                          )
        self.SLICE = True

        # Set current flow data reference frame & current flow state
        # index of first file we are interpolating flow from (idx+1 for next file)
        self.flow_data_idx = 0
        # Current environment frame
        self.current_flow_state = self.processed_flow_data[self.flow_data_idx]

        # Create & load flow vector modifiers (for transitioning from one dataset to the next over time_steps - 1hr transition)
        # Only if we loaded multiple flow files for dynamic environmnent
        if len(self.processed_flow_data) > 1:
            self.flow_data_modifiers = self._update_flow_modifiers(
                self.processed_flow_data[self.flow_data_idx],
                self.processed_flow_data[self.flow_data_idx + 1],
            )
            
    def reset(self):
        self.task_dict = {}
        
        for a_id in self.agent_loc_dict.keys():
            self.agent_loc_dict[a_id] = self.base_loc
            
    def get_local_flow(self, loc):
        """
        Get the local flow vector at a given location

        @param loc: Coordinate location from which to extract flows

        @returns list of [x,y,z] flow components
        """

        # Get from loc (km range) to nearest pos (list idxs)
        x_coords = self.cropped_coords["x"]
        y_coords = self.cropped_coords["y"]
        z_coords = self.cropped_coords["z"]

        local_x = np.argmin(np.abs(x_coords.values - loc[0]))
        local_y = np.argmin(np.abs(y_coords.values - loc[1]))

        # Get flow vector closest to this list idx
        if not self.SLICE:
            local_z = np.argmin(np.abs(z_coords.values - loc[2]))
            # z,y,x idx
            local_flow_x = self.current_flow_state["u"][local_z][local_y][
                local_x
            ].values
            local_flow_y = self.current_flow_state["v"][local_z][local_y][
                local_x
            ].values
            local_flow_z = self.current_flow_state["w"][local_z][local_y][
                local_x
            ].values
            local_flow = [local_flow_x, local_flow_y, local_flow_z]
        else:
            local_flow_x = self.current_flow_state["u"][0][local_y][local_x].values
            local_flow_y = self.current_flow_state["v"][0][local_y][local_x].values
            # print('Flow vect at:', local_x, local_y, ' is', local_flow_x, local_flow_y)
            local_flow = [local_flow_x, local_flow_y]

        modified_flows = np.multiply(self.flow_multiplier, local_flow)

        return modified_flows

            
    def _process_flow_data(self,
                              thin_params,
                              env_dimensions,
                              topography_file,
                              flow_data_files,
                              ):

        # Process datasets (thin dataset & crop to dimensions)
        x_thinning = thin_params[0]
        y_thinning = thin_params[1]
        z_thinning = thin_params[2]

        self.SLICE = False
        if env_dimensions[2][0] == env_dimensions[2][1]:
            self.SLICE = True

        # Crop & thin topography data
        topography = xr.open_dataset(topography_file) 
        x_coords = topography["xx"][
            env_dimensions[0][0]: env_dimensions[0][1]: x_thinning
        ]
        y_coords = topography["yy"][
            env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
        ]
        z_coords = topography["zz"][
            env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
        ]
        self.cropped_coords = {"x": x_coords, "y": y_coords, "z": z_coords}

        # Crop & thin flow field data
        processed_flow_data = []
        for filename in flow_data_files:
                # (z: 68, y: 200, x: 294)
                # zonal_vel = data['uu'] # zonal velocity (along the slope, in the x-direction, positive eastward) m/s
                # merid_vel = data['vv'] # meridional velocity (across the slope, in the y-direction, positive northward) m/s
                # vert_vel = data['ww'] # vertical velocity (positive upward) m/s
                data = xr.open_dataset(filename)
                if not self.SLICE:
                    u_vecs = [
                        [
                            row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                            for row in plane[
                                env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                            ]
                        ]
                        for plane in data["uu"][
                            env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
                        ]
                    ]
                    v_vecs = [
                        [
                            row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                            for row in plane[
                                env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                            ]
                        ]
                        for plane in data["vv"][
                            env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
                        ]
                    ]
                    w_vecs = [
                        [
                            row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                            for row in plane[
                                env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                            ]
                        ]
                        for plane in data["ww"][
                            env_dimensions[2][0]: env_dimensions[2][1]: z_thinning
                        ]
                    ]
                else:
                    u_vecs = [
                        [
                            row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                            for row in data["uu"][env_dimensions[2][0]][
                                env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                            ]
                        ]
                    ]
                    v_vecs = [
                        [
                            row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                            for row in data["vv"][env_dimensions[2][0]][
                                env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                            ]
                        ]
                    ]
                    w_vecs = [
                        [
                            row[env_dimensions[0][0]: env_dimensions[0][1]: x_thinning]
                            for row in data["ww"][env_dimensions[2][0]][
                                env_dimensions[1][0]: env_dimensions[1][1]: y_thinning
                            ]
                        ]
                    ]

                processed_data = {"u": u_vecs, "v": v_vecs, "w": w_vecs}
                processed_flow_data.append(processed_data)
                
        return processed_flow_data


    def _update_flow_modifiers(self, flow_hour1, flow_hour2):
        """
        Process modifiers for interpolating between two datasets. Modifiers are added to current
        flow state at each time step

        @param flow_hour1: first hour ocean currents dataset
        @param flow_hour2: second hour ocean currents dataset

        @returns list of flow modifiers to be applied at each env step
        """
        # Find differences
        u_diff = np.subtract(
            np.array(flow_hour2["u"]), np.array(flow_hour1["u"]))
        v_diff = np.subtract(
            np.array(flow_hour2["v"]), np.array(flow_hour1["v"]))
        w_diff = np.subtract(
            np.array(flow_hour2["w"]), np.array(flow_hour1["w"]))

        # Find gradients
        u_step_mod = u_diff / self.time_steps
        v_step_mod = v_diff / self.time_steps
        w_step_mod = w_diff / self.time_steps

        modifiers = {"u": u_step_mod, "v": v_step_mod, "w": w_step_mod}

        return modifiers

    def _check_is_loc_in_env(self, dims_ranges, loc) -> bool:
        """
        Returns true if loc coordinates are within dims_ranges. False otherwise.

        @param dims_ranges:
        @param loc: (x,y) location to evaluate
        """
        x_check = loc[0] >= dims_ranges[0][0] and loc[0] <= dims_ranges[0][1]
        y_check = loc[1] >= dims_ranges[1][0] and loc[1] <= dims_ranges[1][1]
        z_check = True
        if not self.SLICE:
            z_check = loc[2] >= dims_ranges[2][1] and loc[2] <= dims_ranges[2][0]
        return x_check and y_check and z_check

    
    def get_dim_ranges(self):
        x_min = min(self.cropped_coords["x"].values)
        x_max = max(self.cropped_coords["x"].values)
        y_min = min(self.cropped_coords["y"].values)
        y_max = max(self.cropped_coords["y"].values)
        if not self.SLICE:
            z_min = min(self.cropped_coords["z"].values)
            z_max = max(self.cropped_coords["z"].values)
            return ((x_min, x_max), (y_min, y_max), (z_min, z_max))  # meters
        else:
            return ((x_min, x_max), (y_min, y_max), (0, 0))

    def setup_random_base_loc(self):
        ranges = self.get_dim_ranges()
        base_x = np.random.randint(ranges[0][0], ranges[0][1])
        base_y = np.random.randint(ranges[1][0], ranges[1][1])
        if not self.SLICE:
            base_z = np.random.randint(ranges[2][0], range[2][1])
        else:
            base_z = 0
        base_loc = [base_x, base_y, base_z]
        self.base_loc = base_loc
        for a in self.agent_loc_dict.keys():
            loc = base_loc[:]
            loc[0] = base_loc[0] + random.randint(10, 1000)
            loc[1] = base_loc[1] + random.randint(10, 1000)
            loc[2] = base_loc[2] + random.randint(10, 1000)

            self.agent_loc_dict[a] = loc

        return base_loc

    def step(self, joint_action):
        """
        Advance global actual environment by one time step. Updates robot locations & energy levels. FUTURE: Updates flow field.
        
        Update task completion status.
        
        Returns a joint_reward vector and "done"
        """
        # Take actions, complete tasks
        for a_id, act in enumerate(joint_action):
            # Update agent location
            new_loc = self.agent_loc_dict[a_id] + act
            if self._check_is_loc_in_env(self.get_dim_ranges(),
                                         new_loc):
                self.agent_loc_dict[a_id] += act
            
            # Complete tasks
            for task in self.task_dict.values():
                if task.complete:
                    continue
                if np.abs(np.linalg.norm(task.location-self.agent_loc_dict[a_id])) <= task.arrival_thresh:
                    task.complete = True
               
        # TODO Compute shaped rewards
             
        # Check if done, compute global reward
        num_complete_tasks = 0
        done = True
        for task in self.task_dict.values():
            if not task.complete:
                done = False
            else:
                num_complete_tasks += 1
                
        joint_reward = np.full(len(joint_action), num_complete_tasks)        

        return joint_reward, done


# TODO Update to take in a folder of tidal files for dynamic env
def make_environment_from_config(
    config_filepath, topo_filepath: str, tidal_folderpath: str
) -> Environment:
    """
    Create an environmnet from parameters

    @param topo_fp: filepath to environment topogrophy xarray file
    @param tidal_fp: filepath to environment tides xarray file
    @param dims: dimensions of environment

    @returns: an Environment
    """
    # Load the environment configuration
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)

        dims = (
            tuple(config["xCoordRange"]),
            tuple(config["yCoordRange"]),
            tuple(config["zCoordRange"]),
        )

        thinning = (config["xThin"], config["yThin"], config["zThin"])

        # Load in agents at starting locations
        base_loc = None
        agent_loc_dict = {}
        for i in range(config["num_robots"]):
            # loc = list(config["base_loc"])
            # # Add coords plus some noise NOTE - maybe update noise input
            # loc[0] = loc[0] + random.randint(10, 1000)
            # loc[1] = loc[1] + random.randint(10, 1000)
            # loc[2] = loc[2] + random.randint(10, 1000)

            agent_loc_dict[i] = base_loc # loc

        agent_loc_dict[config["m_id"]] = base_loc # loc
        
    flow_vec_mod = config["flow_mag_modifier"]

    # TODO Process folder of tidal filepaths into list here once we start using time-varying environment
    tidal_fp = random.choice(os.listdir(tidal_folderpath))
    tidal_fp = os.path.join(tidal_folderpath, tidal_fp)
    print("Selected flow data file", tidal_fp)
    tidal_fps = [tidal_fp]

    return Environment(
        topo_filepath,
        tidal_fps,
        dims,
        agent_loc_dict,
        base_loc,
        thinning,
        flow_vec_mod=flow_vec_mod,
    )
