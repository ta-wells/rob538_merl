import numpy as np
import yaml

from utils.helpers import sample_from_range


class Task:
    # NOTE Potentially make seperate function for representing environment model as graph with stochastic edges
    def __init__(self,
                 id: int,
                 location: tuple,
                 work: int,
                 reward: int,
                 thresh: int
                 ) -> None:
        self.id = id
        self.location = location
        self.work = work
        self.reward = reward
        self.arrival_thresh = thresh
        
        self.distances_to_tasks = {}  # Normal distribution edge costs
        self.complete = False

    def calc_dist_to_task(self, task_id, location):
        diff_vec = self.location - location
        vec_mag = np.linalg.norm(diff_vec)
        if vec_mag == 0.0:
            vec_mag = 0.05  # Set to number close to 0.0 to prevent error
        self.distances_to_tasks[task_id] = vec_mag
        print("Distance", (self.id, task_id), ":", vec_mag)



def generate_tasks_from_config(problem_config_fp, env):

    with open(problem_config_fp, "r") as f:
        config = yaml.safe_load(f)

        task_dict = {}

        for i in range(config["problem_size"]):

            dim_ranges = env.get_dim_ranges()
            x = sample_from_range(dim_ranges[0][0],
                                  dim_ranges[0][1])
            y = sample_from_range(dim_ranges[1][0],
                                  dim_ranges[1][1])
            z = sample_from_range(dim_ranges[2][0],
                                  dim_ranges[2][1])

            work = sample_from_range(config["work_range"][0],
                                     config["work_range"][1])
            reward = sample_from_range(config["reward_range"][0],
                                       config["reward_range"][1])
            thresh = config["arrival_radius"]

            task_dict["v"+str(i)] = Task("v"+str(i),
                                         np.array([x, y, z]), 
                                         work,
                                         reward, 
                                         thresh)

    return task_dict
