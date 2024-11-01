import time
from argparse import ArgumentParser, Namespace

import numpy as np
import yaml

from control.mothership import gen_mother_from_config
from control.passenger import generate_passengers_from_config
from control.task import generate_tasks_from_config
from env.environment import make_environment_from_config
from utils.logger import FileLogger, init_logger
# from utils.plotter import plot_results_from_log
from utils.visualizer import set_up_visualizer


def get_args() -> Namespace:
    """Parse the script arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "test_config",
        type=str,
        help="Full path to the test configuration file",
    )
    parser.add_argument(
        "topo_file",
        type=str,
        help="Full path to the topography data file",
    )
    parser.add_argument(
        "tide_folder",
        type=str,
        help="Full path to the tidal data folder",
    )

    return parser.parse_args()

class Policy:
    
    def __init__(self, num_agents, num_tasks, agent_velocity) -> None:
        
        self.fitness = None
        self.action_dict = {0: [1,0,0],
                            1: [1,1,0],
                            2: [0,1,0],
                            3: [-1,1,0],
                            4: [-1,0,0],
                            5: [-1,-1,0],
                            6: [0,-1,0],
                            7: [1,-1,0],
                            }
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.agent_velocity = agent_velocity
        
        # TODO Initialize Q-Network (MAKE SURE INPUT DIMS MATCH OBSERVATION DIMS)
    
    def get_action(self, observation):
        """
        8 Discrete Actions (perhaps):
        [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]
        """
        # TODO Q-Network inference here
        
        action_id = np.random.randint(0,7)     
        act_with_vel = np.multiply(self.action_dict[action_id], 
                                   self.agent_velocity)
        return act_with_vel


# === TRAINING OVERVIEW ===

# Initialize policies
# Initialize experience buffer

# For each EA trial
    # For each unevaluated policy, evaluate fitness:
        # For each test (here we run policy in many random environments)
            # Generate random environment
            # Run simulation
                # Log random experiences in buffer
            # Log reward (!! Difference Rewards and/or Shaping Here)
        # Evaluate fitness
    # Reduce population
    # Mutate policies, produce new offspring
    
    # == Periodically, do RL on best policy using experience buffer ==
    # Select best policy
    # For each RL trial
        # Sample random experience (State, Action, Rew) from buffer
        # Action' <- Policy(State)
        # ?? Somehow evaluate Rew of selected action vs stored ??
        # Do a value update    


if __name__ == "__main__":
    args = get_args()

    # Load testing params
    with open(args.test_config, "r") as f:
        test_config = yaml.safe_load(f)
        logging = test_config["logging"]
        num_agents = test_config["num_robots"]
        num_tasks = test_config["problem_size"]
        agent_vel = test_config["velocity"]
        test_name = test_config["test_name"]
        evo_trials = test_config["trials"]
        tests = test_config["tests"]
        timeout = test_config["sim_timeout"]
        policy_pop = test_config["policy_pop"]
        show_viz = test_config["viz"]
        comms_max_range = test_config["comms_max_range"]

    print("Initializing...")

    # === Initialize ===

    # Initialize results logger here
    if logging:
        logger = init_logger("Simulations")
        file_logger = FileLogger(filename=test_name)
    
    # Initialize environment
    print("\t Environment...")
    env = make_environment_from_config(args.test_config,
                                       args.topo_file,
                                       args.tide_folder
                                       )
    
    # Initialize agents
    print("\t Agents...")
    passenger_list = generate_passengers_from_config(
        args.test_config)
    mothership = gen_mother_from_config(args.test_config,
                                        passenger_list,
                                        )
    for p in passenger_list:
        p.agent_ids = [a.id for a in passenger_list]
        p.mothership_id = mothership.id
    
    # Initialize a population of k policies
    print("\t Policies...")
    policies = [Policy(num_agents, num_tasks, agent_vel) for _ in range(policy_pop)]

    print("Begin Training")
    for tr in range(evo_trials):
        # High-level evolution loop
        print("\n==== Trial", tr, " ====")
        
        # === Evaluate Policies (Rollout) ===
        
        # Loop through each unevaluated policy. All agents use pol to get action values
        for pol in policies:
            if pol.fitness != None:
                continue
        
            tests_final_rewards = [] # track final rewards from multiple tests
        
            # Rollout policy in multiple randomized environments
            for ts in range(tests):
                print("\t == Test ", ts, " ==")
                                
                # Reset to random environment
                print("\t Reset Env & Agents")
                env.reset()
                random_base = env.setup_random_base_loc()
                env.task_dict = generate_tasks_from_config(args.test_config,
                                                           env,
                                                           )
                # Reset agents
                mothership.reset(env)
                for p in passenger_list:
                    p.reset(env)
        
                # Start Visualizer
                if show_viz:
                    title = "Tr " + str(tr) + ", Ts", str(ts)
                    viz = set_up_visualizer(env, env.task_dict, title)

                # Run Simulation
                done = False
                step = 0
                print("\t Start Sim")
                while not done and step < timeout:
                    step += 1
                    
                    # Assemble joint action
                    joint_action = []
                    for p in passenger_list:
                        # Update observations
                        p.update_observation(env)
                            
                        # Use policy to get next action
                        action = pol.get_action(p.observation)
                        joint_action.append(action)
                        
                    # print("\t\t Step", step, " Joint Action:", joint_action)
                    
                    # Update environment
                    # (NOTE Try out different reward shaping approaches here)
                    joint_reward, done = env.step(joint_action)
                    
                    # TODO RL: Add observations, actions, rewards to replay buffer for RL

                    # Opt. update visual
                    if show_viz:
                        viz.display_env(passenger_list, static=False)
                        time.sleep(0.05)

                if show_viz:
                    viz.close_viz()

                # Calculate final global reward, add it to reward tracker 
                final_reward = sum(t.complete for t in env.task_dict.values())
                tests_final_rewards.append(final_reward)
                print("\t Sim Complete. Reward:", final_reward)
                
            # Evaluate fitness, assign to policy
            pol.fitness = np.mean(tests_final_rewards)
            print("Policy fitness:", pol.fitness)
            

        # TODO Select population of policies to retain
        
        # TODO Mutate some policies, flag as unevaluated
        
        # TODO Do some logging (like fitness that was just evaluated)
        
        
        # ==== PERIODICALLY ==== TODO RL: Do some RL using a buffer of experiences
        
        
    # TODO After training, somehow save the best-performing policies