import time
import numpy as np
import yaml
import copy

from control.mothership import gen_mother_from_config
from control.passenger import generate_passengers_from_config
from control.task import generate_tasks_from_config
from env.environment import make_environment_from_config, Environment
from utils.logger import FileLogger, init_logger
from utils.visualizer import set_up_visualizer

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== NN Multi-Agent Policy Network Functions ====

class MultiHeadedPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, num_agents):
        """
        Initializes a multi-headed policy network.
        
        Args:
            input_dim (int): The size of the input observation (M).
            hidden_dim (int): The number of units in the hidden layers.
            num_actions (int): The number of possible actions (N).
            num_agents (int): The number of agents (K).
        """
        super(MultiHeadedPolicyNetwork, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Create separate heads for each agent
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, num_actions) for _ in range(num_agents)])

    def forward(self, input):
        """
        Forward pass through the network.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            list[torch.Tensor]: A list of tensors, each of shape (batch_size, num_actions),
                                representing action probability distributions for each agent.
        """
        # Pass the input through the shared backbone
        trunk_output = self.trunk(input)
        
        # Generate action probabilities for each agent using separate heads
        outputs = []
        for head in self.heads:
            head_output = head(trunk_output)
            action_probs = F.softmax(head_output, dim=-1)
            outputs.append(action_probs)
        
        return outputs
    
    def get_action(self, observation):
        """Get actions for all agents given an observation."""
        with torch.no_grad():
            action_distributions = self.forward(observation)
            actions = [torch.multinomial(probs, 1).item() for probs in action_distributions]
        return actions


class Policy:
    
    def __init__(self, observation_size, hidden_dim, num_agents, action_dict, agent_velocity) -> None:
        
        self.fitness = None
        self.reward = None
        self.action_dict = action_dict 
        self.num_agents = num_agents
        self.agent_velocity = agent_velocity
        
        self.policy_net = MultiHeadedPolicyNetwork(observation_size,
                                                   hidden_dim,
                                                   len(action_dict),
                                                   num_agents)
    
    def get_action(self, observation):
        """
        8 Discrete Actions (perhaps):
        [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]
        """

        obs_tensor = torch.tensor(np.array(observation)).float()
        obs_tensor = obs_tensor.flatten() # prepare network input

        joint = self.policy_net.get_action(obs_tensor) # query network

        actions_with_vel = []
        for action_id in joint:
            # process output action id's into robots joint motions
            action = np.multiply(self.action_dict[action_id], 
                                   self.agent_velocity)
            actions_with_vel.append(action)

        return actions_with_vel

# ==== EA Functions ====

def crossover(parent1: Policy, parent2: Policy):
    """Perform crossover between two networks."""
    child = copy.deepcopy(parent1)
    child.fitness = None
    for param1, param2 in zip(child.policy_net.parameters(), parent2.policy_net.parameters()):
        mask = torch.rand_like(param1) < 0.5
        param1.data[mask] = param2.data[mask]
    return child

def mutate(network: Policy, mutation_rate=0.1):
    """Mutate a network by adding noise to its parameters."""
    for param in network.policy_net.parameters():
        if torch.rand(1).item() < mutation_rate:
            noise = torch.randn_like(param) * 0.1
            param.data.add_(noise)

def tournament_selection(population: list[Policy], tournament_size=3):
    """
    Select one individual using tournament selection. Randomly pick `tournament_size` individuals and select the one with the highest fitness.
    """
    policies = np.random.choice(population, tournament_size)
    best = max(policies, key=lambda pol: pol.fitness)
    return best


# ==== Reward and PBRS Functions ====

# NOTE Eventually modify this to consider dead robots as well
def compute_reward(env: Environment):
    """
    Compute global reward as sum of completed tasks.
    """
    return sum(t.complete for t in env.task_dict.values())

def compute_potential(env: Environment):
    """
    Compute a global potential function for the current environment state
    """
    # Let's try average distance to nearest incomplete task for each agent
    inv_dists_to_nearest_task = []
    
    for a in env.agent_loc_dict:
        if a == -1: # skip mothership
            continue

        min_dist = np.inf
        for t in env.task_dict.values():
            dist = np.linalg.norm(env.agent_loc_dict[a] - t.location)
            if dist < min_dist:
                min_dist = dist

        # Append inverse because we want to provide positive reward when min_dist value goes down
        inv_dists_to_nearest_task.append(1/min_dist)

    return np.mean(inv_dists_to_nearest_task)

if __name__ == "__main__":
    test_config = "config/testing_config.yaml"
    topo_file = "datasets/topogrophy/topography.nc"
    tide_folder = "datasets/currents"

    # Load testing params
    with open(test_config, "r") as f:
        config = yaml.safe_load(f)
        logging = config["logging"]
        verbose = config["verbose"]
        num_agents = config["num_robots"]
        num_tasks = config["problem_size"]
        agent_vel = config["velocity"]
        test_name = config["test_name"]
        epochs = config["epochs"]
        tests = config["tests"]
        timeout = config["sim_timeout"]
        num_policies = config["num_policies"]
        show_viz = config["viz"]
        comms_max_range = config["comms_max_range"]
        hidden_dim = config["hidden_dim"]
        obs_size = 2*(num_agents + num_tasks) # (x,y) relative pos of agents+base and tasks
    f.close()

    print("Initializing...")

    # === Initialize ===

    # Initialize results logger here
    if logging:
        logger = init_logger("Simulations")
        file_logger = FileLogger(filename=test_name)
    
    # Initialize environment
    print("\t Environment...")
    env = make_environment_from_config(test_config,
                                       topo_file,
                                       tide_folder
                                       )
    # Generate tasks & based (FIXED ENVIRONMENT)
    env.task_dict = generate_tasks_from_config(test_config, env)
    random_base = env.setup_random_base_loc()
    
    # Initialize agents
    print("\t Agents...")
    passenger_list = generate_passengers_from_config(test_config)
    mothership = gen_mother_from_config(test_config, passenger_list)
    mothership.agent_ids = [a.id for a in passenger_list]
    for p in passenger_list:
        p.agent_ids = [a.id for a in passenger_list]
        p.mothership_id = mothership.id
    
    # Initialize a population of k policies
    print("\t Policies...")
    action_dict ={0: [1,0,0],
                1: [1,1,0],
                2: [0,1,0],
                3: [-1,1,0],
                4: [-1,0,0],
                5: [-1,-1,0],
                6: [0,-1,0],
                7: [1,-1,0],
                }
    policies = [Policy(obs_size, hidden_dim, num_agents, action_dict, agent_vel) for _ in range(num_policies)]

    print("Begin Training")
    for ep in range(epochs):
        if ep == epochs-1: show_viz = True # Show last training cycle

        # High-level evolution loop
        if verbose: print("\n==== Trial", ep, " ====")

        # === Evaluate Policies (Rollout) ===
        
        # Loop through each unevaluated policy. All agents use pol to get action values
        for pol in policies:
            if pol.fitness != None:
                continue
        
            tests_final_rewards = [] # track final rewards from multiple tests
            tests_rewards_with_potentials = [] # track G+F from multiple tests
            aggr_potential = 0
            prev_state_potential = 0
        
            # Rollout policy in multiple randomized environments
            for ts in range(tests):

                if verbose: print("\t == Test ", ts, " ==")
                                
                # Reset tasks and agent locs
                if verbose: print("\t Reset Env & Agents")
                env.reset()

                # Reset agents
                mothership.reset(env)
                for p in passenger_list:
                    p.reset(env)
        
                # Start Visualizer
                if show_viz:
                    title = "Tr " + str(ep) + ", Ts", str(ts)
                    viz = set_up_visualizer(env, env.task_dict, title)

                # Run Simulation
                done = False
                step = 0
                if verbose: print("\t Start Sim")
                while not done and step < timeout:
                    step += 1
                    
                    # Assemble joint action
                    for p in passenger_list:
                        # Update observations
                        p.sense_location_from_env(env)
                    # Update global observation (same as local)
                    mothership.update_observation(env)
                    # Use policy to get next joint action
                    joint_action = pol.get_action(mothership.observation)

                    # Update environment
                    # (NOTE Try out different reward shaping approaches here)
                    # TODO Add consideration for comms max range penalty
                    joint_reward, done = env.step(joint_action)

                    # TODO Calculate new state potential(s) here for PBRS
                    state_potential = compute_potential(env)
                    aggr_potential += (state_potential-prev_state_potential) # P(s')-P(s)
                    prev_state_potential = state_potential
                    
                    # TODO RL: Add observations, actions, rewards to replay buffer for RL

                    # Opt. update visual
                    if show_viz:
                        viz.display_env(passenger_list, static=False)
                        time.sleep(0.05)

                if show_viz:
                    viz.close_viz()

                # Calculate final global reward with PBRS, add it to reward tracker
                final_reward = compute_reward(env)
                tests_final_rewards.append(final_reward)
                tests_rewards_with_potentials.append(final_reward + aggr_potential) # Use G+F, with F=sum(P(s')-P(s))
                if verbose: print("\t Sim Complete. Reward:", final_reward)
                
            # Evaluate fitness as average G+F over rollouts, assign to policy
            pol.fitness = np.mean(tests_rewards_with_potentials) # Fitness = G+F
            pol.reward = np.mean(tests_final_rewards)
            if verbose: print("Policy fitness:", pol.fitness)

        # Select population of policies to retain (top half)
        policies.sort(key=lambda policy: policy.fitness, reverse=True)
        print("Gen", ep, "Policies Fitnesses: ", [p.fitness for p in policies], " | Best Reward:", policies[0].reward)
        new_policies = [policies[0]] # keep best policy

        # Break early (no new children) if in last cycle
        if ep == epochs-1:
            continue

        # Mutate & crossover to make new policies, flag as unevaluated
        while len(new_policies) < num_policies:
            parent1 = tournament_selection(policies, num_policies//3)
            parent2 = tournament_selection(policies, num_policies//3)
            child = crossover(parent1, parent2)
            mutate(child)
            new_policies.append(child)
        policies = new_policies[:]
        
        # TODO Do some logging (like fitness that was just evaluated)
        
        
        # ==== PERIODICALLY ==== TODO RL: Do some RL using a buffer of experiences
        
        
    # TODO After training, somehow save the best-performing policies
    best_policy = max(policies, key=lambda policy: policy.fitness)


    # TODO Add visualizer for best policy