from copy import deepcopy

from control.agent import Agent, load_data_from_config
from control.task import Task
from env.comms_manager import CommsManager


class Passenger(Agent):

    def __init__(self, id: int, sim_data: dict) -> None:

        super().__init__(id, sim_data)

        self.type = self.PASSENGER       

        # Passenger action variables
        self.finished = False
        self.work_remaining = 0

   
    # === ACTIONS ===

    def action_update(self, comms_mgr):
        """
        Update action according to current agent status and local schedule
        """
        # print("Passenger", self.id, " Current action:",
        #       self.action, " | Schedule:", self.schedule, " | Dead:", self.dead, " | Complete:", self.finished)
        # ("IDLE", -1)
        self.task_dict[self.sim_data["rob_task"]].location = self.location

        # === BREAKING CRITERIA ===
        if self.finished or self.dead:
            # print(self.id, "Finished/Dead! Finished:",
            #       self.finished, " Dead:", self.dead)
            return

        # If out of energy, don't do anything
        if self.sim_data["budget"] < 0 and not self.finished:
            print(self.id, "Dead!")
            self.action[0] = self.IDLE
            self.dead = True
            return

        # If waiting for init schedule command from comms
        if len(self.schedule) == 0 and self.action[1] == "Init":
            if self.my_action_dist != None:
                self.schedule = self.my_action_dist.best_action().action_seq[:]
            else:
                return

        if self.action[0] == self.IDLE and len(self.schedule) > 0:
            # print("Traveling to first task in sched:", self.schedule)
            # Start tour & remove first element from schedule
            self.action[0] = self.TRAVELING
            # leaving = self.schedule.pop(0)
            # self.task_dict[leaving].complete = True
            self.action[1] = self.schedule.pop(0)

        if self.action[0] == self.IDLE:
            # print(self.id, "Resuming travel to", self.action[1])
            self.action[0] = self.TRAVELING

        task = self.task_dict[self.action[1]]
        arrived = False

        # 0) Update travel progress if traveling, check arrived
        if self.action[0] == self.TRAVELING:
            # print(self.id, "Travel in progress...")
            if self.sim_data["basic"]:
                # Travel progress for basic
                self.travel_remaining -= self.sim_data["velocity"]
                if self.travel_remaining <= 0:
                    arrived = True
            else:
                # Travel progress for ocean sim
                self.update_position_mod_vector()
                arrived = self.env_model.check_location_within_threshold(
                    self.location, task.location, self.ARRIVAL_THRESH
                )

        # 1) If traveling and arrived at task, begin work
        if self.action[0] == self.TRAVELING and arrived:
            # print(self.id, "Arrived at task. starting Work. Work remaining:", task.work)
            self.action[0] = self.WORKING
            self.work_remaining = task.work

        # 2) If working and work is complete, become Idle
        if self.action[0] == self.WORKING and self.work_remaining <= 0:
            # print(self.id, "Work complete, becoming Idle")
            # self.event = True
            self.action[0] = self.IDLE
            # Mark task complete
            self.glob_completed_tasks.append(self.action[1])
            self.task_dict[self.action[1]].complete = True
            self.stored_reward_sum += self.task_dict[self.action[1]].reward
            # content = (
            #     self.id, self.sim_data["m_id"], "Complete Task", self.glob_completed_tasks)

            # distributed-type message passing here
            for a in self.agent_ids:
                if a.id != self.id:
                    if a.type == self.PASSENGER and self.neighbors_status[a.id]:
                        # print(self.id, "Sending completed task to", a.id)
                        content = (
                            self.id, a.id, "Complete Task", [self.action[1]])
                        self.send_msg_up_chain(comms_mgr, content)
            # print(self.id, "Sending completed task to M")
            content = (
                self.id, self.sim_data["m_id"], "Complete Task", [self.action[1]])
            self.send_msg_up_chain(comms_mgr, content)

        # 3) Otherwise continue doing work
        elif self.action[0] == self.WORKING and self.work_remaining > 0:
            # otherwise, continue working
            # print(self.id, "Work in progress")
            self.work_remaining -= 1

        # If arrived at home, set finished to true
        if (arrived and self.action[0] == self.IDLE
                and self.action[1] == self.sim_data["end"]):
            # if no additional tasks are reachable
            self.finished = True
            return

        # print("Agent", self.id, " Updated action is", self.action,
        #       " | Schedule:", self.schedule)



# === Setup Functions ===

def generate_passengers_with_data(sim_data) -> list[Passenger]:
    pssngr_list = []
    for i in range(sim_data["num_robots"]):
        p = Passenger(i,
                      sim_data=deepcopy(sim_data),
                      )
        pssngr_list.append(p)
    return pssngr_list


def generate_passengers_from_config(test_config_fp,
                                    rand_base=None
                                    ) -> list[Passenger]:
    sim_data = load_data_from_config(test_config_fp, rand_base)

    return generate_passengers_with_data(sim_data)
