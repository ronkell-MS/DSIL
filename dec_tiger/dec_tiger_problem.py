"""
States: tiger-left, tiger-right
Actions: open-left, open-right, listen so: all the combinations
(open-left,open-left),(open-left,open-right),(open-right,open-left),(open-right,open-right),(open-left,open-left).....
Rewards:
    +10 for opening treasure door. -100 for opening tiger door, -25 for opening tiger door togthere.
    -1 for listening.
Observations: You can hear either "tiger-left", or "tiger-right"
so for multiagent vector 4 options:
"tiger-left,tiger-left","tiger-left,tiger-right","tiger-right,tiger-left","tiger-right,tiger-right".

Note that in this example, the TigerProblem is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more
complicated examples.)
De compose of DECpomdp

action:
There is X as seprator instead of , because parser cant recive it
agent 1
(listen,idle)
(open-left,idle)
(open-right,idle)

agent 2
(idle,listen)
(idle,open-left)
(idle,open-right)

collab
(open-left,open-left)
(open-right,open-right)
"""
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys


class DecTigerState(pomdp_py.State):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, DecTigerState):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "DecTigerState(%s)" % self.name
    def other(self):
        if self.name.endswith("left"):
            return DecTigerState("tiger-right")
        else:
            return DecTigerState("tiger-left")


class DecTigerAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, DecTigerAction):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "DecTigerAction(%s)" % self.name

class DecTigerObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, DecTigerObservation):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "DecTigerObservation(%s)" % self.name

class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listenXidle" or action.name == "idleXlisten":
            # heard the correct growl
            if observation.name == next_state.name:
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listenXidle" or action.name == "idleXlisten":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0,1) < thresh:
            return DecTigerObservation(next_state.name)
        else:
            return DecTigerObservation(next_state.other().name)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [DecTigerObservation(s)
                for s in {"tiger-left", "tiger-right"}]
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        action_name = action.name.split('X')
        if action_name[0].startswith("open") or action_name[1].startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        action_name = action.name.split('X')
        if action_name[0].startswith("open") or action_name[1].startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return DecTigerState(state.name)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [DecTigerState(s) for s in {"tiger-left", "tiger-right"}]

class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        action_name = action.name.split('X')
        if action_name[0] == 'open-left' or action_name[1] == 'open-left':
            if state.name == 'tiger-right':
                return 10
            elif action_name[0] == action_name[1]:
                return -25
            else:
                return -100
        elif action_name[0] == 'open-right' or action_name[1] == 'open-right':
            if state.name == 'tiger-left':
                return 10
            elif action_name[0] == action_name[1]:
                return -25
            else:
                return -100
        else: # listen
            return -1



    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    ACTIONS = {DecTigerAction(s)
              for s in {"open-leftXidle", "open-rightXidle", "listenXidle",
                        'idleXopen-left','idleXopen-right','idleXlisten',
                        'open-leftXopen-left','open-rightXopen-right'}}

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


class DecTigerProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(obs_noise),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="DecTigerProblem")

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right';
                         True state of the environment
            belief (float): Initial belief that the target is
                            on the left; Between 0-1.
            obs_noise (float): Noise for the observation
                               model (default 0.15)
        """
        init_true_state = DecTigerState(state)
        init_belief = pomdp_py.Histogram({
            DecTigerState("tiger-left"): belief,
            DecTigerState("tiger-right"): 1.0 - belief
        })
        dec_tiger_problem = DecTigerProblem(obs_noise,
                                     init_true_state, init_belief)
        dec_tiger_problem.agent.set_belief(init_belief, prior=True)
        return dec_tiger_problem

def test_planner(dec_tiger_problem, planner, nsteps=3,
                 debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        tiger_problem (TigerProblem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        action = planner.plan(dec_tiger_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(dec_tiger_problem.agent.tree)
            import pdb; pdb.set_trace()

        print("==== Step %d ====" % (i+1))
        print("True state:", dec_tiger_problem.env.state)
        print("Belief:", dec_tiger_problem.agent.cur_belief)
        print("Action:", action)
        # There is no state transition for the tiger domain.
        # In general, the ennvironment state can be transitioned
        # using
        #
        #   reward = tiger_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        reward = dec_tiger_problem.env.reward_model.sample(dec_tiger_problem.env.state, action, None)
        print("Reward:", reward)

        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this observation
        # should be sampled from agent's observation model, as
        #
        #    real_observation = tiger_problem.agent.observation_model.sample(tiger_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that tiger_problem.env.state stores the
        # environment state after action execution.
        real_observation = DecTigerObservation(dec_tiger_problem.env.state.name)
        print(">> Observation:",  real_observation)
        dec_tiger_problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(dec_tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(dec_tiger_problem.agent.cur_belief,
                      pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                dec_tiger_problem.agent.cur_belief,
                action, real_observation,
                dec_tiger_problem.agent.observation_model,
                dec_tiger_problem.agent.transition_model)
            dec_tiger_problem.agent.set_belief(new_belief)
        action_name = action.name.split('X')
        if action_name[0].startswith("open") or action_name[1].startswith("open"):
            # Make it clearer to see what actions are taken
            # until every time door is opened.
            print("\n")


def main():
    init_true_state = random.choice([DecTigerState("tiger-left"),
                                     DecTigerState("tiger-right")])
    init_belief = pomdp_py.Histogram({DecTigerState("tiger-left"): 0.5,
                                      DecTigerState("tiger-right"): 0.5})
    dec_tiger_problem = DecTigerProblem(0.15,  # observation noise
                                 init_true_state, init_belief)

    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(dec_tiger_problem, vi, nsteps=3)

    # Reset agent belief
    dec_tiger_problem.agent.set_belief(init_belief, prior=True)

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                           num_sims=4096, exploration_const=50,
                           rollout_policy=dec_tiger_problem.agent.policy_model,
                           show_progress=True)
    test_planner(dec_tiger_problem, pouct, nsteps=10)
    TreeDebugger(dec_tiger_problem.agent.tree).pp

    # Reset agent belief
    dec_tiger_problem.agent.set_belief(init_belief, prior=True)
    dec_tiger_problem.agent.tree = None

    print("** Testing POMCP **")
    dec_tiger_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True)
    pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                           num_sims=1000, exploration_const=50,
                           rollout_policy=dec_tiger_problem.agent.policy_model,
                           show_progress=True, pbar_update_interval=500)
    test_planner(dec_tiger_problem, pomcp, nsteps=10)
    TreeDebugger(dec_tiger_problem.agent.tree).pp

if __name__ == '__main__':
    main()
