import numpy as np
import torch

### mainly refere to https://github.com/DHDev0/Muzero/blob/main/monte_carlo_tree_search.py


class Node(object):
    def __init__(self, prior: float, q_value: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.q_value = q_value
        self.children = {}
        self.kv_cache = 0
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self) -> float:
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MinMaxStats(object):
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTS():
    def __init__(
        self, 
        pb_c_base=19652, 
        pb_c_init=1.25, 
        discount=0.99, 
    ):
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.node = None
        self.overall_graph = []

    def generate_root_node(self):
        self.root = Node(0, 0)
        self.min_max_stats = MinMaxStats()

    def expand_the_children_of_the_root_node(self, policy, critic):
        assert policy.shape[0] == 1 and critic.shape[0] == 1, "Batch is not supported yet"
        for j in range(policy.shape[-1]):
            self.root.children[j] = Node(prior=policy[0, j], q_value=critic[0, j])

    def initialize_history_node_searchpath_variable(self):
        history = []
        self.node = self.root
        search_path = [self.root]
        return history, search_path

    def ucb_score(self, parent, child):
        # the probability of invalid actions are near 0
        if child.prior < 1e-12:
            return -np.inf
        
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        prior_score = (np.sqrt(parent.visit_count) *  pb_c * child.prior) / (child.visit_count + 1)
        
        if child.visit_count > 0:
            value_score = self.min_max_stats.normalize(child.reward + self.discount * child.value())
        else:
            value_score = 0  # child.q_value  # 0

        return prior_score + value_score

    def select_child(self):
        _, action, child = max((self.ucb_score(self.node, child), action, child)
                               for action, child in self.node.children.items())
        return action, child

    def choose_node_to_expand_using_max_ucb_score(self, history, search_path, max_depth=np.inf):
        depth = 0

        while self.node.expanded():
            action, self.node = self.select_child()
            history.append(action)
            search_path.append(self.node)

            depth += 1
            if depth >= max_depth:
                break
        
        return search_path[-2]

    def update_reward_and_kv_cache_for_the_chosen_node(self, reward, kv_cache):
        self.node.reward, self.node.kv_cache = reward, kv_cache

    def create_new_node_in_the_chosen_node_with_action_and_policy(self, policy, critic):
        assert policy.shape[0] == 1 and critic.shape[0] == 1
        for j in range(policy.shape[-1]):
            self.node.children[j] = Node(prior=policy[0, j], q_value=critic[0, j])

    def back_propagate_and_update_min_max_bound(self, search_path, value):
        for bnode in reversed(search_path):
            bnode.value_sum += value
            bnode.visit_count += 1
            self.min_max_stats.update(bnode.value())
            value = bnode.reward + self.discount * value