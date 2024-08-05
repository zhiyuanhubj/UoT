import numpy as np

from uot.chat_utils import ques_and_cls_given_items, cls_given_repo, initialize_open_set, renew_open_set


class UoTNode:
    def __init__(self, question, answer, items, parent: 'UoTNode' = None, model="gpt-4", reply=None):
        self.children = []
        self.question = question
        self.answer = answer  # True for "YES" and False for "No"
        self.reply = reply
        self.items = items
        self.parent = parent
        self.depth = self.parent.depth + 1 if self.parent else 0
        self.model = model
        self.n_extend_layers = -1
        self.accumulation = True
        self.expected_method = 'avg'
        self.print()

    def set_config(self, n_extend_layers: int, none_acc: bool, exp: str):
        self.n_extend_layers = n_extend_layers
        self.accumulation = not none_acc
        self.expected_method = exp

    def _create_children_nodes(self, task, items: list, n, asked_ques: list = None):
        items = list(set(items))
        if self.is_terminal:
            return
        ans = ques_and_cls_given_items(task, items, n, asked_ques)
        if not ans:
            return
        self.children.extend(
            (UoTNode(a["question"], True, a["items_yes"], parent=self, model=self.model),
             UoTNode(a["question"], False, a["items_no"], parent=self, model=self.model))
            for a in ans
        )

    def find_children(self, task=None, n=None):
        if self.is_terminal:
            return None
        if task and n and (not self.children or len(self.children) < n):
            asked_ques = [ns[0].question for ns in self.children] if self.children else []
            self._create_children_nodes(task, self.items, n - len(asked_ques), asked_ques)
        return self.children

    def find_children_sep(self, task=None, n=None, prun=0):
        _children = self.find_children(task, n)
        return_list = [c[0] for c in _children] + [c[1] for c in _children] if _children else None
        if prun < 0 and return_list:
            return_list = sorted(return_list, key=lambda x: x.idiv_reward, reverse=True)[:int(-prun*len(return_list))]
        return return_list

    def handle_self_repo(self, task, repo, translate=False):
        if task.open_set_size > 0:
            a = initialize_open_set(task, repo)
            node_y = UoTNode("self-report", True, a, parent=self, model=self.model)
            node_n = UoTNode("self-report", False, [], parent=self, model=self.model)
        else:
            a = cls_given_repo(task, self.items, repo, translate, self_repo=True)
            node_y = UoTNode("self-report", True, a["items_yes"], parent=self, model=self.model)
            node_n = UoTNode("self-report", False, a["items_no"], parent=self, model=self.model)

        exist_leaves = []
        for c in self.children:
            exist_leaves.extend([c[0], c[1]])
        if node_y in exist_leaves:
            return exist_leaves[exist_leaves.index(node_y)]
        self.children.append((node_y, node_n))
        return node_y

    def handle_free_answer(self, task, question, answer):
        repo = f"Doctor: {question}\nPatient: {answer}\n"
        a = cls_given_repo(task, self.items, repo, self_repo=False)
        exist_leaves = []
        for c in self.children:
            exist_leaves.extend([c[0], c[1]])
        node_y = UoTNode(question, True, a["items_yes"], parent=self, model=self.model, reply=answer)
        node_n = UoTNode(question, False, a["items_no"], parent=self, model=self.model, reply=answer)
        if node_y in exist_leaves:
            return exist_leaves[exist_leaves.index(node_y)]
        self.children.append((node_y, node_n))
        return node_y

    def ans2node(self, answer: bool):
        return self if self.answer == answer else next(
            (pair[0] if answer else pair[1] for pair in self.parent.children if self.question == pair[0].question),
            None
        )

    @staticmethod
    def reward_function(x, lamb=0.4):
        return ((-x * np.log2(x) - (1 - x) * np.log2(1 - x)) / (1 + abs(2 * x - 1) / lamb)) if x not in [0, 1] else 0

    def count_M_U(self):
        if not self.parent:
            return None
        c_1 = len(self.items)
        c_2 = len(self.ans2node(not self.answer).items)
        return c_1, c_2

    @property
    def idiv_reward(self):
        c = self.count_M_U()
        if not c or abs(c[0] - c[1]) == 1:
            return 1.
        return self.reward_function(c[0] / (c[0] + c[1]))

    @staticmethod
    def accumulated_reward(node, level, accum=True):
        term = 0 if (level == 1 or not accum) else node.accumulated_reward(node.parent, level - 1, accum)
        return node.idiv_reward + term

    @staticmethod
    def avg_expected(setting_node, child_list, n_extend_layers, level, prob):
        if not child_list:
            return 0
        child_r = 0.
        for child_node in child_list:
            child_node.set_config(setting_node.n_extend_layers, setting_node.accumulation, setting_node.expected_method)
            child_r += child_node.expected_reward(n_extend_layers, level=level + 1)
        return child_r * prob / len(child_list) if len(child_list) > 0 else 1

    @staticmethod
    def max_expected(setting_node, child_list, n_extend_layers, level, prob):
        if not child_list:
            return 0
        child_r = 0.
        for child_node in child_list:
            child_node.set_config(setting_node.n_extend_layers, setting_node.accumulation, setting_node.expected_method)
            child_r = max(child_node.expected_reward(n_extend_layers, level=level + 1), child_r)
        return child_r * prob

    def expected_reward(self, n_extend_layers, level=1):
        if not self.parent:
            return 1.
        c_1, c_2 = self.count_M_U()
        p = c_1 / (c_1 + c_2)
        partner = self.ans2node(not self.answer)
        if level == self.n_extend_layers - 1 or self.is_terminal or not self.children:
            return self.accumulated_reward(self, level, self.accumulation)
        else:
            expected_function = self.avg_expected if self.expected_method == 'avg' else self.max_expected
            avg_1 = expected_function(self, self.find_children_sep(), n_extend_layers, level, p)
            avg_2 = expected_function(self, partner.find_children_sep(), n_extend_layers, level, 1 - p)
        return (p * (self.idiv_reward + avg_1) +
                (1 - p) * (partner.idiv_reward + avg_2))

    @property
    def reward(self):
        self.set_config(self.parent.n_extend_layers, self.parent.accumulation, self.expected_method)
        return self.expected_reward(self.n_extend_layers)

    @property
    def is_terminal(self):
        return len(self.items) <= 2

    def print(self):
        print(
            f"""question: {self.question}; answer: {self.answer}; items: {len(self.items)}; depth: {self.depth}; is_terminal: {self.is_terminal}""")

    def __eq__(self, other):
        if isinstance(other, UoTNode):
            if len(self.items) != len(other.items) or self.depth != other.depth:
                return False
            for i in self.items:
                if i not in other.items:
                    return False
            return True
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, UoTNode):
            return self.reward < other.reward
        raise ValueError("Comparison with non-UoTNode object")

    def __gt__(self, other):
        if isinstance(other, UoTNode):
            return self.reward > other.reward
        raise ValueError("Comparison with non-UoTNode object")


def expand(task, root):
    n_layer = task.n_extend_layers
    nodes = [[] for _ in range(n_layer)]

    new_nodes = root.find_children_sep(task, task.n_potential_actions)
    if not new_nodes:
        return None
    nodes[0].extend(new_nodes)
    print(0, len(nodes[0]))

    for layer in range(1, n_layer):
        nodes[layer].extend(new_node for cur_node in nodes[layer - 1] for new_node in
                            (cur_node.find_children_sep(task, task.n_potential_actions, prun=task.n_pruned_nodes) or [cur_node]))
        if task.n_pruned_nodes > 0:
            nodes[layer] = sorted(nodes[layer], reverse=True)[:int(task.n_pruned_nodes)]
        print(layer, len(nodes[layer]))

    return nodes[n_layer - 1]


def select(task, node):
    leaf_nodes = expand(task, node)
    candidates = node.find_children_sep()
    if not leaf_nodes or not candidates:
        return None
    return max(candidates, key=lambda n: n.reward, default=None)


def renew_node_to_root(task, node, history):
    a = renew_open_set(task, history, node.items)
    node_y = UoTNode("renew", True, a, parent=task.root, model=node.model)
    node_n = UoTNode("renew", False, [], parent=task.root, model=node.model)
    exist_leaves = []
    for c in task.root.children:
        exist_leaves.extend([c[0], c[1]])
    if node_y in exist_leaves:
        return exist_leaves[exist_leaves.index(node_y)]
    task.root.children.append((node_y, node_n))
    return node_y
