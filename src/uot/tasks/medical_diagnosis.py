import os
import json

from uot.chat_utils import import_prompts_by_task
from uot.uot import UoTNode


class MDTask:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = False
        self.max_turn = 5
        self.prompts = import_prompts_by_task("md")
        self.set = []
        self.data = json.loads(args.dataset)
        self.root = None

    def load_dataset(self, name):
        if name == "DX":
            self.set = ['Allergic rhinitis', 'upper respiratory tract infection (URTI)', 'pneumonia',
                        'Hand foot and mouth disease in children', 'Infantile diarrhea']\
                if self.open_set_size <= 0 else self.set
        elif name == "MedDG":
            self.free_answer = True
            self.set = ['Enteritis', 'Gastritis', 'Gastroenteritis', 'Esophagitis',
                        'Cholecystitis', 'Appendicitis', 'Pancreatitis', 'Gastric ulcer',
                        'Constipation', 'Cold', 'Irritable bowel syndrome', 'Diarrhea',
                        'Allergic rhinitis', 'Upper respiratory tract infection', 'Pneumonia']\
                if self.open_set_size <= 0 else self.set
        else:
            raise NotImplementedError
        return json.loads(os.path.join(os.path.dirname(__file__), f"../data/{name}.json").read())

    def create_root(self, root=None):
        if not root:
            self.root = UoTNode("ROOT", True, self.set, None, self.guesser_model)
        else:
            root.set_config(self.n_extend_layers, not self.none_acc_reward, self.expected_reward_method)
            self.root = root
