import os

from uot.chat_utils import import_prompts_by_task
from uot.uot import UoTNode


class MDTask:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = False
        self.max_turn = 5
        self.prompts = import_prompts_by_task("md")
        self.set = []
        self.data = self.load_dataset(args.dataset)
        self.root = None

    def load_dataset(self, name):
        if name == "DX":
            self.set = ['Allergic rhinitis', 'upper respiratory tract infection (URTI)', 'pneumonia',
                        'Hand foot and mouth disease in children', 'Infantile diarrhea']
            return load_dx_dataset(os.path.join(os.path.dirname(__file__), "../data/DX_dialog.txt"))
        elif name == "MedDG":
            self.free_answer = True
            self.set = ['Enteritis', 'Gastritis', 'Gastroenteritis', 'Esophagitis',
                        'Cholecystitis', 'Appendicitis', 'Pancreatitis', 'Gastric ulcer']
            return load_meddg_dataset(os.path.join(os.path.dirname(__file__), "../data/MedDG_dialog.txt"))
        else:
            raise NotImplementedError

    def create_root(self, root=None):
        if not root:
            self.root = UoTNode("ROOT", True, self.set, None, self.guesser_model)
        else:
            root.n_extend_layers = self.n_extend_layers
            self.root = root


def load_dx_dataset(file_path):
    dic = {"过敏性鼻炎": 'Allergic rhinitis', "肺炎": 'pneumonia', "小儿腹泻": 'Infantile diarrhea',
           "上呼吸道感染": 'upper respiratory tract infection (URTI)',
           "小儿手足口病": 'Hand foot and mouth disease in children'}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = eval(f.read())
        repo_dataset = [{'self_repo': dialog['self_repo_en'], 'target': dic[dialog['disease_tag']]} for dialog in data]
    return repo_dataset


def load_meddg_dataset(file_path):
    repo_dataset = []
    flag = 0
    disease, self_repo, dialog = "", "", ""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("dialog"):
                flag = 1
                disease = line.split("|")[1][:-1]
            elif line.startswith("{"):
                content = eval(line)
                dialog += f"{content['id']}: {content['Sentence']}\n"
                if flag:
                    self_repo = content['self_repo_en']
                    flag = 0
            else:
                repo_dataset.append({'self_repo': self_repo, 'target': disease, 'conv_hist': dialog})
                disease, self_repo, dialog = "", "", ""
    return repo_dataset
