from uot.chat_utils import import_prompts_by_task
from uot.uot import UoTNode


class Q20Task:
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.free_answer = False
        self.max_turn = 20
        self.prompts = import_prompts_by_task("20q")
        self.set = []
        self.data = self.load_dataset(args.dataset)
        self.root = None

    def load_dataset(self, name):
        from uot.data.data_20q import BIG_BENCH_CONCEPT, COMMON
        if name == "bigbench":
            self.set = BIG_BENCH_CONCEPT
            return [{"target": x} for x in BIG_BENCH_CONCEPT]
        elif name == "common":
            self.set = COMMON
            return [{"target": x} for x in COMMON]
        else:
            raise NotImplementedError

    def create_root(self, root=None):
        if not root:
            self.root = UoTNode("ROOT", True, self.set, None, self.guesser_model)
        else:
            root.n_extend_layers = self.n_extend_layers
            self.root = root

