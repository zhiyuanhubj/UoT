import copy
import importlib

from uot.models import get_response_method

task_parameter_mapping = {
    "20q": "twenty_question",
    "md": "medical_diagnosis",
    "tb": "troubleshooting",
}


def import_prompts_by_task(task_name):
    parameter = task_parameter_mapping.get(task_name)
    module_name = f"uot.tasks.prompts.{parameter}"
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        raise ImportError(f"Failed to import module: {module_name}")


def ques_and_cls_given_items(task, items: list, n, asked_ques: list = None, rest=False):
    response = get_response_method(task.guesser_model)
    if len(items) <= 1:
        return None

    if rest:
        asked = '\n'.join([f"Question {i + 1}: {asked_ques[i]}" for i in range(len(asked_ques))])
        message = [{"role": "user", "content": task.prompts.generate_prompt_rest.format(
            items_str=', '.join(items), n=n, asked=asked, Q1=asked_ques[0])}]
    else:
        asked = "(The question should not be '" + "' or '".join(asked_ques) + "')" if asked_ques else ""
        message = [{"role": "user", "content": task.prompts.generate_prompt.format(items_str=', '.join(items), n=n, asked=asked)}]
    print(message)
    rsp = "#" + response(message, model=task.guesser_model, max_tokens=2000)
    print([rsp])

    def process_ans(rsp):
        ans = []
        for i in range(n):
            if f"Question {i + 1}: " not in rsp:
                continue
            rsp = rsp.split(f"Question {i + 1}: ", 1)[1]
            q = rsp.split("\n", 1)[0]
            rsp = rsp.split("YES: ", 1)[1]
            if rsp[0] == '\n':
                continue
            items_y = rsp.split("\n", 1)[0].split(", ")
            items_y = list(set(items_y))
            rsp = rsp.split("\nNO: ", 1)[1] if "\nNO: " in rsp else rsp.split("NO: ", 1)[1]
            if rsp[0] == '\n':
                continue
            items_n = rsp.split("\n", 1)[0].split(", ")
            items_n = list(set(items_n))
            ans.append({"question": q, "items_yes": items_y, "items_no": items_n})
        return ans

    def format_rsp(rsp):
        gpt3_response = get_response_method("gpt-3.5-turbo")
        message.append({"role": "system", "content": rsp})
        message.append({"role": "user", "content": task.prompts.format_generated_prompt.format(rsp=rsp)})
        return gpt3_response(message, "gpt-3.5-turbo", max_tokens=500)

    try:
        return process_ans(rsp)
    except Exception:
        try:
            rsp = format_rsp(rsp)
            return process_ans(rsp)
        except Exception as e:
            print(e)
            return ques_and_cls_given_items(task, items, n, asked_ques, rest)


def cls_given_repo(task, items: list, repo, translate=False, self_repo=True):
    response = get_response_method(task.guesser_model)
    if self_repo:
        if translate:
            message = [{"role": "user", "content": f"Translate to English: {repo}"}]
            gpt3_response = get_response_method("gpt-3.5-turbo")
            repo = gpt3_response(message, model="gpt-3.5-turbo", max_tokens=500)
        repo = task.prompts.self_repo_prompt.format(repo=repo)
    else:
        repo = task.prompts.free_answer_prompt.format(repo=repo)
    message = [{"role": "user", "content": task.prompts.classify_prompt.format(item_list_str=', '.join(items), repo=repo)}]
    rsp = response(message, model=task.guesser_model, max_tokens=len(items)*(task.expected_target_tokens+5))
    print([rsp])

    def extract_items(rsp, keyword):
        _items = []
        if keyword in rsp:
            rsp_part = rsp.split(keyword, 1)[1]
            if not rsp_part or rsp_part[0] != '\n':
                _items = rsp_part.split("\n", 1)[0].split(", ")
                _items = list(set(_items))
        return _items

    try:
        items_y = extract_items(rsp, "YES: ")
        items_n = extract_items(rsp, "NO: ")
        if len(items_y) == 0 and len(items_n) == 0:
            raise ValueError("No items extracted from the response.")

        return {"items_yes": items_y, "items_no": items_n}

    except Exception as e:
        print(e)
        return cls_given_repo(task, items, repo, translate, self_repo)


def initialize_open_set(task, repo=""):
    response = get_response_method(task.guesser_model)
    size = task.open_set_size
    
    if isinstance(repo, str):
        message = [{"role": "user", "content": task.prompts.init_open_set_prompt.format(repo=repo, size=size)}]
    else:
        message = repo + [{"role": "user", "content": task.prompts.init_open_set_prompt.format(size=size)}]
    rsp = response(message, model=task.guesser_model, max_tokens=15*size)
    print([rsp])
    try:
        rsp = set(eval(rsp))
        return list(rsp)
    except Exception as e:
        print(e)
        return initialize_open_set(task, repo)


def renew_open_set(task, history, items):
    response = get_response_method(task.guesser_model)
    size = task.open_set_size
    message = copy.deepcopy(history) + [{"role": "user", "content": task.prompts.renew_open_set_prompt.format(size=size, item_list=str(items))}]
    rsp = response(message, model=task.guesser_model, max_tokens=15*size)
    print([rsp])
    try:
        rsp = set(eval(rsp))
        return list(rsp)
    except Exception as e:
        print(e)
        return renew_open_set(task, history, items)