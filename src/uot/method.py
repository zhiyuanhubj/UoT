import copy

from uot.chat_utils import renew_open_set
from uot.models import get_response_method
from uot.uot import select, renew_node_to_root


def get_examiner_response(task, history):
    response = get_response_method(task.examiner_model)
    msg = [history[0]] + history[-3:] if len(history) > 3 else history
    return response(msg, model=task.examiner_model)


def get_guesser_response(task, history, ques_id, node):
    response = get_response_method(task.guesser_model)

    def simplify_rsp(rsp):
        gpt3_response = get_response_method("gpt-3.5-turbo")
        if len(rsp.split(" ")) > task.expected_action_tokens:
            m = [{"role": "user", "content": task.prompts.extract_q_prompt.format(rsp=rsp)}]
            rsp = gpt3_response(m, model="gpt-3.5-turbo", max_tokens=task.expected_action_tokens)
        return rsp

    if len(node.items) == 1:
        target_question = task.prompts.target_question_FA if task.free_answer else task.prompts.target_question
        if target_question.format(target=node.items[0]) not in [h["content"] for h in history]:
            return node, target_question.format(target=node.items[0]), False
        else:
            targeting_prompt_free = task.prompts.targeting_prompt_free_FA if task.free_answer else task.prompts.targeting_prompt_free
            msg = copy.deepcopy(history) + [{"role": "user", "content": targeting_prompt_free}]
            return node, simplify_rsp(response(msg, model=task.guesser_model)), False

    if ques_id < int(task.max_turn*0.6):
        n = select(task, node)
        if n:
            return n, n.question, True

    targeting_prompt_set = task.prompts.targeting_prompt_set_FA if task.free_answer else task.prompts.targeting_prompt_set
    msg = copy.deepcopy(history) + [{"role": "user", "content": targeting_prompt_set.format(item_list_str=', '.join(node.items))}]
    return node, simplify_rsp(response(msg, model=task.guesser_model)), False


def get_guesser_naive_response(task, history, ques_id):
    response = get_response_method(task.guesser_model)

    msg = copy.deepcopy(history)
    prompt = ""
    if ques_id > int(task.max_turn*0.7):
        prompt += task.prompts.urge_prompt
        if task.inform:
            prompt += task.prompts.inform_prompt.format(item_list_str=', '.join(task.set))
    prompt += "\nYou must reply me with 1 question to ask only."
    msg[-1]["content"] += " " + prompt
    rsp = response(msg, model=task.guesser_model)

    def extract_ques(rsp):
        gpt3_response = get_response_method("gpt-3.5-turbo")
        message = [{"role": "user", "content": task.prompts.extract_q_prompt.format(rsp=rsp)}]
        return gpt3_response(message, model="gpt-3.5-turbo")

    return extract_ques(rsp) if len(rsp.split(" ")) > task.expected_action_tokens else rsp


def converse(task, i):
    item = task.data[i]["target"]
    target_decl = task.prompts.target_declaration.format(target=item)
    print(target_decl)
    print("------ DIALOGUE START ------")
    count = 0

    if not task.free_answer:
        history_e = [{'role': 'user', 'content': task.prompts.examiner_prologue.format(item=item)}]
    else:
        history_e = [{'role': 'user', 'content': task.prompts.simulator_prologue.format(item=item, conv_hist=task.data[i]["conv_hist"])}]

    if "self_repo" in task.data[i]:
        guesser_prologue = task.prompts.guesser_prologue_FA if task.free_answer else task.prompts.guesser_prologue
        history_g = [{'role': 'user', 'content': guesser_prologue.format(repo=task.data[i]["self_repo"])}]
        print("Self-report:", task.data[i]["self_repo"])
        node = task.root.handle_self_repo(task, task.data[i]["self_repo"])
    else:
        history_g = [{'role': 'user', 'content': task.prompts.guesser_prologue}]
        # !! for openset uot !!
        if task.open_set_size > 0 and task.n_pre_ask > 0:
            for _ in range(task.n_pre_ask):
                bot1_response = get_guesser_naive_response(task, history_g, count+1)
                print("Bot 2:", bot1_response)
                history_g.append({'role': 'system', 'content': bot1_response})
                history_e.append({'role': 'user', 'content': bot1_response})
                bot2_response = get_examiner_response(task, history_e)
                print("Bot 1:", bot2_response)
                history_g.append({'role': 'user', 'content': bot2_response})
                history_e.append({'role': 'system', 'content': bot2_response})
                count += 1
                print('------', count, '-------------')
        node = task.root.handle_self_repo(task, history_g) if task.open_set_size > 0 else task.root

    node, bot1_response, flag = get_guesser_response(task, history_g, count + 1, node)
    print("Bot 2:", bot1_response)

    history_g.append({'role': 'system', 'content': bot1_response})
    history_e.append({'role': 'user', 'content': bot1_response})

    while True:
        bot2_response = get_examiner_response(task, history_e)  # chatbot 2 is the examiner
        if task.free_answer and flag:
            node = node.handle_free_answer(task, bot1_response, bot2_response)
        elif bot2_response.startswith("Yes"):
            node = node.ans2node(True)
        elif bot2_response.startswith("No"):
            node = node.ans2node(False)
        history_g.append({'role': 'user', 'content': bot2_response})
        history_e.append({'role': 'system', 'content': bot2_response})
        print("Bot 1:", bot2_response)

        if "guessed it" in bot2_response or "are right." in bot2_response:
            state = 1
            break

        count += 1
        print('------', count, '-------------')

        if count >= task.max_turn:
            print("Bot 1: Sorry, time's up. You lose this game.", target_decl)
            state = -1
            break

        # renew
        if count <= int(task.max_turn*0.3) + task.n_pre_ask and task.open_set_size > 0 and len(node.items) < task.size_to_renew:
            node = renew_node_to_root(task, node, history_g)

        node, bot1_response, flag = get_guesser_response(task, history_g, count + 1, node)
        print("Bot 2:", bot1_response)
        history_g.append({'role': 'system', 'content': bot1_response})
        history_e.append({'role': 'user', 'content': bot1_response})

    if count < task.max_turn:
        state = 1

    return {'turn': count, 'history_g': history_g, 'history_e': history_e, 'state': state, 'item': task.data[i]["target"]}


def naive_converse(task, i):
    item = task.data[i]["target"]
    target_decl = task.prompts.target_declaration.format(target=item)
    print(target_decl)

    if "self_repo" in task.data[i]:
        guesser_prologue = task.prompts.guesser_prologue_FA if task.free_answer else task.prompts.guesser_prologue
        history_g = [{'role': 'user', 'content': guesser_prologue.format(repo=task.data[i]["self_repo"])}]
        print("Self-report:", task.data[i]["self_repo"])
    else:
        history_g = [{'role': 'user', 'content': task.prompts.guesser_prologue}]

    if not task.free_answer:
        history_e = [{'role': 'user', 'content': task.prompts.examiner_prologue.format(item=item)}]
    else:
        history_e = [{'role': 'user', 'content': task.prompts.simulator_prologue.format(item=item, conv_hist=task.data[i]["conv_hist"])}]

    print("------ DIALOGUE START ------")
    count = 0

    bot1_response = get_guesser_naive_response(task, history_g, count+1)
    print("Bot 2:", bot1_response)

    history_g.append({'role': 'system', 'content': bot1_response})
    history_e.append({'role': 'user', 'content': bot1_response})

    while True:
        bot2_response = get_examiner_response(task, history_e)
        history_g.append({'role': 'user', 'content': bot2_response})
        history_e.append({'role': 'system', 'content': bot2_response})
        print("Bot 1:", bot2_response)

        if "guessed it" in bot2_response or "are right." in bot2_response:
            state = 1
            break

        count += 1
        print('------', count, '-------------')

        if count >= task.max_turn:
            print("Bot 1: Sorry, time's up. You lose this game.", target_decl)
            state = -1
            break

        bot1_response = get_guesser_naive_response(task, history_g, count+1)
        print("Bot 2:", bot1_response)
        history_g.append({'role': 'system', 'content': bot1_response})
        history_e.append({'role': 'user', 'content': bot1_response})

    if count < task.max_turn:
        state = 1

    return {'turn': count, 'history_g': history_g, 'history_e': history_e, 'state': state, 'item': task.data[i]["target"]}
