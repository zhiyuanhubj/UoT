import json


def evaluate_performance(file, task):
    cnt = success = 0
    efficiency = success_efficiency = 0
    with (open(file, 'r') as f):
        data = json.load(f)
    for i in data:
        cnt += 1
        if i['state'] == 1:
            success += 1
            success_efficiency += i['turn']
            efficiency += i['turn']
        else:
            efficiency += task.max_turn

    print('Dialogue Count:', cnt)
    print('Success Rate:', success / cnt)
    print('Efficiency in Successful Cases:', success_efficiency / success)
    print('Efficiency:', efficiency / cnt)
