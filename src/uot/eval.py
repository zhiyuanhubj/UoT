import json


def evaluate_performance(file, task):
    cnt = success = 0
    length = success_length = 0
    with (open(file, 'r') as f):
        data = json.load(f)
    for i in data:
        cnt += 1
        if i['state'] == 1:
            success += 1
            success_length += i['turn']
            length += i['turn']
        else:
            length += task.max_turn

    print('Dialogue Count:', cnt)
    print('Success Rate:', success / cnt)
    print('Mean Conversation Length in Successful Cases:', success_length / success)
    print('Mean Conversation Length:', length / cnt)

