target_declaration = "Target is '{target}'"

extract_q_prompt = '''{rsp}

Extract the question it want to ask'''

format_generated_prompt = '''{rsp}

Rewrite the response following the format:
Question 1: ...?
YES: aaa, bbb, ...
Count of YES: ...
NO: ccc, ddd, ...
Count of NO: ...
Question 2: ...
'''
