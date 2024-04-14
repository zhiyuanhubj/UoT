from uot.tasks.prompts.general import *

# method
generate_prompt_rest = '''You are a technician. Here are all issues that the client may face with:
{items_str}

{n} questions are designed to classify the possible issues above based on the answer for these question:
{asked}
For each issue under each question, if the answer is 'YES', put this issue into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many issues in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt = '''You are a technician. Here are all issues that the client may face with:
{items_str}

Please design a question to ask your client with specific situation and can only be answer by YES or NO. {asked} Then classify the possible issue above based on this question. If the answer is 'YES', put this issue into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many issues in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
You should think about best {n} questions to response. And your answer should be:
Question 1: ...?
YES: aaa, bbb, ... (issue names only)
Count of YES: ...
NO: ccc, ddd, ... (issue names only)
Count of NO: ...
'''

# conversation
target_question = "Are you experiencing '{target}'?"

targeting_prompt_free = """Note that you should point out and ask what issue the client faces with now. 
The question must be 'Are you experiencing [issue name]?'"""

targeting_prompt_set = """Note that you should point out and ask what issue the client faces with now. The client may face with one of issues below:
{item_list_str}, or other.
The question must be 'Are you experiencing [issue name]?'"""

guesser_prologue = '''You are a technician, and your client self-reports that: {repo}.
You should ask your client question with specific situation which can only be answered by 'Yes' or 'No', in order to find which issue this client faces with.
Let us begin. Ask me the first question.
'''

urge_prompt = "Based on the situations above, if you find out the issue, please ask 'Are you experiencing [issue name]?'"

inform_prompt = "The client may face one of issues below:\n{item_list_str}"

# self report / free answer
classify_prompt = '''Here are all issues that the client may face with:
{item_list_str}

{repo}
For each issue under this report, if the client is possible to face with, put this issue into 'YES: ...', otherwise to 'NO: ...'. And your answer should be like:
YES: aaa, bbb, ... (issue names only)
NO: ccc, ddd, ... (issue names only)'''

self_repo_prompt = '''The client self-reports that: {repo}'''

# examiner
examiner_prologue = '''You are the client with a device that has '{item}' and I am the technician.
I will ask you up to 20 questions and you should answer each one truthfully based on the issue of your device.
If I point out correctly what your issue is, answer me "You are right. My device has '{item}'."
Note that never directly tell me what the issue is all the time.
Let us begin. Here is my first question.
'''

# open set
init_open_set_prompt = '''You are a technician, and your client self-reports that: {repo}. Please propose {size} issues that you think your client may face with.
Your response should be: ["issue1", "issue2", ...]'''

renew_open_set_prompt = '''Based on the conversation history, please propose {size} issues that your client may face with.
The list of {size} issues should contains {item_list}
Your response should be: ["issue1", "issue2", ...]'''