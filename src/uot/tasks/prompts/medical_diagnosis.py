from uot.tasks.prompts.general import *

# method
generate_prompt_rest = '''You are a doctor. Here are all diseases that the patient may suffer from:
{items_str}

{n} questions are designed to classify the possible diseases above based on the answer for these question:
{asked}
For each disease under each question, if the answer is 'YES', put this disease into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many diseases in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt = '''You are a doctor. Here are all the possible diseases that the patient may suffer from:
{items_str}

Please design a question to ask your patient with symptoms about disease and can only be answer by YES or NO. {asked} Then classify the possible disease above based on this question. If the answer is 'YES', put this disease into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
You should think about best {n} questions to response. And your answer should be:
Question 1: ...?
YES: aaa, bbb, ... (disease names only)
Count of YES: ...
NO: ccc, ddd, ... (disease names only)
Count of NO: ...
'''

# conversation
target_question = "Are you experiencing '{target}'?"
target_question_FA = "Are you experiencing '{target}'?"

targeting_prompt_free = """Note that you should point out and ask what disease the patient suffer from now. 
The question must be 'Are you experiencing [disease name]?' You must reply me with 1 question to ask only."""
targeting_prompt_free_FA = """Note that you should point out and ask what disease the patient suffer from now. 
The question must be 'Are you experiencing [disease name]?' You must reply me with 1 question to ask only."""

targeting_prompt_set = """Note that you should point out and ask what disease the patient suffer from now. The patient may suffer from one of diseases below:
{item_list_str}, or other.
The question must be 'Are you experiencing [disease name]?' You must reply me with 1 question."""
targeting_prompt_set_FA = """Note that you should point out and ask what disease the patient suffer from now. The patient may suffer from one of diseases below:
{item_list_str}, or other.
The question must be 'Are you experiencing [disease name]?' You must reply me with 1 question."""

guesser_prologue = '''You are a doctor and your patient self-reports that: {repo}.
You should ask your patient question in English with symptoms which can only be answered by 'Yes' or 'No', in order to find what disease this patient suffers. 
Let us begin. Ask me the first question.
'''
guesser_prologue_FA = '''You are a doctor and your patient self-reports that: {repo}.
You should ask your patient question in English with symptoms, in order to find what disease this patient suffers. 
Let us begin. Ask me the first question.
'''

urge_prompt = "Based on the symptons above, if you find out the disease, please ask 'Are you experiencing [disease name]?'"

inform_prompt = "The patient may suffer from one of diseases below:\n{item_list_str}"

# self report / free answer
classify_prompt = '''Here are all diseases that the patient may suffer from:
{item_list_str}

{repo}
For each disease under this report, if the patient is possible to have, put this disease into 'YES: ...', otherwise to 'NO: ...'. And your answer should be like:
YES: aaa, bbb, ... (disease names only)
NO: ccc, ddd, ... (disease names only)'''

self_repo_prompt = '''The patient self-reports that: {repo}'''

free_answer_prompt = '''The doctor and patient's conversation:
{repo}
'''

simulator_prologue = '''You are a patient suffering from the disease of {item}, and communicating with a doctor.
Here is your conversation history with another doctor:
{conv_hist}

Please imitate the conversation above to answer the doctor's question in English and DO NOT tell the doctor the name of the disease.
Moreover, if the doctor ask whether you experience {item}, you must answer 'You guessed it. I have {item}.'."
'''

# examiner
examiner_prologue = '''You are the patient suffering '{item}' and I am the doctor.
I will ask you up to 5 questions and you should answer each one truthfully based on your disease.
If I point out correctly what disease you experience, answer me "You are right. I am experiencing '{item}'."
Note that never directly tell me what disease is all the time.
Let us begin. Here is my first question.
'''

# open set
init_open_set_prompt = '''You are a doctor and your patient self-reports that: {repo}. Please propose {size} diseases that you think your patient may suffer from.
Your response should be: ["disease1", "disease2", ...]'''

renew_open_set_prompt = '''Based on the conversation history, please propose {size} diseases that your patient may suffer from.
The list of {size} diseases should contains {item_list}
Your response should be: ["disease1", "disease2", ...]'''