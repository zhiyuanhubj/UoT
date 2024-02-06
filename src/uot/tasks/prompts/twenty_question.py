from uot.tasks.prompts.general import *

# method
generate_prompt_rest = '''Here are all the X:
{items_str}

{n} questions are designed to classify the possible X above based on the answer for these question:
{asked}
For each X under each question, if the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt = '''Here are all the X:
{items_str}

Please design a question about X and can only be answer by YES or NO. {asked} Then classify the possible X above based on this question. If the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
You should think about best {n} questions to response. And your answer should be:
Question 1: Is X ...?
YES: aaa, bbb, ...
Count of YES: ...
NO: ccc, ddd, ...
Count of NO: ...
'''

# conversation
target_question = "Is X a '{target}'?"

targeting_prompt_free = """Note that you should guess and ask what X exactly is from now on.
The question must start with 'Is X ...'"""

targeting_prompt_set = """Note that you should guess and ask what X exactly is from now on. X is possible a:
{item_list_str}, or other.
The question must start with 'Is X ...'"""

guesser_prologue = '''Let us play the game of 20 questions. I am impersonating the thing, X. You will ask me up to 20 questions which start with 'Is X' and can only be answered by yes or no, and I will answer each one truthfully based on being X.
Let us begin. Ask me the first question.
'''

urge_prompt = ""

inform_prompt = "The X may be one of items below:\n{item_list_str}"


# examiner
examiner_prologue = '''Let us play the game of 20 questions. You are the answerer and I am guesser. X is '{item}'. 
I will ask you up to 20 questions and you should answer each one truthfully based on being X. 
If I guess correctly what X is, answer me "You guessed it. X is '{item}'."
Note that never directly tell me what X is all the time.
Let us begin. Here is my first question.
'''