import os
import time
import copy

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY != "":
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"OPENAI_API_KEY: ****{OPENAI_API_KEY[-4:]}")
else:
    print("Warning: OPENAI_API_KEY is not set")

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
if COHERE_API_KEY != "":
    import cohere

    co = cohere.Client(COHERE_API_KEY)
    print(f"COHERE_API_KEY: ****{COHERE_API_KEY[-4:]}")

PALM2_API_KEY = os.getenv("PALM2_API_KEY", "")
if PALM2_API_KEY != "":
    import google.generativeai as palm

    palm.configure(api_key=PALM2_API_KEY)
    print(f"PALM2_API_KEY: ****{PALM2_API_KEY[-4:]}")

CLAUDE2_API_KEY = os.getenv("CLAUDE2_API_KEY", "")
if CLAUDE2_API_KEY != "":
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

    anthropic = Anthropic(api_key=CLAUDE2_API_KEY, base_url="https://api.aiproxy.io")
    print(f"CLAUDE2_API_KEY: ****{CLAUDE2_API_KEY[-4:]}")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
if TOGETHER_API_KEY != "":
    from openai import OpenAI
    import os

    llama_client = OpenAI(api_key=TOGETHER_API_KEY,
                          base_url='https://api.together.xyz',
                          )

time_gap = {"gpt-4": 3, "gpt-3.5-turbo": 0.5}


def gpt_response(message: list, model="gpt-4", temperature=0, max_tokens=500):
    time.sleep(time_gap.get(model, 3))
    try:
        rps = client.chat.completions.create(model=model, messages=message, temperature=temperature, n=1,
                                             max_tokens=max_tokens)
        return rps.choices[0].message.content
    except Exception as e:
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return gpt_response(message, model, temperature, max_tokens)


def cohere_response(message: list, model=None, temperature=0, max_tokens=500):
    msg = copy.deepcopy(message[:-1])
    new_msg = message[-1]["content"]
    for m in msg:
        m.update({"role": "CHATBOT" if m["role"] == "system" else "USER", "message": m.pop("content")})

    try:
        return co.chat(chat_history=msg, message=new_msg).text
    except Exception as e:
        print(e)
        time.sleep(1)
        return cohere_response(message)


def palm_response(message: list, model=None, temperature=0, max_tokens=500):
    msg = [{'author': '1' if m["role"] == "user" else '0', **m} for m in message]
    for m in msg:
        m.pop("role", None)
    try:
        rsp = palm.chat(messages=msg)
        return rsp.last
    except Exception as e:
        print(e)
        time.sleep(1)
        return palm_response(message, temperature=temperature)


def claude_response(message, model=None, temperature=0, max_tokens=500):
    prompt = ""
    for m in message:
        prompt += AI_PROMPT if m["role"] in ["system", "assistant"] else HUMAN_PROMPT
        prompt += " " + m["content"]
    prompt += AI_PROMPT
    try:
        res = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            prompt=prompt,
        )
        return res.completion
    except Exception as e:
        print(e)
        time.sleep(1)
        return claude_response(message, model, temperature, max_tokens)


def llama_response(message, model=None, temperature=0, max_tokens=500):
    try:
        chat_completion = llama_client.chat.completions.create(
            messages=message,
            model="meta-llama/Llama-2-70b-chat-hf",
            max_tokens=max_tokens
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        time.sleep(1)
        llama_response(message, model, temperature, max_tokens)


def get_response_method(model):
    response_methods = {
        "gpt": gpt_response,
        "cohere": cohere_response,
        "palm": palm_response,
        "claude": claude_response,
        "llama": llama_response
    }
    return response_methods.get(model.split("-")[0], lambda _: NotImplementedError())