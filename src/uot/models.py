import os
import time
import copy

CNT = 0

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
time_gap = {"gpt-4": 3, "gpt-3.5-turbo": 0.5}
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY != "":
    # import google.generativeai as palm
    import google.generativeai as genai
    import google.ai.generativelanguage as glm

    # palm.configure(api_key=GOOGLE_API_KEY)
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"GOOGLE_API_KEY: ****{GOOGLE_API_KEY[-4:]}")

CLAUDE2_API_KEY = os.getenv("CLAUDE2_API_KEY", "")
if CLAUDE2_API_KEY != "":
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

    anthropic = Anthropic(api_key=CLAUDE2_API_KEY, base_url="https://api.aiproxy.io")
    print(f"CLAUDE2_API_KEY: ****{CLAUDE2_API_KEY[-4:]}")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if ANTHROPIC_API_KEY != "":
    from anthropic import Anthropic

    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"ANTHROPIC_API_KEY: ****{ANTHROPIC_API_KEY[-4:]}")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
if TOGETHER_API_KEY != "":
    from openai import OpenAI

    llama_client = OpenAI(api_key=TOGETHER_API_KEY, base_url='https://api.together.xyz')
    print(f"TOGETHER_API_KEY: ****{TOGETHER_API_KEY[-4:]}")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
if MISTRAL_API_KEY != "":
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage

    mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
    print(f"MISTRAL_API_KEY: ****{MISTRAL_API_KEY[-4:]}")
    
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY != "":
    import os
    from groq import Groq

    groq_client = Groq(api_key=GROQ_API_KEY)
    print(f"GROQ_API_KEY: ****{GROQ_API_KEY[-4:]}")
    

def success_call():
    global CNT
    CNT = 0
    

def fail_call():
    global CNT
    CNT += 1
    if CNT > 20:
        assert 0


def gpt_response(message: list, model="gpt-4", temperature=0, max_tokens=500):
    time.sleep(time_gap.get(model, 3))
    try:
        res = client.chat.completions.create(model=model, messages=message, temperature=temperature, n=1,
                                             max_tokens=max_tokens)
        success_call()
        return res.choices[0].message.content
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(time_gap.get(model, 3) * 2)
        return gpt_response(message, model, temperature, max_tokens)


def cohere_response(message: list, model=None, temperature=0, max_tokens=500):
    msg = copy.deepcopy(message[:-1])
    new_msg = message[-1]["content"]
    for m in msg:
        m.update({"role": "CHATBOT" if m["role"] == "system" else "USER", "message": m.pop("content")})
    try:
        res = co.chat(chat_history=msg, message=new_msg)
        success_call()
        return res.text
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(1)
        return cohere_response(message)


# def palm_response(message: list, model=None, temperature=0, max_tokens=500):
#     msg = [{'author': '1' if m["role"] == "user" else '0', **m} for m in message]
#     for m in msg:
#         m.pop("role", None)
#     try:
#         res = palm.chat(messages=msg)
#         success_call()
#         return res.last
#     except Exception as e:
#         print(e)
#         fail_call()
#         time.sleep(1)
#         return palm_response(message, temperature=temperature)
    

def gemini_response(message: list, model="gemini-1.0-pro", temperature=0, max_tokens=500):
    time.sleep(10)
    msg = []
    for m in message[:-1]:
        role = "user" if m["role"] == "user" else "model"
        msg.append(glm.Content(parts=[glm.Part(text=m["content"])], role=role))
    genai_model = genai.GenerativeModel(model_name=model)
    try:
        chat = genai_model.start_chat()
        res = chat.send_message(message[-1]["content"])
        success_call()
        return res.text  
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(20)
        return gemini_response(message, model, temperature, max_tokens)


def claude_aiproxy_response(message, model=None, temperature=0, max_tokens=500):
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
        success_call()
        return res.completion
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(1)
        return claude_aiproxy_response(message, model, temperature, max_tokens)


def claude_response(message, model="claude-3-sonnet-20240229", temperature=0, max_tokens=500):
    msg = []
    for m in message:
        role = m["role"] if m["role"] == "user" else "assistant"
        if msg and msg[-1]["role"] == role:
            msg[-1]["content"] += m["content"]
        else:
            msg.append({"role": role, "content": m["content"]})
    try:
        res = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=msg
        )
        success_call()
        return res.content[0].text
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(3)
        return claude_response(message, model, temperature, max_tokens)


def llama_response(message, model=None, temperature=0, max_tokens=500):
    try:
        chat_completion = llama_client.chat.completions.create(
            messages=message,
            model="meta-llama/Llama-2-70b-chat-hf",
            max_tokens=max_tokens
        )
        success_call()
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(1)
        llama_response(message, model, temperature, max_tokens)

def llama_groq_response(message: list, model="llama3-8b-8192", temperature=0, max_tokens=500):
    time.sleep(5)
    try:
        res = groq_client.chat.completions.create(model=model, messages=message, temperature=temperature,
                                                        max_tokens=max_tokens)
        success_call()
        return res.choices[0].message.content
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(15)
        return llama_groq_response(message, model, temperature, max_tokens)


def mistral_response(message: list, model="mistral-large-latest", temperature=0, max_tokens=500):
    msg = [ChatMessage(role=m["role"], content=m["content"]) for m in message]
    try:
        res = mistral_client.chat(model=model, messages=msg)
        success_call()
        return res.choices[0].message.content
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(1)
        return mistral_response(message, model, temperature, max_tokens)


def gemma_response(message: list, model=None, temperature=0, max_tokens=500):
    import uot.model_gemma as gm
    try:
        res = gm.gemma_response(history=message, output_len=max_tokens)
        success_call()
        return res
    except Exception as e:
        print(e)
        fail_call()
        time.sleep(1)
        return gemma_response(message, model, temperature, max_tokens)


def get_response_method(model):
    response_methods = {
        "gpt": gpt_response,
        "cohere": cohere_response,
        # "palm": palm_response,
        "_claude": claude_aiproxy_response,
        "claude": claude_response,
        "llama": llama_response,
        "llama3": llama_groq_response,
        "mistral": mistral_response,
        "gemma": gemma_response,
        "gemini": gemini_response,
    }
    return response_methods.get(model.split("-")[0], lambda _: NotImplementedError())

if __name__ == '__main__':
    # print(gpt_response([{"role": "user", "content": "Hi"}], model="gpt-4", temperature=0, max_tokens=20))
    print(claude_response([{"role": "user", "content": "Hi"}], model="claude-3-opus-20240229", temperature=0, max_tokens=20))