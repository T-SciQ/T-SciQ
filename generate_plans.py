import json
import openai
import random

openai.api_key = "sk-"

# model = "gpt-3.5-turbo"
# model = "text-davinci-002"
model = "text-davinci-003"
option_names = ["(A)", "(B)", "(C)", "(D)", "(E)"]


def chat(model, messages):
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=512)
    return response

def completion(model, prompt):
    response = openai.Completion.create(model=model, prompt=prompt, max_tokens=512, temperature=0)
    return response


with open(f"skill_demonstrations.json", mode="r", encoding="utf-8") as f:
    demonstrations = json.load(f)
with open(f"lectures.json", mode="r", encoding="utf-8") as f:
    lectures = json.load(f)
plans = {}

for k, v in demonstrations.items():
    print(k)

    instruct1 = f'Here are some problems about "{k}".\n\n'
    instruct2 = f'The lecture about "{k}" is "{lectures[k]}"\n\n'
    instruct3 = f'Based on the lecture above and these problems, let\'s understand these problems and devise a general and brief plan step by step to solve these problems (begin with 1, 2, 3...).'
    
    if len(v) > 5:
        v = random.sample(v, 5)

    prompt = ""
    for item in v:
        prompt += item
    
    content = instruct1 + instruct2 + prompt + instruct3
    if model == "gpt-3.5-turbo":
        messages = [{"role": "user", "content": content}]
        response = chat(model, messages)
        plan = response["choices"][0]["message"]["content"].strip()
        plans[k] = plan
    else:
        response = completion(model, content)
        plan = response["choices"][0]["text"].strip()
        plans[k] = plan

    with open(f"plans.json", mode="w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2)
