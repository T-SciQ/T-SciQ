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
lectures = {}

for k, v in demonstrations.items():
    print(k)

    instruct1 = f'Here are some problems about "{k}".\n\n'
    instruct2 = f'Based on the problems above, please give a general lecture on the "{k}" type of question in one sentence.'
    
    if len(v) > 5:
        v = random.sample(v, 5)

    prompt = ""
    for item in v:
        prompt += item
    
    content = instruct1 + prompt + instruct2
    if model == "gpt-3.5-turbo":
        messages = [{"role": "user", "content": content}]
        response = chat(model, messages)
        lecture = response["choices"][0]["message"]["content"].strip()
        lectures[k] = lecture
    else:
        response = completion(model, content)
        lecture = response["choices"][0]["text"].strip()
        lectures[k] = lecture

    with open(f"lectures.json", mode="w", encoding="utf-8") as f:
        json.dump(lectures, f, indent=2)
