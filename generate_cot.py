import json
import openai

openai.api_key = "sk-"

# model = "gpt-3.5-turbo"
# model = "text-davinci-002"
model = "text-davinci-003"
option_names = ["(A) ", "(B) ", "(C) ", "(D) ", "(E) "]


def chat(model, messages):
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=512)
    return response

def completion(model, prompt):
    response = openai.Completion.create(model=model, prompt=prompt, max_tokens=512, temperature=0)
    return response

with open(f"problems.json", mode="r", encoding="utf-8") as f:
    problems = json.load(f)
new_problems = {}

for k, v in problems.items():
    print(k)

    instruct = f"Please give me a detailed explanation."

    new_problems[k] = v
    new_problems[k]["lecture"] = ""

    question = f"Question: {v['question']}\n"
    context = 'Context: N/A\n' if len(v["hint"]) == 0 else f"Context: {v['hint']}\n"
    choices = v["choices"]
    option = "Options: "
    for i, choice in enumerate(choices):
        option += option_names[i] + choices[i] + " "
    option = option.strip() + "\n"
    answer = f"Correct Answer: {option_names[v['answer']].strip()} {choices[v['answer']]}\n"

    sample = question + context + option + answer
    content = sample + instruct

    if model == "gpt-3.5-turbo":
        messages = [{"role": "user", "content": content}]
        response = chat(model, messages)
        solution = response["choices"][0]["message"]["content"]
        new_problems[k]["solution"] = solution.strip()
    else:
        response = completion(model, content)
        solution = response["choices"][0]["text"]
        new_problems[k]["solution"] = solution.strip()

    with open(f"problems_cot.json", mode="w", encoding="utf-8") as f:
        json.dump(new_problems, f, indent=2)
