import json

# Q-A-CoT prediction json path
# experiments/xxx/predictions_ans_test.json
qa_cot_path = "experiments/xxx/predictions_ans_test.json"
# Q-A-PCoT prediction json path
# experiments/xxx/predictions_ans_test.json
qa_pcot_path = "experiments/xxx/predictions_ans_test.json"

with open("data/scienceqa/pid_splits.json", mode="r", encoding="utf-8") as f:
    pid_splits = json.load(f)
    test_split = pid_splits["test"]

with open(qa_cot_path, mode="r", encoding="utf-8") as f:
    ans_test = json.load(f)
    preds_cot = ans_test["preds"]
    label_cot = ans_test["labels"]

error_ids_cot = []
for i, (pred, gt) in enumerate(zip(preds_cot, label_cot)):
    if pred != gt:
        error_ids_cot.append(test_split[i])  # qid

assert len(test_split) == len(preds_cot) and len(preds_cot) == len(label_cot)

error_skills_cot = {}
with open("data/scienceqa/problems.json", mode="r", encoding="utf-8") as f:
    problems = json.load(f)
    for qid in error_ids_cot:
        skill = problems[qid]["skill"]
        if skill not in error_skills_cot.keys():
            error_skills_cot[skill] = 1
        else:
            error_skills_cot[skill] += 1

with open(qa_pcot_path, mode="r", encoding="utf-8") as f:
    ans_test = json.load(f)
    preds_pcot = ans_test["preds"]
    label_pcot = ans_test["labels"]

assert len(test_split) == len(preds_pcot) and len(preds_pcot) == len(label_pcot)

error_ids_pcot = []
for i, (pred, gt) in enumerate(zip(preds_pcot, label_pcot)):
    if pred != gt:
        error_ids_pcot.append(test_split[i])  # qid

error_skills_pcot = {}
with open("data/scienceqa/problems.json", mode="r", encoding="utf-8") as f:
    problems = json.load(f)
    for qid in error_ids_pcot:
        skill = problems[qid]["skill"]
        if skill not in error_skills_pcot.keys():
            error_skills_pcot[skill] = 1
        else:
            error_skills_pcot[skill] += 1

bad_skills = [] 
for skill in error_skills_pcot.keys():
    if skill not in error_skills_cot.keys():
        bad_skills.append(skill)
    else:
        if error_skills_pcot[skill] > error_skills_cot[skill]:
            bad_skills.append(skill)


with open("data/scienceqa/problems_cot.json", mode="r", encoding="utf-8") as f1:
    problems_cot = json.load(f1)
with open("data/scienceqa/problems_pcot.json", mode="r", encoding="utf-8") as f2:
    problems_pcot = json.load(f2)

problems = {}
for k, v in problems_pcot.items():
    if v["skill"] in bad_skills:
        problems[k] = problems_cot[k]
    else:
        problems[k] = problems_pcot[k]
with open("data/scienceqa/problems-T-SciQ.json", mode="w", encoding="utf-8") as f:
    json.dump(problems, f, indent=2)
