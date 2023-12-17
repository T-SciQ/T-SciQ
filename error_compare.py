import json
import re

def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    res = pattern.findall(ans)
    
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED" 
    return answer 

with open("data/scienceqa/pid_splits.json", mode="r", encoding="utf-8") as f:
    pid_splits = json.load(f)
    test_split = pid_splits["test"]

###
with open("experiments/answer_allenai-unifiedqa-t5-base_unifiedqa_merge_003_003/predictions_ans_test.json", mode="r", encoding="utf-8") as f:
    ans_test = json.load(f)
    preds_ps = ans_test["preds"]
    label_ps = ans_test["labels"]

error_ids_qar = []
for i, (pred, gt) in enumerate(zip(preds_ps, label_ps)):
    pred = extract_ans(pred)
    gt = extract_ans(gt)
    if pred != gt:
        error_ids_qar.append(test_split[i])  # qid

assert len(test_split) == len(preds_ps) and len(preds_ps) == len(label_ps)

error_skills_qar = {}
with open("data/scienceqa/problems_qar_003.json", mode="r", encoding="utf-8") as f:
    problems_ps = json.load(f)
    for qid in error_ids_qar:
        skill = problems_ps[qid]["skill"]
        if skill not in error_skills_qar.keys():
            error_skills_qar[skill] = 1
        else:
            error_skills_qar[skill] += 1

###
with open("experiments/answer_allenai-unifiedqa-t5-base_unifiedqa_ps003/predictions_ans_test.json", mode="r", encoding="utf-8") as f:
    ans_test = json.load(f)
    preds_paper = ans_test["preds"]
    label_paper = ans_test["labels"]

assert len(test_split) == len(preds_paper) and len(preds_paper) == len(label_paper)

error_ids_ps = []
for i, (pred, gt) in enumerate(zip(preds_paper, label_paper)):
    pred = extract_ans(pred)
    gt = extract_ans(gt)
    if pred != gt:
        error_ids_ps.append(test_split[i])  # qid

error_skills_ps = {}
with open("data/scienceqa/problems_ps_003.json", mode="r", encoding="utf-8") as f:
    problems_paper = json.load(f)
    for qid in error_ids_ps:
        skill = problems_paper[qid]["skill"]
        if skill not in error_skills_ps.keys():
            error_skills_ps[skill] = 1
        else:
            error_skills_ps[skill] += 1

###
bad_skills = [] 
for skill in error_skills_ps.keys():
    if skill not in error_skills_qar.keys():
        bad_skills.append(skill)
    else:
        if error_skills_ps[skill] > error_skills_qar[skill]:
            bad_skills.append(skill)

print(bad_skills)
print("******")
print(error_skills_qar)
print("******")
print(error_skills_ps)
num_qar = 0
for k, v in error_skills_qar.items():
    num_qar += v
print(num_qar)
num_ps = 0
for k, v in error_skills_ps.items():
    num_ps += v
print(num_ps)
