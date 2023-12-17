def get_question_text(problem):
    question = problem['question']
    return question


def get_context_text(problem, use_caption=False):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    #print(choice_txt)
    return choice_txt


def get_origin_answer(problem, options):
    return problem['choices'][problem['answer']]


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_train_prompt(pid, problems_train):
    problem = problems_train[pid]
    question = get_question_text(problem)
    context = get_context_text(problem)
    choice = get_choice_text(problem, ["A", "B", "C", "D", "E"])
    answer_option = get_answer(problem, ["A", "B", "C", "D", "E"])
    answer = "(" + answer_option + ")"
    input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    output = f"Answer: The answer is {answer}."
    return input + output


def create_test_prompt(pid, problems_test):
    problem = problems_test[pid]
    question = get_question_text(problem)
    context = get_context_text(problem)
    choice = get_choice_text(problem, ["A", "B", "C", "D", "E"])
    answer_option = get_answer(problem, ["A", "B", "C", "D", "E"])
    answer = "(" + answer_option + ")"
    lecture = get_lecture_text(problem)
    solution = get_solution_text(problem)
    input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    output = f"Answer: The answer is {answer}."
    return input + output