# Copyright (c) Alibaba, Inc. and its affiliates.
COT_FLAG = '\nOverall, the answer is:\n\n'

def format_qa(qa_data):
    qa_tmpl = "# Question\n{question}\n\n# Answer\n{answer}"
    question, answer = qa_data['question'], qa_data['answer']
    return qa_tmpl.format(question=question, answer=answer)


def format_question(question, task_prompt):
    prompt_tmpl = "{task_prompt}\n\n# Question\n{question}\n\n# Answer\n"
    return prompt_tmpl.format(task_prompt=task_prompt, question=question)


def format_answer(answer, explanation=None):
    if explanation:
        answer = f'{explanation}{COT_FLAG}{answer}'
    return answer


def parse_answer(answer: str):
    if COT_FLAG in answer:
        answer = answer.split(COT_FLAG)[1]
    return answer