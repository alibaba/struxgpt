# Copyright (c) Alibaba, Inc. and its affiliates.


def format_demo_qa(src_dict):
    question, options, answer_id, rationale = \
        src_dict['question'], src_dict['options'], src_dict['answer_idx'], src_dict['rationale']
    question += '\nOptions:\n' + '\n'.join(f'{idx}. {opt}' for idx, opt in options.items())
    if isinstance(answer_id, list):
        answer_id = ",".join(answer_id)
    answer_id = answer_id.replace(' OR ', ',')

    return question, answer_id, rationale


def format_trainval_qa(lang, src_dict, is_with_rationale, use_alpaca=True, 
                       use_cot=False, rag=False, retriever=None, **kwargs):
    question = src_dict["question"]
    options = kwargs.get('options', '')
    if options == '':
        for key in src_dict["options"].keys():
            content = src_dict["options"][key]
            options += f"{key}. {content}\n"
    answer_id = kwargs.get("answer_id", '')
    if answer_id == '':
        if isinstance(src_dict["answer_idx"], str):
            answer_id = src_dict["answer_idx"]
        elif isinstance(src_dict["answer_idx"], list):
            answer_id = ",".join(src_dict["answer_idx"])
        answer_id = answer_id.replace(' OR ', ',')
    rationale = src_dict["rationale"]
    
    if rag:
        assert not use_cot
        references = retriever.retrieve(question, return_text=True)
        references = '\n'.join(references)
    else:
        references = None
    
    alpaca_prompt_template = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    response = answer_id
    if is_with_rationale:
        if use_alpaca:
            data_with_rationale = {
                "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. Let’s solve this step-by-step. You should first give the reason in {lang} for your choice. Then you should give the right answer index of the question.",
                "input":f"###Question: {question}\nWhich of the following is the best treatment for this patient?\n\n###Options:\n{options}",
                "output":f"###Rationale: {rationale}\n\n###Answer: OPTION {answer_id} IS CORRECT."
            }
            prompt = alpaca_prompt_template.format(**data_with_rationale)
            response = data_with_rationale['output']
        else:
            prompt = (
                f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}.\n\n"
                f"###Question:\n{question}\n\n"
                f"###Options:\n{options}\n"
            )
            if use_cot:
                prompt += "Which option is the best treatment for this patient? Think step by step."
                response = f'{answer_id}\n\n{rationale}\nOverall, the answer is: {answer_id}'
            else:
                prompt += "Answer with the best option index directly. Do not output any other words."
    else:
        if use_alpaca:
            data_without_rationale = {
                "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
                "input":f"###Question: {question}\nWhich of the following is the best treatment for this patient?\n\n###Options:\n{options}",
                "output":f"###Answer: OPTION {answer_id} IS CORRECT."
            }    
            prompt = alpaca_prompt_template.format(**data_without_rationale)
            response = data_without_rationale['output']
        else:
            prompt = f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}.\n\n"
            if rag:
                prompt += f"Here are some possibly related references:\n###References\n```\n{references}\n```\n"
                prompt += 'Answer the following question based on the references as well as your own knowledge.\n\n'
            prompt += (
                f"###Question:\n{question}\n\n"
                f"###Options:\n{options}\n"
            )
            if use_cot:
                prompt += "Which option is the best treatment for this patient? Think step by step."
            else:
                prompt += "Answer with the best option index directly. Do not output any other words."

    return prompt, answer_id, rationale, response


def eval_accuracy(prediction: str, answer: str):
    return set(prediction.split(',')) == set(answer.split(','))