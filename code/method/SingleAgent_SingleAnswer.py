import os
import openai
from openai import OpenAI
import numpy as np
import multiprocessing as mp
import torch

def split_list(lst, k):
    # Calculate the size of each part
    avg_len = len(lst) // k
    remainder = len(lst) % k  # Number of parts that will have one extra element

    result = []
    start = 0

    for i in range(k):
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result

def verify_answer(config, final_answer, answer):
    if config == 'MMLU':
        if '('+answer+')' in final_answer:
            return True
        else:
            return False
    elif config == 1:
        final_answer = final_answer.replace("1 or 2", "")
        if ('('+answer+')' in final_answer or answer+')' in final_answer or answer in final_answer) and ('('+str(3-int(answer))+')' not in final_answer and str(3-int(answer))+')' not in final_answer and str(3-int(answer)) not in final_answer):
            return True
        else:
            return False

def gen_response(config, tokenizer, model, message):
    if config.model in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4-turbo', 'o1-mini']:
        import http.client
        import json

        conn = http.client.HTTPSConnection("api.chatanywhere.tech")
        payload = json.dumps({
        "model": config.model,
        "messages": message,
        "temperature":config.temperature,
        "max_tokens":config.max_completion_tokens,
        })
        headers = {
        'Authorization': config.api_key,
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = data.decode("utf-8")
        data = json.loads(data)
        if "choices" not in data:
            print("error generating answer")
            return data
        return data["choices"][0]["message"]["content"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = ""
        for i in range(len(message)):
            m += message[i]["content"]
            m += " "
        message = m

        inputs = tokenizer(message, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.max_completion_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                num_return_sequences=1,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
def test_(config, questions, answers, id, correct_list):
    model_method = SingleAgent_SingleAnswer(config)
    correct = 0
    for question, answer in zip(questions, answers):
        if(verify_answer(config, model_method.answer(question), answer)):
            correct += 1
    correct_list[id] = correct
    return correct

def test(config, questions, answers):
    p_list = []
    config.num = 1
    correct_list = [0 for i in range(config.process_num)]
    question_list = split_list(questions, config.process_num)
    answer_list = split_list(answers, config.process_num)
    for id in range(config.process_num):
        p_list.append(mp.Process(target=test_, args=(config, question_list[id], answer_list[id], id, correct_list,)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    return sum(correct_list)/len(questions)

class SingleAgent_SingleAnswer:
    def __init__(self, config):
        self.config = config
        from transformers import LlamaTokenizer, LlamaForCausalLM

        model_name_or_path = os.path.join(os.pardir, 'model', config.model)

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, device_map="auto")
    
    def answer(self, question):
        messages=[
            {
                "role": "user",
                "content": f"Please {question} from your professional perspective."
            }
        ]
        final_answer = gen_response(self.config, self.tokenizer, self.model, messages)
        final_answer.replace(question,"")
        print(final_answer)

        return final_answer