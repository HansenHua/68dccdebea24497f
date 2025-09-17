# https://platform.openai.com/docs/assistants/overview
import os
import openai
from openai import OpenAI
import argparse
import numpy as np
import importlib
import multiprocessing as mp
import torch.nn as nn
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
    model_method = ROMA(config)
    correct = 0
    for question, answer in zip(questions, answers):
        if(verify_answer(config, model_method.answer(question), answer)):
            correct += 1
    correct_list[id] = correct
    return correct

def test(config, questions, answers):
    p_list = []
    correct_list = [0 for i in range(config.process_num)]
    question_list = split_list(questions, config.process_num)
    answer_list = split_list(answers, config.process_num)
    for id in range(config.process_num):
        p_list.append(mp.Process(target=test_, args=(config, question_list[id], answer_list[id], id, correct_list,)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    return sum(correct_list)/len(questions)

def check_agreement(response):
    response_list = response.split(' ')
    if 'agree' in response_list:
        return True
    else:
        return False

class Agent:
    def __init__(self, config, role):
        self.config = config
        self.base_role = role
        self.role = role
        from transformers import LlamaTokenizer, LlamaForCausalLM

        model_name_or_path = os.path.join(os.pardir, 'model', config.model)

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, device_map="auto")
        questions = np.load(os.path.join(os.path.dirname(__file__)[:-1],'dataset', self.config.dataset, 'question_answer.npy'), allow_pickle=True).item()["questions"][self.base_role]
        answers = np.load(os.path.join(os.path.dirname(__file__)[:-1],'dataset', self.config.dataset, 'question_answer.npy'), allow_pickle=True).item()["answers"][self.base_role]
        self.question_list = questions[:int(0.7*len(questions))]
        self.answer_list = answers[:int(0.7*len(answers))]
    
    def response(self, message):
        messages=[
            {
                "role": "system",
                "content": f"Suppose you are an expert in {self.role}, your job is to answer questions from your expertise."
            },
            {
                "role": "user",
                "content": message
            }
        ]
        r = gen_response(self.config, self.tokenizer, self.model, messages)
        return r

class role_generator(nn.Module):
    def __init__(self, config):
        self.config = config
        self.base_role = config.client_expert
        self.role = self.base_role
    
    def forward(self, agent_list, question):
        pass

    def train(self):
        pass

class ROMA:
    def __init__(self, config):
        self.config = config
        self.agent_list = [Agent(self.config, role) for role in self.config.client_expert]
        self.base_role = self.config.client_expert
        self.role_generator = role_generator(self.config)
        from transformers import LlamaTokenizer, LlamaForCausalLM

        model_name_or_path = os.path.join(os.pardir, 'model', config.model)

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, device_map="auto")
    
    def set_role(self, question):
        role_list = role_generator(agent, question)
        for id, agent in enumerate(self.agent_list):
            agent.role = role_list[id]

    def answer_(self, question, id, answer_list):
        messages=f"Please {question}."
        for id, agent in enumerate(self.agent_list):
            response=agent.response(messages)
            response.replace(question,"")
            answer_list[id]=response

    def answer_debate(self, question, id, answer_list):
        messages=f"Your answer is not good enough. Here are some answers from experts in other fields: {answer_list} Please {question} again from your professional perspective."
        for id, agent in enumerate(self.agent_list):
            response=agent.response(messages)
            response.replace(question,"")
            answer_list[id]=response
    
    def answer(self, question):
        self.set_role(question)
        response_client_list = []
        for _ in range(self.config.num):
            response_client_list.append("")

        for id in range(self.config.num):
            self.answer_(question, id, response_client_list)
        
        # debate
        for _ in range(self.config.rounds):
            # server evaluate agreement
            messages=[
                {
                    "role": "system",
                    "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question} and rank them."
                },
                {
                    "role": "user",
                    "content": f"Do you think the following answers have reached an agreeent? please answer with agree or disagree. {response_client_list}"
                }
            ]
            server_response=gen_response(self.config, self.tokenizer, self.model, messages)
            server_response.replace(question,"")
            if(check_agreement(server_response)):
                break

            for id in range(self.config.num):
                self.answer_debate(question, id, response_client_list)
        
        # final answer
        messages=[
            {
                "role": "system",
                "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question} and rank them."
            },
            {
                "role": "user",
                "content": f"Please go through the following responses {response_client_list} Then, summarize your final answer to the question {question}. "
            }
        ]
        final_answer=gen_response(self.config, self.tokenizer, self.model, messages)
        final_answer.replace(question,"")
        return final_answer