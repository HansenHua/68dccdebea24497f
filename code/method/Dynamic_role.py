# https://platform.openai.com/docs/assistants/overview
import os
import openai
from openai import OpenAI
import argparse
import numpy as np
import importlib
import multiprocessing as mp
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM
import copy
import random
import numpy as np
import subprocess
import torch.nn.functional as F

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
    
class ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4096, 512)
    def forward(self, a):
        return self.fc(a)

class LightweightF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)
    def forward(self, context):
        return self.fc(context)

def init_role(role_pool, question, F_model, N, tokenizer, model):
    role_list = []
    question_vec = sentence2vec(question, tokenizer, model)
    M = role_pool.size(0)
    best_loss = float("inf")
    best_subset = None

    from itertools import combinations
    for subset in combinations(range(M), N):
        subset_vecs = role_pool[list(subset)]
        proj = F_model(subset_vecs)
        omega = torch.linalg.lstsq(proj, question_vec).solution
        recon = proj.T @ omega
        loss = torch.norm(recon - question_vec).item()
        if loss < best_loss:
            best_loss = loss
            best_subset = subset
    for i in range(len(subset)):
        role_list.append(role_pool[best_subset[i]])
        
    return role_list

def train_init(role_pool, questions, answers, t5_tokenizer, t5_model, tokenizer, model):
    f_model = ProjectionModel()
    optimzier_f = torch.optim.Adam(f_model.parameters())
    F_model = LightweightF()
    optimzier_F = torch.optim.Adam(F_model.parameters())
    for question, answer in zip(questions, answers):
        v_p = sentence2vec(question, t5_tokenizer, t5_model)
        inputs = tokenizer(answer+question, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        a_c = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
        f = f_model(a_c)
        loss = nn.MSELoss()(v_p, f)
        optimzier_f.zero_grad()
        loss.backward()
        optimzier_f.step()
    for question, answer in zip(questions, answers):
        for role in role_pool:
            v_i = sentence2vec(role+question, t5_tokenizer, t5_model)
            F_i = F_model(v_i)
            inputs = tokenizer(role+question, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            a_i = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
            f_i = f_model(a_i).detach()
            loss = nn.MSELoss()(F_i, f_i)
            optimzier_F.zero_grad()
            loss.backward()
            optimzier_F.step()
    return F_model

def split_list(lst, k):
    avg_len = len(lst) // k
    remainder = len(lst) % k

    result = []
    start = 0

    for i in range(k):
        end = start + avg_len + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result

def sentence2vec(sentence, tokenizer, model):
    device = "cpu"
    inputs = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.encoder(input_ids=inputs)
        outputs = outputs.last_hidden_state
    sentence_vector = torch.sum(outputs, dim=0)
    return sentence_vector

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

def gen_response(config, tokenizer, model, message, id=1):
    device = "cpu"
    m = ""
    for i in range(len(message)):
        if id == 1:
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

def test_(config, F_model, agent_list, questions, answers, id, correct_list):
    print('process ',id)
    model_method = Dynamic_role(config)
    # result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(result.stdout)
    correct = 0
    for question, answer in zip(questions, answers):
        role_list = init_role(config.client_expert, question, F_model, len(model_method.agent_list), model_method.t5_tokenizer, model_method.t5_model)
        model_method.agent_list = copy.deepcopy(agent_list)
        for i, agent in enumerate(model_method.agent_list):
            agent.role = role_list[i]
        if(verify_answer(config, model_method.answer(question), answer)):
            correct += 1
    correct_list[id] = correct
    return correct

def test(config, questions, answers):
    mp.set_start_method('spawn')
    model_method = Dynamic_role(config)
    F_model = train_init(config.client_expert, questions, answers, model_method.t5_tokenizer, model_method.t5_model, model_method.tokenizer, model_method.model)
    model_method.train_role(questions)
    for i in range(len(model_method.agent_list)):
        del model_method.agent_list[i].role_generator
        torch.cuda.empty_cache()
    del model_method
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    p_list = []
    correct_list = [0 for i in range(config.process_num)]
    question_list = split_list(questions, config.process_num)
    answer_list = split_list(answers, config.process_num)
    for id in range(config.process_num):
        p_list.append(mp.Process(target=test_, args=(config, F_model, model_method.agent_list, question_list[id], answer_list[id], id, correct_list,)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    return sum(correct_list)/len(questions)

def check_agreement(response):
    response_list = response.split(' ')
    if 'agree' in response_list:
        return True
    else:
        return False

def cal_alpha_loss(alpha, r, r_b):
    alpha_loss = - nn.MSELoss()(r, r_b).detach() * alpha
    return alpha_loss
    
def cal_loss(config, agent_list, id, tokenizer, model, questions, t5_tokenizer, t5_model):
    for _ in range(config.rounds):
        response_list = ["" for _ in range(len(agent_list))]
        for i in range(len(agent_list)):
            responses = ""
            for j in range(len(response_list)):
                if j == i:
                    continue
                responses += response_list[j]
            role, logits = agent_list[i].role_generator.gen_role(questions+responses)
            message = f"Suppose you are an expert in {role} " + questions
            response_list[i] = gen_response(config,tokenizer,model,message,0)
            response_list[i].replace(message,"")
            inputs = tokenizer(role+questions, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            a_r = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
            inputs = tokenizer(response_list[i]+questions, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            a_p = torch.sum(hidden_states[-1].squeeze(dim=0),dim=0)
            l_1 = torch.sqrt(torch.mean((a_r- a_p) ** 2)) * 1e-3
            l_2 = torch.sqrt(torch.mean((torch.sum(sentence2vec(role, t5_tokenizer, t5_model),dim=0) - torch.sum(sentence2vec(agent_list[i].base_role, t5_tokenizer, t5_model),dim=0)) ** 2))
            l = (l_1.detach() + l_2.detach() * agent_list[i].alpha()) * logits
            agent_list[i].role_generator.role_optimizer.zero_grad()
            l.backward()
            agent_list[i].role_generator.role_optimizer.step()
            alpha_loss = cal_alpha_loss(agent_list[i].alpha(), torch.sum(sentence2vec(role, t5_tokenizer, t5_model),dim=0).detach(),sentence2vec(agent_list[i].base_role, t5_tokenizer, t5_model).detach())
            agent_list[i].alpha_optimizer.zero_grad()
            alpha_loss.backward()
            agent_list[i].alpha_optimizer.step()


class Agent:
    def __init__(self, config, role, t5_t, t5_m):
        self.config = config
        self.base_role = role
        self.role = role
        self.t5_tokenizer = t5_t
        self.t5_model = t5_m
        self.role_generator = role_generator(self.config, role, self.t5_tokenizer, self.t5_model)
        self.alpha = Scalar(self.config.alpha_init)
        self.alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=self.config.lr)
    
    def response(self, tokenizer, model, message):
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
        r = gen_response(self.config, tokenizer, model, messages)
        return r
    
    def set_role(self, question):
        self.role, _ = self.role_generator.gen_role(question)

class role_generator(nn.Module):
    def __init__(self, config, role, t5_t, t5_m):
        super(role_generator, self).__init__()
        self.config = config
        self.base_role = role
        self.t5_tokenizer = t5_t
        self.t5_model = t5_m
        self.seq_length = 10
        self.hidden_dim = 512
        self.base_role_vector = sentence2vec(self.base_role, self.t5_tokenizer, self.t5_model)
        self.role_network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        device = "cpu"
        self.role_network.to(device)
        self.role_optimizer = torch.optim.Adam(self.role_network.parameters(), lr=self.config.lr)
    
    def forward(self, question):
        print(question)
        q_vec = sentence2vec(question, self.t5_tokenizer, self.t5_model)
        inputs = torch.concat([self.base_role_vector, q_vec])
        print('inputs',inputs.shape)
        hidden_states = self.role_network(inputs)
        print('hidden',hidden_states.shape)
        return hidden_states
    
    def gen_role(self, question):
        hidden_states = self.forward(question)
        hidden_states = hidden_states.unsqueeze(0)
        generated = self.t5_model.generate(inputs_embeds=hidden_states, max_length=100)
        scores = generated[0].scores
        probs = [F.softmax(score, dim=-1) for score in scores]
        probs = torch.stack(probs, dim=0)
        role = self.t5_tokenizer.decode(generated[0], skip_special_tokens=True)
        return role, probs

class Dynamic_role:
    def __init__(self, config):
        self.config = config
        model_name_or_path = os.path.join(os.pardir, 'model', config.model)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, device_map="cpu")
        model_path = os.path.join(os.pardir, "t5-small")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path).to('cpu')
               
        self.agent_list = [Agent(self.config, "", self.t5_tokenizer, self.t5_model) for i in range(config.num)]
        self.base_role = self.config.client_expert
    
    def train_role(self, questions):
        print("start training")
        for _ in range(self.config.train_rounds):
            i = random.randint(0,len(questions)-1)
            question = questions[i]
            role_list = init_role(self.config.client_expert,question,len(self.agent_list), self.t5_tokenizer, self.t5_model)
            for i, agent in enumerate(self.agent_list):
                agent.role = role_list[i]
            for id, agent in enumerate(self.agent_list):
                cal_loss(self.config, self.agent_list, id, self.tokenizer, self.model, question, self.t5_tokenizer, self.t5_model) * agent.alpha()

    
    def set_role(self, question, response_list):
        for id, agent in enumerate(self.agent_list):
            responses = ""
            for i in range(len(response_list)):
                if i == id:
                    continue
                responses += response_list[i]
            agent.set_role(question+responses)

    def answer_(self, question, id, answer_list):
        if self.config.prompt_mode == 'long':
            messages=f"Please {question}."
        else:
            messages=f"Please answer as short as possible {question}."
        for id, agent in enumerate(self.agent_list):
            response=agent.response(self.tokenizer, self.model, messages)
            response.replace(messages,"")
            answer_list[id]=response

    def answer_debate(self, question, id, answer_list):
        answers = ""
        for i in range(self.config.num):
            if i == id:
                continue
            else:
                answers += answer_list[i]
        if self.config.prompt_mode == 'long':
            messages=f"Your answer is not good enough. Here are some answers from experts in other fields: {answers} Please {question} again from your professional perspective."
        else:
            messages=f"Your answer is not good enough. Here are some answers from experts in other fields: {answers} Please {question} again from your professional perspective as short as possible."
        response=self.agent_list[id].response(self.tokenizer, self.model, messages)
        response.replace(messages,"")
        
        return response
    
    def answer(self, question):
        response_client_list = []
        for _ in range(self.config.num):
            response_client_list.append("")

        for id in range(self.config.num):
            self.answer_(question, id, response_client_list)
        
        # debate
        for _ in range(self.config.rounds):
            # server evaluate agreement
            self.set_role(question, response_client_list)
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
            
            temp = copy.deepcopy(response_client_list)
            for id in range(self.config.num):
                response = self.answer_debate(question, id, temp)
                response_client_list[id] = response
        
        # final answer
        if self.config.prompt_mode == 'long':
            messages=[
                {
                    "role": "system",
                    "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question}."
                },
                {
                    "role": "user",
                    "content": f"Please go through the following responses {response_client_list} Then, summarize your final answer to the question {question}. "
                }
            ]
        else:
            messages=[
                {
                    "role": "system",
                    "content": f"You are a leader and summarizer. Your job is to assess how well your group answers {question}."
                },
                {
                    "role": "user",
                    "content": f"Please go through the following responses {response_client_list} Then, try you best to summarize your final answer with less words to the question {question}. "
                }
            ]
        final_answer=gen_response(self.config, self.tokenizer, self.model, messages)
        final_answer.replace(question,"")
        print('#'*50)
        print(final_answer)
        return final_answer