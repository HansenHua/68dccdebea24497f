# https://platform.openai.com/docs/assistants/overview
import os
import openai
from openai import OpenAI
import argparse
import numpy as np
import importlib
import torch

class Config:
    def __init__(self, args):
        self.dataset = args.dataset
        self.api_key = API_KEY
        self.num = args.num
        self.model = args.model
        self.seed = args.seed
        self.temperature = args.temperature
        self.max_completion_tokens = args.max_completion_tokens
        self.top_k = args.top_k
        self.contribution_threshold = args.contribution_threshold
        self.rounds = args.max_rounds
        self.client_expert = client_expert
        self.n = args.n
        self.process_num = args.process_num
        self.alpha = args.alpha
        self.beta = args.beta
        self.distance = 'new'
        # training
        self.lr = 1e-3
        self.train_rounds = args.train_rounds
        self.alpha_init = 10.0
        # prompt
        self.prompt_mode = args.prompt_mode

def gen_response(config, message):
    if True:
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

def gen_api_key():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'api_key.txt')
    with open(filename, 'r') as f:
        return f.read().strip()

def test(questions, answers):
    model_method_module = importlib.import_module(f'method.{method}')
    print("start testing")
    accurcy = model_method_module.test(config, questions, answers)
    print("finish testing")
    print(f"Correct: {accurcy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WinoGrande', help='the name of dataset')
    parser.add_argument('--method', type=str, default='Dynamic_role', help='the name of method')
    parser.add_argument('--model', type=str, default='llama-7b', help='the name of model (gpt-4o/gpt-4o-mini/gpt-3.5-turbo/gpt-4-turbo/o1-mini)')
    parser.add_argument('--num', type=int, default=int(2), help='the number of agents')
    parser.add_argument('--max_rounds', type=int, default=int(2), help='maximum debating rounds')
    parser.add_argument('--seed', type=int, default=int(0), help='random seed')
    parser.add_argument('--n', type=int, default=int(1), help='chat completion choices to generate')
    parser.add_argument('--temperature', type=float, default=float(0.8), help='temperature , [0, 2]')
    parser.add_argument('--alpha', type=float, default=float(1e-7), help='alpha')
    parser.add_argument('--beta', type=float, default=float(1), help='beta')
    parser.add_argument('--max_completion_tokens', type=int, default=int(4096), help='max_completion_tokens')
    parser.add_argument('--top_k', type=int, default=int(20), help='top k combinations to pick')
    parser.add_argument('--contribution_threshold', type=float, default=float(0.5), help='contribution threshold')
    parser.add_argument('--process_num', type=int, default=int(1), help='number of process')
    parser.add_argument('--train_rounds', type=int, default=int(1), help='maximum training rounds')
    parser.add_argument('--prompt_mode', type=str, default='long',help='long prompt constraint')
    args = parser.parse_args()
    API_KEY = gen_api_key()

    proxy_url = 'http://127.0.0.1'
    proxy_port = '21882'

    os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
    os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

    num = args.num
    n = args.n
    model = args.model
    seed = args.seed
    temperature = args.temperature
    max_completion_tokens = args.max_completion_tokens
    top_k = args.top_k
    contribution_threshold = args.contribution_threshold
    dataset_name = args.dataset
    rounds = args.max_rounds
    method = args.method

    verifier = OpenAI(api_key=API_KEY)
    role_prompter = OpenAI(api_key=API_KEY)
    
    # load dataset
    dataset = np.load(os.path.join(os.path.dirname(__file__),'dataset', dataset_name, 'question_answer.npy'), allow_pickle=True).item()
    client_expert = dataset['expert']
    client_expert_description = {'expert':[], 'description':[]}
    for i in range(len(client_expert)):
        client_expert_description['expert'].append(client_expert[i])
        messages=[
            {
                "role": "user",
                "content": f"What can you do if you are a {client_expert[i]}."
            }
        ]
        # response = gen_response(messages)
        # client_expert_description['description'].append(response)
    print("role_prompter is ready")
    num = max(num, len(client_expert))
    config = Config(args)

    test(dataset["questions"], dataset["answers"])