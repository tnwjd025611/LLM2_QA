"""
Adapted from https://github.com/tatsu-lab/stanford_alpaca
"""

import concurrent.futures
import json
import os
import random
import re
import string
import time
from functools import partial
from multiprocessing import Pool

import fire 
import numpy as np
import tqdm
from rouge_score import rouge_scorer

# 상위 폴더로부터 relative import 해결
import sys
from pathlib import Path
utils_path = str(Path(__file__).resolve().parent.parent)
if utils_path not in sys.path:
    sys.path.append(utils_path)

import utils_ollama



def parallel_execution(f, n, inputs):
    # jittering 추가
    def f_with_jitter(input_dict):
        time.sleep(random.uniform(0, 0.1))
        return f(input_dict)
    
    # inputs의 lenth가 n과 동일한지 확인
    if len(inputs) != n:
        raise ValueError('The length of the input list must be equal to the integer n')
    
    # 병렬을 위한 function class Run
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(f_with_jitter, inputs))

    return results

example = """
    Here is the description about ECG dataset.
    
    description: 
  
    
"""

prompt_response = """

"""

prompt_template = """ Here is the question:
{question}

The answer key:
{answer}

Please generate {num_aug} similar questions, along with the correct classification and rationale."""



def encode_prompt(prompt_instructions, num_aug=1, current_keep=None):
    # 여러 prompt instructions을 single string으로 encode
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt = open(f'{current_dir}/prompt_ecg.txt').read() + '\n'

    messages = ''
    messages += prompt_instructions[0]['description']
    messages += "Here is the Question and Options:"+'\n'
    messages += prompt_instructions[0]['question']+'\n'
    messages += f'The Answer key : {prompt_instructions[0]["answer"]}'+'\n'
    messages += prompt+'\n'
    messages += f'Now generate {num_aug} more questions.'

    return messages


def post_process_llama_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = str(response)
    raw_instructions = re.split('[0-9]{1,2}\.\s+Question:', raw_instructions)
    instructions = []
    num_prompt_instructions=0

    for idx, inst in enumerate(raw_instructions):
        if idx == len(raw_instructions) - 1:
                    continue
        idx += num_prompt_instructions + 1
        if not 'Answer' in inst:
            pass
        else:
            splitted_data = re.split(f'(Answer):', inst)

        # if len(splitted_data) != 3:
        #     continue
        
        answers = re.split(f'(####)', splitted_data[2])
        # if len(answers) != 3:
        #     continue
        question = splitted_data[0].strip('\n')
        if len(answers) !=1:
            answering = ''
            for ans in answers:
                answering += ans.strip('\n')
        else:
            answer = answers[0].strip('\n')
        
        instructions.append({'question': question, 'answer': answer})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def generate_data(
        output_dir='./data/output',
        seed_tasks_path='',
        instructions_per_seed_task=4,
        generated_instructions_per_seed_task=4,
        num_prompt_instructions=1,
        # temperature=1.0,
        # top_p=1.0,
        num_cpus=64,
        rouge_score_threshold=0.95,
        generation_workers=10,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, 'r')]
    seed_instruction_data = [
        {'description': t['description'], 'question': t['question'], 'answer': t['answer'], 'ai_answer': t['ai_answer']} # ecg 관련 정보 추가 생각
        for t in seed_tasks
    ]
    print(f'Loaded {len(seed_instruction_data)} human-written seed instructions')

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # LM-generated instructions를 load
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, 'regen.json')):
        machine_instruction_data = utils_ollama.jload(os.path.join(output_dir, 'regen.json'))
        print(f'Loaded {len(machine_instruction_data)} machine-generated instructions')
    
    # similarities = {}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    # 새로운 instructions 생성
    progress_bar = tqdm.tqdm(total=instructions_per_seed_task * len(seed_instruction_data))
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))
    
    # 모든 seed instructions를 tokenizer하고 machine instructions 생성
    all_instructions = [d['question'] for d in seed_instruction_data] + [
        d['question'] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    num_requests= 0

    request_idx = len(machine_instruction_data) // instructions_per_seed_task
    print(f'Starting from instruction {request_idx} with 0 tasks kept for the current seed')


    def generate_task_for_idx(task_idx):
        prompt_instructions = seed_instruction_data[task_idx]
        kept_tasks_for_current_seed = 0
        attempt = 0
        current_keep = []

        while kept_tasks_for_current_seed < instructions_per_seed_task:

            batch_inputs = []
            messages = encode_prompt([prompt_instructions], num_aug=min(generated_instructions_per_seed_task + attempt, 10), current_keep=current_keep if attempt % 2 == 1 else [])
            batch_inputs.append(messages)

            request_start = time.time()
            def f(message): # 모델에게 답변 생성
                
                result = utils_ollama.llamaGeneration(message)

                return result
            
            all_results = parallel_execution(f, batch_inputs) # f, request_batch_size, batch_inputs
            nonlocal num_requests
            num_requests += 1
            request_duration = time.time() - request_start

            process_start = time.time()
            instruction_data = []
            for results in all_results:
                for result in results:
                    new_instructions = post_process_llama_response(num_prompt_instructions, result)
                    instruction_data += new_instructions

            total = len(instruction_data)
            keep = 0
            for instruction_data_entry in instruction_data:
                # pre-tokenized instructions와 similarity 비교
                new_instructions_tokens = scorer._tokenizer.tokenize(instruction_data_entry['question'])
                with Pool(num_cpus) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instructions_tokens),
                        all_instruction_tokens,
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores if score is not None]
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10][::-1]
                }

                if max(rouge_scores) > rouge_score_threshold:
                    continue
                else:
                    keep += 1
                    keep_tasks_for_current_seed += 1
                    current_keep.append(instruction_data_entry)

                instruction_data_entry['most_similar_instructions'] = most_similar_instructions
                instruction_data_entry['avg_similarity_score'] = float(np.mean(rouge_scores))
                instruction_data_entry['seed'] = prompt_instructions
                machine_instruction_data.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry['question'])
                all_instruction_tokens.append(new_instructions_tokens)
                progress_bar.update(1)

                if kept_tasks_for_current_seed >= instructions_per_seed_task:
                    break
            process_duration = time.time() - process_start
            print(f'Instruction {task_idx} Request {num_requests} Attempt {attempt} took {request_duration:.2f}s, processing took {process_duration:.2f}s')
            print(f'Generated {total} instructions, kept {keep} instructions')
            utils_ollama.jdump(machine_instruction_data, os.path.join(output_dir, 'regen.json'))

            attempt += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=generation_workers) as executor:
            print(f'Submitting tasks for request idx {request_idx} to { len(seed_instruction_data)}')

            futures = [executor.submit(generate_task_for_idx, idx) for idx in range(request_idx, len(seed_instruction_data))]

            for future in concurrent.futures.as_completed(futures):
                future.result()

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == '__main__':
    fire.Fire(main)