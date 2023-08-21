#!/usr/bin/env python3

import sys
if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " config-file")
    exit(1)

import json

with open(sys.argv[1], "r") as configs:
    parameters = json.load(configs)

import os

stablediffusion_repository = parameters['stablediffusion_repository']
save_path = os.path.abspath(parameters['save_path'])

import gradio as gr
import random
import time
import guidance
import torch
import shutil
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import PIL
from PIL import Image

import web3
from web3 import Web3, Account
import hashlib

import git
from git import Repo

guidance.llm = guidance.llms.OpenAI(
    'gpt-3.5-turbo',
    caching=False,
    api_type=parameters['llm_api_type'],
    api_key=parameters['llm_api_key'],
    api_base=parameters['llm_api_base'],
    api_version=parameters['llm_api_version'],
    deployment_id=parameters['llm_deployment_id']
)

global is_generation, hash_generated
is_generation = False
hash_generated = False

global user_ai
user_ai = []

global generation_save_path

#LLM Generate Response
def generate_answer(question):
    global is_generation, user_ai
    prev_prompt = '''
{{#system~}}
You are helpful assistant.
{{~/system}}
'''
    for item in user_ai:
        prev_prompt += '{{#user~}}\n' + item[0] + '\n{{~/user}}\n'
        prev_prompt += '{{#assistant~}}\n' + item[1] + '\n{{~/assistant}}\n'

    program1 = guidance(prev_prompt + '''
{{#user~}}
{{query}}
{{~/user}}
{{#system~}}                   
Does the user want you to generate a picture now?
{{~/system}}
{{#assistant~}}                    
{{select 'reply' options=["Yes", "No"]}}
{{~/assistant}}        
''')
    out1 = program1(query=question)
    is_generation = out1['reply'] == 'Yes'
    print("is_generation: " + str(is_generation))
    if is_generation:
        prompt = prev_prompt + """
{{#user~}}
{{query}}
{{~/user}}
{{#system~}}
You are asked to generate a picture. Please just describe its features with words, the system will generate a picture according to the description.
{{~/system}}
{{#assistant~}}
{{gen 'answer' save_prompt="prompt" max_tokens=300}}
{{~/assistant}}
"""
        program2 = guidance(prompt)
        print(prompt)
        out2 = program2(query=question)
        answer = out2['answer']
        for c in out2['answer']:
            yield c
        yield "\n\nGenerating the picture with Stable Diffusion...\n\n"

        ticks = str(time.time())
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        global generation_save_path
        generation_save_path = os.path.join(save_path, "Generation-" + localtime + "-tick:"+ ticks)
        os.mkdir(generation_save_path)
        prompt_path = os.path.join(generation_save_path, 'stable_diffusion_prompt.txt')
        prompt = open(prompt_path, 'w')
        prompt.write(answer)
        prompt.close()
    else:
        yield "Does the user want you to generate a picture?" + "\n\n" + out1['reply'] + "\n\n"
        prompt = prev_prompt + '''
{{#user~}}
{{query}}
{{~/user}}
{{#assistant~}}
    {{gen "answer" save_prompt="prompt" max_tokens=300}}
{{~/assistant}}
'''
        print(prompt)
        program2 = guidance(prompt)
        out2 = program2(query=question)
        answer = out2['answer']
        for c in answer:
            yield c
    user_ai.append([question, answer])

global seed
seed = 512

#StableDiffusion Generate Image
def show_image(prompt, image):
    global seed
    print(seed)

    flag = False
    while (flag == False):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(stablediffusion_repository, torch_dtype=torch.float16)
            flag = True
        except Exception as e:
            print("Exception: ", e)
            time.sleep(0.5)
            continue

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(prompt, generator=generator).images[0]

    global generation_save_path
    image_path = os.path.join(generation_save_path, "image.png")
    image.save(image_path)
    return image

def generate_image_if_needed(prompt, image):
    global is_generation
    if not is_generation:
        return image
    return show_image(prompt, image)

#Enter Text
def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

#Show Response in Chatbot
def bot(history):
    question = history[-1][0]
    history[-1][1] = ""
    for text in generate_answer(question):
        history[-1][1] += text
        yield history
    print('bot return')

#Load Prompt and Image & Calculate Hash
def hash_file():
    global hash_generated
    # make a hash object
    hashvalue1 = hashlib.sha256()
    hashvalue2 = hashlib.sha256()
    hashvalue4 = hashlib.sha256()
    # open file for reading in binary mode
    prompt_path = os.path.join(generation_save_path, "stable_diffusion_prompt.txt")
    with open(prompt_path,'rb') as file1:
        chunk1 = 0
        while chunk1 != b'':
            # read only 1024 bytes at a time
            chunk1 = file1.read(1024)
            hashvalue1.update(chunk1)

    image_path = os.path.join(generation_save_path, "image.png")
    with open(image_path,'rb') as file2:
        chunk2 = 0
        while chunk2 != b'':
            chunk2 = file2.read(1024)
            hashvalue2.update(chunk2)
    
    prompt_hash = hashvalue1.hexdigest().encode('utf-8')
    image_hash = hashvalue2.hexdigest().encode('utf-8')
    hashvalue4.update(prompt_hash)
    hashvalue4.update(image_hash)

    repo = Repo(stablediffusion_repository)
    commit_hash = repo.git.rev_parse("HEAD")
    
    hashvalue4.update(commit_hash.encode('utf-8'))

    hash_prompt = "Hash of Prompt: " + hashvalue1.hexdigest()
    hash_image = "Hash of Generated Image: " + hashvalue2.hexdigest()
    hash_SD = "Hash of Stable Diffusion: " + commit_hash
    hash_final = "Final Hash of AI Generations: " + hashvalue4.hexdigest()
    show_seed = "The Seed of Stable Diffusion is: " + str(seed)

    hashes = hash_prompt+'\n'+hash_image+'\n'+hash_SD+'\n'+hash_final+'\n'+show_seed

    hash_path = os.path.join(generation_save_path, "hash.txt")
    hash_file = open(hash_path, "w")
    hash_file.write(hashes)
    hash_file.close()
    hash_generated = True

    return hashes

w3 = Web3(Web3.HTTPProvider(parameters['my_provider_link']))
global count
count = w3.eth.get_transaction_count(parameters['my_address']) + 1

#Set Seplolia Testnet API
def sepolia_api():
    global count
    print(count)
    network_id = w3.net.version 
    print(f"Connected to network with ID: {network_id}")

    private_key = parameters['my_private_key']
    account = Account.from_key(private_key) 
    print(f"Loaded account: {account.address}")

    my_address = parameters['my_address']
    receiver_address = parameters['receiver_address']
    balance = w3.eth.get_balance(my_address) 
    print(f"Account balance: {w3.from_wei(balance, 'ether')} ETH")

    hash_path = os.path.join(generation_save_path, 'hash.txt')
    hashes = open(hash_path, "r").read()

    nonce = w3.eth.get_transaction_count(my_address)
    transaction_setting = dict(
        nonce=nonce,
        maxFeePerGas=3000000000,
        maxPriorityFeePerGas=2000000000,
        gas=100000,
        to=receiver_address,
        value=100000000000000,
        data=hashes.encode('UTF-8'),
        type=2,  # (optional) the type is now implicitly set based on appropriate transaction params
        chainId=11155111,
    )

    print(transaction_setting['data'])
    print(type(transaction_setting['data']))
    signed_txn = w3.eth.account.sign_transaction(
    transaction_setting,
    private_key,
    )

    transaction_raw_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    transaction_hash = transaction_raw_hash.hex() 
    transaction_details = hashes + '\n' + "Transaction Hash: " + transaction_hash + '\n' + "Transaction Sent!"+ '\n'
    
    count += 1

    return transaction_details

global bundle_count
bundle_count = 0
def move_files():
    global bundle_count
    while True:
        bundle_count += 1
        bundle_path = os.path.join(save_path, "Generation-" + str(bundle_count))
        if not os.path.exists(bundle_path):
            break
    shutil.move(generation_save_path, bundle_path)
    return None

#Design GUI with Gradio
with gr.Blocks() as demo:
    gr.Markdown("**PromptEstate**")

    with gr.Tab("Chatbot"):
        gr.Markdown("**ChatBot**")
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=300)
        with gr.Row():
            with gr.Column():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload an image",
                ).style(container=False)

    with gr.Tab("Generate Image"):
        with gr.Column():
            gr.Markdown("**Stable Diffusion Generate Image**")
            with gr.Row():
                text_input = gr.Textbox(interactive=True, lines=10)
                image_output = gr.Image()
            with gr.Row():
                with gr.Column(scale=0.8, min_width=0):
                    gr.Markdown("**Choose your preferred seed for Stable Diffusion**")
                    seed_change_box = gr.Slider(minimum=0,maximum=2048,value=1024,step=1,interactive=True,scale=1)
                with gr.Column(scale=0.2, min_width=0):
                    seed_change_button = gr.Button(value="Confirm", scale=1)
            image_button = gr.Button("Generate Image")

            hash_button = gr.Button("Generate Hash")
            hash_output = gr.Textbox("Your hash values are shown here")
    with gr.Tab("Blockchain"):
        with gr.Column():
            gr.Markdown("**Hash on Blockchain**")
            with gr.Row():
                hash_input = gr.Textbox(interactive=False, lines=10)
                chain_output = gr.Textbox(interactive=False, lines=10)
            on_chain_button = gr.Button("Send Transaction")

        

    generate_prompt = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )

    def fill_prompt_fn(prompt):
        if is_generation:
            global generation_save_path
            prompt_path = os.path.join(generation_save_path, 'stable_diffusion_prompt.txt')
            return open(prompt_path, "r").read()
        else:
            return prompt
    fill_prompt = generate_prompt.then(fill_prompt_fn, text_input, text_input)

    def show_image_generation_done(history):
        print("show_image_generation_done " + str(is_generation))
        if not is_generation:
            return history
        history[-1][1] += "\n\nPicture generated with seed " + str(seed)+ ". Please view it in the “generate image” tab.\n\n"
        return history
    
    def show_hash():
        if hash_generated:
            global generation_save_path
            hash_path = os.path.join(generation_save_path, 'hash.txt')
            return open(hash_path, "r").read()
        else:
            return "No hash now"
        
    def change_seed(seed_input):
        global seed
        seed = int(seed_input)
        print("Seed: " + str(seed))
        return seed

    image_button.click(show_image, inputs = text_input, outputs = image_output)
    seed_change_button.click(change_seed, inputs = seed_change_box, outputs = None)
    
    generate_image = fill_prompt.then(generate_image_if_needed, inputs=[text_input, image_output], outputs = image_output)
    notify = generate_image.then(show_image_generation_done, inputs=chatbot, outputs=chatbot)
    notify.then(lambda: gr.update(value="", interactive=True), inputs=None, outputs=txt)

    hash_shown = hash_button.click(hash_file, outputs=hash_output)
    hash_shown.then(show_hash, inputs=None, outputs=hash_input)

    publish = on_chain_button.click(sepolia_api, inputs = None, outputs = chain_output)
    publish.then(move_files, inputs=None, outputs=None)


demo.queue()

if __name__ == "__main__":
    demo.launch(share=False)
