# Multi-Agent Question-Answer Interaction code developed by Markus J. Buehler using the Llama-Index chat model

import os
import re
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import Markdown, display

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig,BitsAndBytesConfig,get_linear_schedule_with_warmup
import transformers
from datasets import load_dataset, IterableDataset
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.response.notebook_utils import display_response

import markdown2
import pdfkit
import unidecode
import peft

pd.set_option('display.max_colwidth', None)  

def make_dir_if_needed (dir_path):
    if not os.path.exists(dir_path):
        # Create directory
        os.makedirs(dir_path)
        return "Directory created."
    else:
        return  "Directory already exists."

def remove_markdown_symbols(text):
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    # Remove headers
    text = re.sub(r'#+\s', '', text)
    # Remove bold and italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # Remove strikethrough
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove extra newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove list markers
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    return text.strip()

def completion_to_prompt(completion):
    return "<|start_header_id|>system<|end_header_id|>\n<eot_id>\n<|start_header_id|>user<|end_header_id|>\n" + \
           f"{completion}<eot_id>\n<|start_header_id|>assistant<|end_header_id|>\n"

def messages_to_prompt(messages):
    prompt = "<|start_header_id|>system<|end_header_id|>\n<eot_id>\n"  
    for message in messages:
        if message.role == "system":
            prompt += f"system message<eot_id>\n"
        elif message.role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{message.content}<eot_id>\n"
        elif message.role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{message.content}<eot_id>\n"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def get_chat_engine_from_index_LlamaIndex(llm,index, chat_token_limit=2500,verbose=False, chat_mode="context",
                               system_prompt='You are a chatbot, able to have normal interactions, as well as talk about context provided.'):
    memory = ChatMemoryBuffer.from_defaults(token_limit=chat_token_limit)
    chat_engine = index.as_chat_engine(llm=llm,
        chat_mode=chat_mode,
        memory=memory,
        system_prompt=system_prompt,verbose=verbose,
    )
    return chat_engine

def get_answer_LlamaIndex (llm,
                           q, system_prompt="You are an expert in materials science.", chat_engine=None,
                        max_new_tokens=1024, #temperature=0.7, 
                           messages_to_prompt=None,chat_token_limit=2500,chat_mode="context",
                completion_to_prompt=None,index=None, verbose=False):
    if chat_engine==None:
        '''
        llm = HuggingFaceLLM(model=model, tokenizer=tokenizer,
                     context_window=8192,
                max_new_tokens=max_new_tokens,
               # model_kwargs={"quantization_config": quantization_config},
                generate_kwargs={"temperature": temperature, "top_k": 256, "top_p": 0.9,"do_sample":"True", "eos_token_id":eos_token_id},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                    device_map="cuda:0",system_prompt=system_prompt)
    
        '''
        if index != None:
            chat_engine=get_chat_engine_from_index_LlamaIndex(llm,index, chat_token_limit=chat_token_limit,verbose=verbose,chat_mode=chat_mode,
                                   system_prompt=f'You are a chatbot, able to have normal interactions, as well as talk about data provided. {system_prompt}')
        else:
            chat_engine = SimpleChatEngine.from_defaults(llm=llm, system_prompt=system_prompt)  
    response = chat_engine.stream_chat(q)
    for token in response.response_gen:
        print(token, end="")
    return response.response, chat_engine

class ConversationAgent_LlamaIndex:
    def __init__(self, llm,
                 #model, tokenizer, 
                 name: str, instructions: str,# context_turns: int = 2,
               # temperature=0.3,  max_new_tokens=1024, 
              #   messages_to_prompt=None, completion_to_prompt=None,
                 index=None,chat_token_limit=2500,verbose=False,chat_mode="context",
                ):
        self._name = name
        self._instructions = instructions
        self._source_nodes =[]
        '''
        self.temperature=temperature
        llm = HuggingFaceLLM(model=model, tokenizer=tokenizer,
                     context_window=8192,
                max_new_tokens=max_new_tokens,
               # model_kwargs={"quantization_config": quantization_config},
                generate_kwargs={"temperature": self.temperature, "top_k": 256, "top_p": 0.9,"do_sample":"True", "eos_token_id":eos_token_id},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                device_map="auto",)
        '''            
        if index != None:
            print (f"Set up chat engine, with index, verbose={verbose}, chat_mode={chat_mode}.")
            self.chat_engine=get_chat_engine_from_index_LlamaIndex(llm,index, chat_token_limit=chat_token_limit,verbose=verbose,chat_mode=chat_mode,
                       system_prompt=f'You are a chatbot, able to have normal interactions, as well as talk about data provided.\n\n{self._instructions}')
        else:
            self.chat_engine = SimpleChatEngine.from_defaults(llm=llm, system_prompt=self._instructions)
    @property
    def name(self) -> str:
        return self._name
    def get_conv(self, ) -> str:
        return self.chat_engine.chat_history
    def get_source_nodes(self, ) -> str:
        return self._source_nodes
    def reset_chat(self, ):
        self.chat_engine.reset() 
    def reply(self, question) -> str:
        response = self.chat_engine.stream_chat(question   )
        for token in response.response_gen:
            print(token, end="")
        self._source_nodes.append (response.source_nodes)
        return response.response, response

def conversation_simulator_LlamaIndex(
     llm_answer, llm_question,
    question_gpt_name='Engineer',answer_gpt_name='Biologist', answer_instructions='You answer correctly.',
    question_asker_instructions='You always respond with a tough questions. ',
    q='What is bioinspiration?',
    total_turns: int = 5,data_dir='./',
    marker_ch='>>> ',start_with_q=False,only_last=True, 
    marker_ch_outer='### ',sample_question='',
    answer_index=None,question_index=None, verbose=False,chat_mode="context",chat_token_limit=2500,
    include_N_turns_in_question_development=9999,single_shot_question=True,
    )-> list[dict[str,str]]:
    answer_agent = ConversationAgent_LlamaIndex(llm_answer,  
                                                name=answer_gpt_name, instructions=answer_instructions, 
                                                index=answer_index,verbose=verbose,chat_mode=chat_mode,chat_token_limit=chat_token_limit,
                                               )
    conversation_turns = []
    q_new = q #None
    conversation_turns.append(dict(name=question_gpt_name, text=q_new))
    print (f"### {question_gpt_name}: {q}\n")
    for _ in range(total_turns):
        print (f"### {answer_gpt_name}: ", end="")
        last_reply, response = answer_agent.reply(q_new)
        
        conversation_turns.append(dict(name=answer_gpt_name, text=last_reply))
        if only_last:
            txt= f'Consider this question and response.\n\n{marker_ch_outer}Question: {q}\n\n{marker_ch_outer} Response: {last_reply}'
        else:
            NN=include_N_turns_in_question_development
            NN = NN + 1 if NN % 2 else NN  
            conv=get_entire_conversation_LlamaIndex(q, conversation_turns[-NN:],marker_ch=marker_ch,start_with_q=start_with_q, question_gpt_name=question_gpt_name)
            txt=f'{marker_ch_outer}Read this conversation between you and {answer_gpt_name}:\n\n```{conv}```\n\n"'
        if single_shot_question: 
            q=f"""{txt}\n\n{marker_ch_outer} Briefly respond with a new follow-up question(s) that critically challenges the earlier responses, considering any gaps in logic, alternative methods, or unaddressed points. 
    
DO NOT answer or repeat questions or comment on it."""
            print (f"\n\n### {question_gpt_name}: ", end="")
            q_new, q_chat=get_answer_LlamaIndex (llm_question,
                                            q=q, 
                                            index=question_index,verbose=verbose,chat_mode=chat_mode,chat_token_limit=chat_token_limit,
                 system_prompt=question_asker_instructions)
        q_new=q_new.replace('"', '')
        print (f"\n")
        conversation_turns.append(dict(name=question_gpt_name, text=q_new))
    return conversation_turns, answer_agent.get_conv(), response, answer_agent

def read_and_summarize_LlamaIndex( llm,
                                  txt='This is a conversation.', q='',
                                 ):
    q=f"""Carefully read this conversation: 

>>>{txt}<<<

Develop a final answer to the original question using the conversation as additional context.

""" 

    summary, chat_engine=get_answer_LlamaIndex (llm, q=q,  
                                 system_prompt="You analyze text and provide an accurate account of the content from all sides discussed.",
                                             max_new_tokens=5000)
    return summary
    
def answer_question_LlamaIndex ( #model, tokenizer,         
                      llm_answer,
                    llm_question,   llm_summarize, 
    q='I have identified this amino acid sequence: AAAAAIIAAAA. How can I use it? ',
                    bot_name_1="Biologist",
                    bot_instructions_1 = f"""You are a biologist. You are taking part in a discussion, from a life science perspective.
Keep your answers brief, but accurate, and creative.
""",
                    bot_name_2="Engineer",
                    bot_instructions_2 = """You are a critical engineer. You are taking part in a discussion, from the perspective of engineering.
Keep your answers brief, and always challenge statements in a provokative way. As a creative individual, you inject ideas from other fields. """,
                     include_N_turns_in_question_development=99999,
                    total_turns=4,
                    delete_last_question=True, 
                    save_PDF=True,sample_question='',
                    PDF_name=None,  save_dir='./', 
                     txt_file_path=None, marker_ch='>>> ',marker_ch_outer='### ',
                     start_with_q=False,only_last=True,single_shot_question=True,
                      messages_to_prompt=None,question_index=None, answer_index=None,chat_mode="context",chat_token_limit=2500,
                completion_to_prompt=None,te_on_question=False,verbose=False,
                    ):
    conversation_turns, answer_agent_conv, response, answer_agent = conversation_simulator_LlamaIndex( llm_answer, llm_question,#  model, tokenizer, 
                                                question_gpt_name=bot_name_2,answer_gpt_name=bot_name_1,
                                             #   question_temperature=question_temperature,conv_temperature=conv_temperature,
                                                question_asker_instructions=bot_instructions_2,
                                                q=q, question_index=question_index, answer_index=answer_index,
  include_N_turns_in_question_development=include_N_turns_in_question_development, 
    single_shot_question=single_shot_question,                                                                
                                                total_turns=total_turns, data_dir=save_dir,marker_ch=marker_ch,marker_ch_outer=marker_ch_outer,
                                                           start_with_q=start_with_q,only_last=only_last, sample_question=sample_question,
                                                      verbose=verbose,chat_mode=chat_mode,chat_token_limit=chat_token_limit,
                               #  messages_to_prompt=messages_to_prompt, completion_to_prompt=completion_to_prompt,
                                                          )
    if delete_last_question:
        conversation_turns.pop()
    txt=''
    txt+=f"The question discussed is: **{q.strip()}**\n\n"
    print ("-----------------------------------------")
    for turn in conversation_turns:
        txt +=f"**{turn['name'].strip ()}**: {turn['text']}\n\n"
    summary = read_and_summarize_LlamaIndex(llm_summarize,#model, tokenizer ,
                                                                txt, q=q,  )
    integrated = f"""#### Question and conversation:
    
{txt} 

#### Summary:

{summary}

"""
    if save_PDF:
        # Convert Markdown to HTML
        html_text = markdown2.markdown(integrated)
        
        # Convert HTML to PDF and save it
        max_len_fname=64
        if PDF_name==None:
            PDF_name=f'{save_dir}{q[:max_len_fname].strip()}.pdf'
        
        pdfkit.from_string(html_text, PDF_name)
    
    max_len_fname=64
    if txt_file_path==None:
        txt_file_path = f'{save_dir}{q[:max_len_fname].strip()}.txt'
    save_raw_txt=remove_markdown_symbols(integrated)
    
    with open(txt_file_path, 'w') as file:
        file.write(save_raw_txt)

    return conversation_turns, txt, summary, integrated, save_raw_txt, answer_agent_conv, response, answer_agent

def get_entire_conversation_LlamaIndex (q, conversation_turns, marker_ch='### ', start_with_q=False, question_gpt_name='Question: '):
    txt=''
    if start_with_q:
        txt+=f"{marker_ch}The question discussed is: {q.strip()}\n\n"
    else:
        txt=''
    for turn in conversation_turns:
        txt +=f"{marker_ch}{turn['name'].strip ()}: {turn['text']}\n\n"
    return txt.strip()