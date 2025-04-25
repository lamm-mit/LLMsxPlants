# formatting for llama3.1

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
