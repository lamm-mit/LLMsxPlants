import pandas as pd
import json
import itertools
from typing import List, Optional, Sequence

def extract_bullet_points(response_text):
    lines = response_text.split('\n')
    bullet_points = set()
    for line in lines:
        line = line.strip()
        if line.startswith(('- ', '• ', '* ')):
            bullet_points.add(line[2:].strip())
        elif line and line[0].isdigit() and line[1:3] == '. ':
            bullet_points.add(line[3:].strip())
        elif line.startswith('[') and line[1].isdigit() and line[2] == ']':
            bullet_points.add(line[3:].strip())
    
    return list(bullet_points)

def get_technical_qs(num_generations, prompt, query_engine):
    """ Function to query for QUESTIONS wrt procedure prompt, extracts as bullet point list"""
    all_questions = []
    data_for_df = []

    for gen_num in range(num_generations):
        txt = f"In order to '{prompt}', create a concise list of very basic and fundamental questions that explore the essential properties, definitions, and background relevant to this topic."
        response = query_engine.query(txt)
        questions = extract_bullet_points(response.response)
        all_questions.append(questions)
        for ques in questions:
            data_for_df.append({"Prompt": prompt, "Question": ques})
    flat_questions = list(itertools.chain.from_iterable(all_questions))
    qcount = len(flat_questions)
    df = pd.DataFrame(data_for_df, columns=["Prompt", "Question"])

    return df, flat_questions, qcount


def ans_technicals(df, query_engine):
    """Function that intakes the previously generated df to generate ANSWERS for the QUESTIONS"""
    answers = []
    for idx, row in df.iterrows():
        question = row["Question"]
        txt =f"{question}. Answer concisely and accurately. If you don't know the answer or there isn't enough context from the provided information, state that this area needs further exploration. Do not use citations."
        answer = query_engine.query(txt).response
        answers.append(answer)
        df.at[idx, "Answer"] = answer

    return df, answers
