from difflib import SequenceMatcher
import pandas as pd

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

def are_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() > threshold

def sample_bullets(num_generations, num_ideas, prompt, query_engine):
    """Samples the LLM num_generations number of times, scraping the bullet points of ideas generated"""
    all_bullet_points = set()
    bullet_points_count = []
    bullet_points_per_generation = []
    data_for_df =[]
    for gen_num in range(num_generations):
        txt = f"{prompt} Be very creative but also concise, specific, and practical. Concisely brainstorm {num_ideas} different ideas into a bullet point list."
        response = query_engine.query(txt)
        bullet_points = extract_bullet_points(response.response)
        all_bullet_points.update(bullet_points)
        for bullet_point in bullet_points:
            data_for_df.append({"Prompt": prompt, "Idea": bullet_point})
        bullcount = len(all_bullet_points)
    df = pd.DataFrame(data_for_df, columns=["Prompt", "Idea"])
    return df, all_bullet_points, bullcount


def filter_ideas(new_ideas, unique_ideas, similarity_threshold=0.8):
    """Filters ideas in the list by similarity, can adjust similarity threshold"""
    filtered_ideas = []
    for idea in new_ideas:
        is_unique = all(not are_similar(idea, existing_idea, similarity_threshold) for existing_idea in unique_ideas)
        if is_unique and all(not are_similar(idea, filtered_idea, similarity_threshold) for filtered_idea in filtered_ideas):
            filtered_ideas.append(idea)
    return filtered_ideas

def filter_unique_ideas(df_existing, df_new, similarity_threshold=0.8):
    unique_ideas = df_existing['Idea'].tolist()
    new_ideas = df_new['Idea'].tolist()
    filtered_ideas = filter_ideas(new_ideas, unique_ideas, similarity_threshold)
    df_filtered = df_new[df_new['Idea'].isin(filtered_ideas)]
    return pd.concat([df_existing, df_filtered], ignore_index=True)
