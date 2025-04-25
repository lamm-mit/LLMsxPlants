import random
import json
import pandas as pd

def clean_json_string(json_string):
    cleaned_string = json_string.replace("\n", " ")
    return cleaned_string

def load_ideas_from_files(filenames):
    all_ideas = []
    for filename in filenames:
        df = pd.read_csv(filename)
        ideas = df.iloc[:, 1].tolist()
        all_ideas.extend(ideas)
    return all_ideas


def pairwise_elim_and_rate(filenames, prompt, round_limit, outputfilename, ratings_outputfile, query_engine):
    all_ideas = load_ideas_from_files(filenames)
    random.shuffle(all_ideas)
    round_number = 1
    all_rounds_data = []  # List to collect data for each round
    ratings_data = []  # List to store ratings data for each idea

    while len(all_ideas) > 1 and round_number <= round_limit:
        print(f"--- Round {round_number} ---")
        winners = []
        # Compare ideas in pairs
        for i in range(0, len(all_ideas), 2):
            if i + 1 < len(all_ideas):  
                idea_1 = all_ideas[i]
                idea_2 = all_ideas[i + 1]

                # Only rate the ideas in the first round
                if round_number == 1:
                    # Prepare the rating prompt for these two ideas
                    rating_prompt = f"""To answer this question: {prompt}, ideas were brainstormed. You must critically rate each idea out of 10 for categories: novelty and effectiveness/feasibility.
                    The response must only contain the ratings in the following strict JSON format:
                    {{
                    "Idea 1": {{"novelty": X, "effectiveness": Y}},
                    "Idea 2": {{"novelty": X, "effectiveness": Y}}
                    }}"""
                    
                    ideas_str = f"Idea 1: {idea_1}\nIdea 2: {idea_2}"
                    rating_txt = f"{rating_prompt}\n\n{ideas_str}"
                    rating_response = query_engine.query(rating_txt)
                    cleaned_rating_response = clean_json_string(rating_response.response)

                    try:
                        ratings = json.loads(cleaned_rating_response)
                        ratings_data.append({
                            'Idea': idea_1,
                            'Novelty': ratings.get('Idea 1', {}).get('novelty', None),
                            'Effectiveness': ratings.get('Idea 1', {}).get('effectiveness', None)
                        })
                        ratings_data.append({
                            'Idea': idea_2,
                            'Novelty': ratings.get('Idea 2', {}).get('novelty', None),
                            'Effectiveness': ratings.get('Idea 2', {}).get('effectiveness', None)
                        })
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        continue  

                compare_txt = f"""To answer this question: {prompt} Which idea is better based on novelty and effectiveness?
                Idea 1: {idea_1} 
                Idea 2: {idea_2} 
                Respond with the winner as 'Idea 1' or 'Idea 2' in the following JSON format:
                {{ "winner": "Idea 1" or "Idea 2" }}
                """

                comparison_response = query_engine.query(compare_txt)
                cleaned_comparison_response = clean_json_string(comparison_response.response)

                try:
                    result = json.loads(cleaned_comparison_response)
                    winner = idea_1 if result["winner"] == "Idea 1" else idea_2
                    winners.append(winner)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
            
            else:
                # If there's an odd number of ideas, move the last one directly to the next round
                winners.append(all_ideas[i])

        # Store the remaining ideas after the round
        round_data = [{'Round': round_number, 'Idea': idea} for idea in winners]
        all_rounds_data.extend(round_data)

        # Move the winners to the next round
        all_ideas = winners
        random.shuffle(all_ideas)  # Randomize for the next round
        round_number += 1

    print(f"Final round winners (Top {len(all_ideas)} ideas): {[idea for idea in all_ideas]}")
    
    df_winners = pd.DataFrame(all_ideas)
    df_winners['Round'] = round_number - 1  
    output_file = outputfilename
    df_all_rounds = pd.DataFrame(all_rounds_data)
    final_output_df = pd.concat([df_all_rounds, df_winners])
    final_output_df.to_csv(output_file, index=False)

    print(f"Results (including rounds) exported to {output_file}.")

    df_ratings = pd.DataFrame(ratings_data)
    df_ratings.to_csv(ratings_outputfile, index=False)
    
    print(f"Ratings from the first round exported to {ratings_outputfile}.")
