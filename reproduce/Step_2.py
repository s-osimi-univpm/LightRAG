import json
from transformers import GPT2Tokenizer
import requests
import os


# === SOSTITUISCE OpenAI ===
def ollama_complete(prompt, model="gemma4:12b", host="http://localhost:11440"):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 32768*8
            },
        }
    )
    return response.json()["response"]


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def get_summary(context, tot_tokens=2500):
    tokens = tokenizer.tokenize(context)
    half_tokens = tot_tokens // 2

    start_tokens = tokens[1000 : 1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens) : -1000]
    #end_tokens = tokens[-(1000 + half_tokens) : 1000]

    summary_tokens = start_tokens + end_tokens
    summary = tokenizer.convert_tokens_to_string(summary_tokens)

    return summary


# === MINIMA MODIFICA PATH ===
clses = ["mix"]#["agriculture","legal"] #["legal"]

for cls in clses:
    with open(f"reproduce/dataset/unique_contexts/{cls}_unique_contexts.json", mode="r") as f:
        unique_contexts = json.load(f)

    summaries = [get_summary(context) for context in unique_contexts]

    total_description = "\n\n".join(summaries)

    prompt = f"""
    Given the following description of a dataset:

    {total_description}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """

    # ollama backend
    result = ollama_complete(prompt)

    # ✅ salva in results (non dataset)
    output_dir = "reproduce/questions"
    os.makedirs(output_dir, exist_ok=True)

    file_path = f"{output_dir}/{cls}_questions.txt"

    with open(file_path, "w") as file:
        file.write(result)

    print(f"{cls}_questions written to {file_path}")
