import re
import json
import requests


def ollama_eval(prompt, model="gemma4:31b", host="http://localhost:11440"):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def batch_eval(result1_file, result2_file, output_file_path):
    #with open(query_file, "r") as f:
    #    data = f.read()

    #queries = re.findall(r"- Question \d+: (.+)", data)

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    queries = [i["query"] for i in answers1]
    answers1 = [i["result"] for i in answers1]

    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    answers2 = [i["result"] for i in answers2]

    results = []

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        print(f"Evaluating query {i+1}/{len(queries)}")

        prompt = f"""
            You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

            You will evaluate two answers to the same question based on these criteria:

            - Comprehensiveness: How much detail does the answer provide to cover all aspects of the question?
            - Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
            - Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

            Choose the better answer (Answer 1 or Answer 2) for each criterion and explain why. Then select an overall winner.

            Question:
            {query}

            Answer 1:
            {answer1}

            Answer 2:
            {answer2}

            Output JSON:
            {{
                "Comprehensiveness": {{"Winner": "...", "Explanation": "..."}},
                "Diversity": {{"Winner": "...", "Explanation": "..."}},
                "Empowerment": {{"Winner": "...", "Explanation": "..."}},
                "Overall Winner": {{"Winner": "...", "Explanation": "..."}}
            }}
            """

        evaluation = ollama_eval(prompt)

        results.append({
            "query": query,
            "evaluation": evaluation
        })

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Evaluation saved to {output_file_path}")


if __name__ == "__main__":
    clses = ['mix']#["agriculture","legal"]

    for cls in clses:
        batch_eval(
            f"reproduce/results/{cls}_hybrid_results.json",
            f"reproduce/results/{cls}_naive_results.json",
            f"reproduce/results/{cls}_evaluation.json",
        )