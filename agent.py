import os, glob
from dotenv import load_dotenv
from together import Together

load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

def read(path): 
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

def load_knowledge(folder="knowledge"):
    texts=[]
    for f in glob.glob(f"{folder}/*.txt"):
        texts.append(f"\n# {os.path.basename(f)}\n{read(f)}")
    return "\n".join(texts).strip() or "(no extra knowledge)"

def ask(messages):
    r = client.chat.completion.create(model=MODEL, messages=messages)
    return r.choices[0].message.content

def run(task):
    system = read("prompt.md") or "You are a helpful, concise assistant."
    knowledge = load_knowledge()

    plan = ask([
        {"role":"system","content":system},
        {"role":"user","content":f"Task: {task}\nUse this info:\n{knowledge}\nMake a tight PLAN with steps and risks."}
    ])

    final = ask([
        {"role":"system","content":system},
        {"role":"user","content":f"Based on this PLAN:\n{plan}\nCreate the final answer for the task."}
    ])
    return plan, final

if __name__ == "__main__":
    plan, final = run("Summarize my notes and propose 5 action items for next week.")
    print("\n--- PLAN ---\n", plan[:800])
    print("\n--- FINAL ---\n", final[:800])
