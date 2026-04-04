import ollama
model = "gemma3:1b"
def chat(instruction_prompt, input_query):
    stream = ollama.chat(
        model=model,
        messages=[
            {"role":"system", "content": instruction_prompt},
            {"role":"user", "content": input_query}
        ], stream=True
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)

