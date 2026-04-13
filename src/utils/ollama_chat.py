import ollama

def stream_chat(instruction_prompt, input_query):
    """
    Generator that yields chunks of text from Ollama.
    """
    try:
        stream = ollama.chat(
            model="gemma3:1b",
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query}
            ], 
            stream=True
        )
        for chunk in stream:
            yield chunk["message"]["content"]
    except Exception as e:
        yield f"Error generating explanation: {str(e)}"

