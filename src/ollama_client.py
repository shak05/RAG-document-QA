import ollama

class OllamaClient:

    def generate_answer(
        self,
        question,
        context
    ):

        prompt = f"""
You are a document question answering assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, respond exactly:

I could not find that information in the uploaded documents.

Context:
{context}

Question:
{question}
"""

        try:

            response = ollama.chat(
                model="llama3:latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response["message"]["content"]

        except Exception as e:

            return f"Ollama Error: {str(e)}"