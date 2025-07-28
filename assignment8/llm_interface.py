from transformers import pipeline

# Using FLAN-T5
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = qa_pipeline(prompt, max_new_tokens=100)[0]['generated_text']
    return response
