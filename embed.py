import os
import numpy as np
import re
import random
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai
from flask import Flask, request, jsonify, send_from_directory, render_template_string

# Load environment variables
load_dotenv()

# Check if the API key is set
openai.api_key ="YOUR_API_KEY"
client = openai

# Constants
MAX_HISTORY = 5
EMBEDDING_MODEL = "text-embedding-ada-002"
global doc_texts
conversation_history = []

app = Flask(__name__)

def get_next_1000_words(text, num_words=1000):
    words = text.split()
    total_words = len(words)
    if total_words <= num_words:
        return ' '.join(words)
    start_index = random.randint(0, total_words - num_words)
    selected_words = words[start_index:start_index + num_words]
    result = ' '.join(selected_words)
    return result

def get_embeddings(client, text, model=EMBEDDING_MODEL):
    tlist = "\n".join(text)
    tlist = get_next_1000_words(tlist, 1000)
    response = client.embeddings.create(input=tlist, model=model)
    qe = np.array(response.data[0].embedding)
    qe = qe.reshape(1, -1)
    return qe

def search_documents(client, question, doc_texts, doc_embeddings, top_k=3):
    question_embedding = get_embeddings(client, [question])
    similarities = cosine_similarity(question_embedding, doc_embeddings)
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [doc_texts[i] for i in top_indices]

def sanitize_input(input_text):
    sanitized_text = re.sub(r'[<>]', '', input_text.strip()[:1000])
    return sanitized_text

def extract_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(text)
    return urls

def ask_question(client, question, relevant_docs):
    global conversation_history
    question = sanitize_input(question)
    context = "\n\n".join(relevant_docs)
    messages = [
        {"role": "system", "content": f"Using the below contexts:\n\n{context}\n\n**Please answer the following question.**\n{question}"}
    ]
    for history in conversation_history[-MAX_HISTORY:]:
        messages.append({"role": "user", "content": sanitize_input(history['question'])})
        messages.append({"role": "assistant", "content": sanitize_input(history['answer'])})
    messages.append({"role": "user", "content": question})
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = chat_completion.choices[0].message.content
    conversation_history.append({"question": question, "answer": answer})
    return answer

def init_embed(client):
    global doc_texts
    file_path = 'YOUR_PATH_/thop.txt' #add file path here
    try:
        with open(file_path, "r") as infile:
            lines = infile.readlines()
    except Exception as e:
        print(f'Please make sure the file {file_path} is in your current directory')
        exit(1)
    doc_texts = lines
    doc_embeddings = get_embeddings(client, doc_texts)
    return doc_embeddings

def get_embed_answer(client, doc_embeddings, question):
    global doc_texts
    relevant_docs = search_documents(client, question, doc_texts, doc_embeddings)
    if relevant_docs == ['O']:
        return 'I did not find any relevant documents for your question.'
    answer = ask_question(client, question, relevant_docs)
    return answer

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'messages' not in data:
        return jsonify({"error": "Invalid request"}), 400

    question = data['messages'][-1]['content']
    
    # Initialize embeddings
    doc_embeddings = init_embed(client)

    # Generate answer for the question
    response = get_embed_answer(client, doc_embeddings, question)
    urls = extract_urls(response)

    print(f"Question: {question}")
    print(f"Answer: {response}")
    print(f"Extracted URLs: {urls}")

    return jsonify({
        "answer": response,
        "urls": urls
    })

# Added route for the root URL with a form for user input
@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ""
    if request.method == 'POST':
        question = request.form['question']
        doc_embeddings = init_embed(client)
        answer = get_embed_answer(client, doc_embeddings, question)
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG App API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f8f9fa;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                h1 {
                    color: #343a40;
                }
                form {
                    margin-bottom: 20px;
                }
                label {
                    font-weight: bold;
                }
                input[type="text"] {
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                }
                input[type="submit"] {
                    background-color: #007bff;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
                .result {
                    background-color: #e9ecef;
                    padding: 10px;
                    border-radius: 4px;
                }
                .question, .answer {
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to the RAG App API</h1>
                <form method="post">
                    <label for="question">Enter your question:</label><br>
                    <input type="text" name="question" id="question" value="{{ question }}" required><br><br>
                    <input type="submit" value="Ask">
                </form>
                {% if answer %}
                <div class="result">
                    <h2>Result</h2>
                    <div class="question"><strong>Question:</strong> {{ question }}</div>
                    <div class="answer"><strong>Answer:</strong> {{ answer }}</div>
                </div>
                {% endif %}
            </div>
        </body>
        </html>
    ''', answer=answer, question=question)

# Added route for the favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    # List of questions to be answered
    qlist = [
        "How did the rise of Christianity influence the history of Palestine?",
        "What were the key developments during the Islamic Arabic Conquest of Palestine?", 
        "What role did the Fatimids play in the history of Palestine?", 
        "What was the impact of the British Mandate on Palestine from 1923 to 1948?"
    ]

    # Initialize embeddings
    doc_embeddings = init_embed(client)

    # Process and generate answers for each question
    for question in qlist:
        response = get_embed_answer(client, doc_embeddings, question)
        urls = extract_urls(response)
        print(f"Question: {question}")
        print(f"Answer: {response}")
        print(f"Extracted URLs: {urls}")
        print("=" * 150)

    app.run(debug=True)
