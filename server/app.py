from flask import Flask, render_template, request
import os
import sys

sys.path.insert(0, 'C:/Users/madhu/projects/chat_with_lotm')
from scripts import embedding
from groq import Groq

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['GET','POST'])
def chat(result=None):
    if request.method == 'POST':
        with open('resources/paragraphs.txt', 'r', encoding='utf-8') as file:
            paragraphs = file.readlines()
        
        paragraphs = paragraphs[:93524]
        groq_api_key = os.environ.get('GROQ_API_KEY')
        query = request.form['input_text']

        scores, indices = embedding.similarity_search(query=query, embeddings_file_path='resources/embeddings.pt', top_k=5)
        context = embedding.get_context(indices=indices, paragraphs=paragraphs)

        client = Groq(
            api_key=groq_api_key,
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                        You are a novel reader who've completed the novel and now conversing with a fellow novel reader casually and answering some queries. You are provided with some context which is enough to answer the queries.
                            Query : {query}
                            Context : {context}
                    Answer with as many details as possible and as relevant data as possible, with concise description. Also gather everything from the context and present it as a if you've read the whole novel rather than based on the scene. Also dont mention the characters appearing in the context, just try to integrate their stories into your answer rather than directly repeating the scene."""
                }
            ],
            model="llama3-8b-8192",
        )
        result = chat_completion.choices[0].message.content
        return render_template('index.html', result=result)

app.run(debug=True)