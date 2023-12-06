from flask import Flask, send_file, abort, Response
import os
import pinecone, openai
import json

api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=api_key, environment=env)
index = pinecone.GRPCIndex('ref-jeff')
print(pinecone.whoami())
print(index.describe_index_stats())
app = Flask(__name__)

@app.route('/')
def home():
    return 'fish'

@app.route('/hello')
def hello_world():
    return 'Hello, World!'

@app.route('/get-pdf/<path:filename>')
def get_pdf(filename):
    directory = 'corpus'
    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        abort(404)

    return send_file(file_path, as_attachment=True)

@app.route('/query/<path:query>')
def query(query):
    res = openai.Embedding.create(
        input=[query],
        engine='text-embedding-ada-002'
    )
    xq = res['data'][0]['embedding']
    res = index.query(xq, top_k=200, include_metadata=True)
    res = [match['metadata']['rel_path'] for match in res['matches']]
    res_json = json.dumps(res)
    return Response(res_json, mimetype='application/json')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
