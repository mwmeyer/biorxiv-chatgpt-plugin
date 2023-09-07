import json
import quart
import quart_cors
from quart import request
import requests
import urllib
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()

app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

BIORXIV_URL = 'https://www.biorxiv.org'

CSS_SELECTORS = {
    'article_div': 'highwire-cite highwire-cite-highwire-article highwire-citation-biorxiv-article-pap-list clearfix',
    'title': 'highwire-cite-title',
    'authors_div': 'highwire-cite-authors',
    'author': 'highwire-citation-author',
    'doi': 'highwire-cite-metadata-doi',
    'link': 'highwire-cite-linked-title'
}

def extract_webpage_articles(html):
    soup = BeautifulSoup(html, 'html.parser')
    articles = []
    for result in soup.find_all('div', class_=CSS_SELECTORS['article_div']):
        title = result.find('span', class_=CSS_SELECTORS['title']).text.strip()
        authors_div = result.find('div', class_=CSS_SELECTORS['authors_div'])
        authors = [a.text.strip() for a in authors_div.find_all('span', class_=CSS_SELECTORS['author'])] if authors_div else []
        doi_url = result.find('span', class_=CSS_SELECTORS['doi']).text.strip()
        # extract doi after https://doi.org/DOI
        doi = doi_url.split('.org/')[1]
        pdf = f'https://www.biorxiv.org/content/{doi}.full.pdf'
        link = 'https://www.biorxiv.org' + result.find('a', class_=CSS_SELECTORS['link'])['href']
        articles.append({
            'title': title,
            'authors': authors,
            'pdf': pdf,
            'link': link,
        })
    return articles

def generate_search_url(query):
    query_string = f'{query} numresults:25'
    return f"{BIORXIV_URL}/search/{urllib.parse.quote(query_string)}"

@app.route("/search_biorxiv", methods=['GET'])
async def search_biorxiv():
    query = request.args.get('query')
    url = generate_search_url(query)
    response = requests.get(url)
    articles = extract_webpage_articles(response.text)

    return quart.Response(json.dumps(articles), mimetype='application/json')


def extract_text_from_pdf(pdf_path, page=None):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ""
    
        if page is not None:
            page = page - 1 # pdf_reader.pages is 0-indexed
            text += pdf_reader.pages[page].extract_text()
        else:
            for page_content in pdf_reader.pages:
                text += page_content.extract_text()
                
    return text

def get_local_pdf_path(pdf_url):
    return "./pdfs/" + pdf_url.split("/")[-1]

def download_and_save_pdf(pdf_url):
    response = requests.get(pdf_url)
    local_path = get_local_pdf_path(pdf_url)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    return local_path

@app.route("/download_pdf", methods=['GET'])
async def download_pdf():
    pdf = request.args.get('pdf')
    local_path = get_local_pdf_path(pdf)
    
    if not os.path.exists(local_path):
        download_and_save_pdf(pdf)
    
    return quart.jsonify({'content': "success"})

@app.route("/extract_text", methods=['GET'])
async def extract_text():
    page = request.args.get('page', default=1, type=int)
    pdf = request.args.get('pdf')
    local_path = get_local_pdf_path(pdf)
    if not os.path.exists(local_path):
        download_and_save_pdf(pdf)
    
    text = extract_text_from_pdf(local_path, page)
    return quart.jsonify({'content': text})

def build_knowledge_base(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

@app.route("/ask_corpus", methods=['GET'])
async def ask_corpus():
    query = request.args.get('query')
    pdf = request.args.get('pdf')
    local_path = get_local_pdf_path(pdf)
    if not os.path.exists(local_path):
        download_and_save_pdf(pdf)
    text = extract_text_from_pdf(local_path)

    knowledge_base = build_knowledge_base(text)
    
    docs = knowledge_base.similarity_search(query)

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    
    with get_openai_callback() as cost:
        response = chain.run(input_documents=docs, question=query)
        print(cost)
    return response

@app.get("/logo.png")
async def plugin_logo():
    filename = 'logo.png'
    return await quart.send_file(filename, mimetype='image/png')

@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    host = request.headers['Host']
    with open("./.well-known/ai-plugin.json") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/json")

@app.get("/openapi.yaml")
async def openapi_spec():
    host = request.headers['Host']
    with open("openapi.yaml") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/yaml")

def main():
    app.run(debug=True, host="0.0.0.0", port=5003)

if __name__ == "__main__":
    main()
