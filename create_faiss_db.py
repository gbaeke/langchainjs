from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import feedparser

# load env vars
load_dotenv()

tokenizer = tiktoken.get_encoding('cl100k_base')

def get_feed_urls(feed_url, entries):
    # Parse the RSS feed using feedparser
    feed = feedparser.parse(feed_url)

    # Extract the URLs from the first 10 entries in the feed
    urls = []
    for entry in feed.entries[:entries]:
        urls.append(entry.link)

    return urls


# used as len function for text_splitter
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_page_text(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        page = soup.find("div", {"class": "entry-content"})
        if page:
            page = page.text
        else:
            print('Error: Could not find entry content')
            return None
    else:
        print('Error:', response.status_code)
        return None

    return page

feed_url = 'https://blog.baeke.info/feed/'
urls = get_feed_urls(feed_url, 10)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20, separators=['\n\n', '\n', ' ', ''], length_function=tiktoken_len)
embeddings = OpenAIEmbeddings(client=None)

docs = []
for entry in urls:
    page_text = get_page_text(entry)
    if page_text:
        docs_for_entry = text_splitter.split_text(page_text)
        docs.extend(docs_for_entry)

db = FAISS.from_texts(docs, embeddings)

result = db.similarity_search('What is streamlit?', k=1)
print(result)

db.save_local('faiss_db')








