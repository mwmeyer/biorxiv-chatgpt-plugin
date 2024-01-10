# bioRxiv ChatGPT Plugin

This is a ChatGPT plugin designed to facilitate browsing of biorxiv.org from https://chat.openai.com.

The plugin enables users to search, download, and extract text of paper PDFs.

## Limitations
Because LLMs have a maximum context length, entire contents of a paper can't be sent directly to ChatGPT all at once.

The plugin trys to overcome this limitation in two ways:
1. By letting chatgpt pass the page number to extract text from. You can therefore read the paper gradually to incrementally build context.
2. (optional) By providing a [langchain based Q&A](https://medium.com/@johnthuo/chat-with-your-pdf-using-langchain-f-a-i-s-s-and-openai-to-query-pdfs-e7bfde086155) functionality that converts the entire text corpus to embeddings. You can tell chatgpt to "ask" the plugin specifc questions about the paper, and it attempt to return a response based on matching terminology in a ephemeral vector store.

## Setup locally

An OPENAI_API_KEY key is needed for the Q&A functionality. Langchain makes calls to OpenAI so a cost is incurred for each question.

To install the required packages for this plugin, run the following command:

```bash
pip install -r requirements.txt
```


To start the plugin, enter the following command:

```bash
python main.py
```

Once the local server is running:

Navigate to https://chat.openai.com.
In the Model drop down, select "Plugins" (note, if you don't see it there, you don't have access yet).
Select "Plugin store"
Select "Develop your own plugin"
Enter in localhost:5003 since this is the URL the server is running on locally, then select "Find manifest file".
The plugin should now be installed and enabled!