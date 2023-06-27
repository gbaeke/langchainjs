import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { VectorDBQAChain } from "langchain/chains";
import readline from 'readline';
import path from 'path';
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import 'dotenv/config'

const envPath = path.join(__dirname, '..', '.env');
require('dotenv').config({ path: envPath });



(async function main() {
    const loader = new DirectoryLoader("./articles",
        {
            ".txt": (path) => new TextLoader(path),
        });

    const docs = await loader.load();

    // how many pages?
    console.log('Pages loaded: ', docs.length);

    // create a vector store, here we use the memory vector store
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

    // use a chat model to answer questions
    const model = new ChatOpenAI({ temperature: 0 });

    // create a vector db qa chain using the in memory vector store and the chat model
    const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
      k: 2,
      returnSourceDocuments: true,
    });

    const res = await chain.call({ query: "How can I lock my room?" });
    console.log({ res });

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });
      
    async function askForInput() {
        while (true) {
            const input: string = await new Promise(resolve => {
            rl.question('Enter something (type "quit" to exit): ', resolve);
            });
        
            if (input.toLowerCase() === 'quit') {
                rl.close();
                break;
            }

            const chat1 = await chain.call({ query: input });
            console.log({chat1, memory: await chain.memory?.loadMemoryVariables({})});
        }
    }
    
    askForInput();

})();

