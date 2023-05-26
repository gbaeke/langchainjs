import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { VectorDBQAChain } from "langchain/chains";
import readline from 'readline';




// load from dotenv
import 'dotenv/config'


(async function main() {
    const loader = new PDFLoader("pdf_docs/ebpf.pdf");

    // create one document per page (default)
    // add { splitPages: false } to create one document for the whole PDF
    const docs = await loader.load();

    // how many pages?
    console.log(docs.length);

    // create a vector store, here we use the memory vector store
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

    // use a chat model to answer questions
    const model = new ChatOpenAI({ temperature: 0 });

    // create a vector db qa chain using the in memory vector store and the chat model
    const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
      k: 5,
      returnSourceDocuments: true,
    });

    // kick of the questions, you can ask further questions such as
    // "Can it be used for networking?"
    const res = await chain.call({ query: "What is eBPF?" });
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

