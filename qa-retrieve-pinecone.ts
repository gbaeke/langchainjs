import { OpenAI } from "langchain/llms/openai";
import 'dotenv/config'
import { PineconeClient } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { VectorDBQAChain } from "langchain/chains";
import process from 'process'

(async function main() {
  try {
    // Use QA with Pinecone
    const client = new PineconeClient();
    const apiKey = process.env.PINECONE_API_KEY as string;
    const environment = process.env.PINECONE_ENVIRONMENT as string;
    const index = process.env.PINECONE_INDEX as string;
  
    if (!apiKey || !environment || !index) {
      throw new Error('Missing required environment variables');
    }
  
    await client.init({ apiKey, environment });
    const pineconeIndex = client.Index(index);
  
    const vectorStore = await PineconeStore.fromExistingIndex(
      new OpenAIEmbeddings(),
      { pineconeIndex, textKey: "text" }
    );
  
    const model = new OpenAI({ temperature: 0 });


    const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
      k: 5,
      returnSourceDocuments: true,
    });
    const res = await chain.call({ query: "Can Redis be used as a vector database?" });
    console.log({ res });
    
    const docs = res.sourceDocuments;

    console.log({ docs });

  } catch (error) {
    console.error('An error occurred:', error);
  }

})();
  


