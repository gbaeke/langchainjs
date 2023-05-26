import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";
import { Document } from "langchain/document";
import 'dotenv/config'
import { PineconeClient } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { VectorDBQAChain } from "langchain/chains";
import process from 'process'



(async function main() {
  const llmA = new OpenAI({ verbose: true});
  const chainA = loadQAStuffChain(llmA);
  const docs = [
    new Document({ pageContent: "Geert Baeke knows some Kubernetes." }),
    new Document({ pageContent: "John Doe is a front-end developer." }),
  ];
  const resA = await chainA.call({
    input_documents: docs,
    question: "Who's the developer?",
  });
  console.log({ resA });

  const llmB = new OpenAI({ maxConcurrency: 10, verbose: true });
  const chainB = loadQAMapReduceChain(llmB);
  const resB = await chainB.call({
    input_documents: docs,
    question: "Who is the developer?",
  });
  console.log({ resB });

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
  } catch (error) {
    console.error('An error occurred:', error);
  }
  
  
  
  
  

})();
  


