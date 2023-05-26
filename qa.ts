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

})();
  


