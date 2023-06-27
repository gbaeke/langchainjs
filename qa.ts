import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain, loadQAMapReduceChain, loadQARefineChain } from "langchain/chains";
import { Document } from "langchain/document";
import 'dotenv/config'
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";



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
  const chainB = loadQARefineChain(llmB);
  const loader = new PDFLoader("pdf_docs/ebpf.pdf");
  const pdfDocs = await loader.load();

  const question = "Who wrote this eBPF book?";

  // get only relevant documents to avoid going over every page in the PDF
  const store = await MemoryVectorStore.fromDocuments(pdfDocs, new OpenAIEmbeddings());
  const relevantDocs = await store.similaritySearch(question, 5);

  const resB = await chainB.call({
    input_documents: relevantDocs,
    question: question,
  });
  console.log({ resB });

})();
  


