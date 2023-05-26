import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import axios from 'axios';
import cheerio from 'cheerio';




import 'dotenv/config'

(async function main() {
    const webResponse=await axios.get("https://blog.baeke.info/2023/03/21/storing-and-querying-for-embeddings-with-redis/")
    const $ = cheerio.load(webResponse.data);
    const content = $('div.entry-content').text();

    // split text
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 400,
        chunkOverlap: 50,
    });

    const chunks = await splitter.createDocuments([content]);

    // create vector store
    const vectorStore = await FaissStore.fromDocuments(
        chunks, new OpenAIEmbeddings()
    );

    const result = await vectorStore.similaritySearch("Redis", 1);
    console.log(result);

    vectorStore.save('faiss-store');
      

})();
  


