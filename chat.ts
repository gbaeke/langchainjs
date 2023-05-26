import { ChatOpenAI } from "langchain/chat_models/openai";
import { CallbackManager } from "langchain/callbacks";
import { HumanChatMessage, SystemChatMessage } from "langchain/schema";
import 'dotenv/config'
import process from 'process'
import {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "langchain/prompts";
import { LLMChain, ConversationChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import * as readline from 'readline';


(async function main() {

  const chat = new ChatOpenAI({ modelName: "gpt-4", streaming: true });

  const res = await chat.call([
    new SystemChatMessage("You are a bot that answers wrong"),
    new HumanChatMessage("What's the capital of France?")],
    undefined,
    [
      {
        handleLLMNewToken(token: string) {
          process.stdout.write( token );
        },
      }
    ]
  );

  console.log("\n");
  console.log(res);

  // using prompt templates with chat
  const botPrompt = ChatPromptTemplate.fromPromptMessages(
    [
      SystemMessagePromptTemplate.fromTemplate("You are a bot that answers {mode}"),
      HumanMessagePromptTemplate.fromTemplate("{question}"),
    ]
  );

  // I do not like the approach of having to pass the prompt template to the chat
  // const res2 = await chat.generatePrompt(
  //   [
  //     await botPrompt.formatPromptValue({ mode: "wrong", question: "How old is Earth?" }),
  //   ]
  // );

  // instead, let's use a chain
  const chain = new LLMChain({
    prompt: botPrompt,
    llm: chat,
  });

  const res2 = await chain.call({
    mode: "wrong",
    question: "How old is Earth?",
  });

  console.log(res2.text);

  // let's turn to chatting where the first step is a chat prompt template
  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      "You are a funny chatbot that provides funny answers to users. Use the context to answer the user."
    ),
    new MessagesPlaceholder("history"),  // this is where the history of the chat will be inserted
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  // with a chat model, ensure memory returns a list of chat messages, not just a string
  // you need to use returnMessages: true and pass a memoryKey; use that memoryKey in the prompt
  // use MessagesPlaceholder to insert the history into the prompt
  const chatChain = new ConversationChain({
    prompt: chatPrompt,
    llm: chat,
    memory: new BufferMemory({ returnMessages: true, memoryKey: "history" })
  });

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

      const chat1 = await chatChain.call({ input: input });
      console.log({chat1, memory: await chatChain.memory?.loadMemoryVariables({})});
    }
  }

  askForInput();

})();
  


