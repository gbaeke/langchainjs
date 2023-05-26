import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAI } from "langchain/llms/openai";
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate } from "langchain/prompts";
import { HumanChatMessage, SystemChatMessage } from "langchain/schema";
import { SimpleSequentialChain, LLMChain } from "langchain/chains";
import { PromptLayerOpenAI } from "langchain/llms/openai";
import 'dotenv/config'

(async function main() {
  // we have already see chains in action, let's create a simple chain as a reminder
  // we will use a chat model that defaults to gpt-3.5-turbo

  // chat model will default to gpt-3.5-turbo
  const chat = new ChatOpenAI({ temperature: 0.9 });

  // prompt template for chat with system message and one question from the human
  const botPrompt = ChatPromptTemplate.fromPromptMessages(
    [
      SystemMessagePromptTemplate.fromTemplate("You are a quirky bot that answers questions {mode}"),
      HumanMessagePromptTemplate.fromTemplate("{question}"),
    ]
  );

  // create a chain linking the chat model and the prompt template
  const chain = new LLMChain({
    prompt: botPrompt,
    llm: chat,
  });

  // use chain.call instead of run (multiple params require call)
  const res = await chain.call({ mode: "wrong", question: "How old is Earth?" });

  console.log(res);
  
  // now let's use a sequential chain with a non-chat model
  const llm = new OpenAI({ temperature: 0.8, maxTokens: 1000 });

  const scriptTemplate = new PromptTemplate({
    template: "Create a movie script for the following movie title: {title}",
    inputVariables: ["title"],
  })

  const scriptChain = new LLMChain({ llm, prompt: scriptTemplate });

  const reviewTemplate = new PromptTemplate({
    template: `You are a cranky movie critic that writes a review given the synopsis of a movie.
    
    Movie synopsis:
    {movie}
    Review from the critic:`,
    inputVariables: ["movie"],
  })

  const reviewChain = new LLMChain({ llm, prompt: reviewTemplate });

  const movieChain = new SimpleSequentialChain({
    chains: [scriptChain, reviewChain],
    verbose: true,
  });

  const review = await movieChain.run("A day in the life of a .NET Developer");
  console.log(review);


})();
  


