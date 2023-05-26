import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAI } from "langchain/llms/openai";
import { PromptTemplate } from "langchain/prompts";
import { HumanChatMessage, SystemChatMessage } from "langchain/schema";
import { LLMChain } from "langchain/chains";
import { PromptLayerOpenAI } from "langchain/llms/openai";
import 'dotenv/config'

(async function main() {
  const llm = new OpenAI({modelName: "text-davinci-003", temperature: 0.9,
    n:1, maxTokens: 140});

  const res1 = await llm.call("Generate a funny tweet about .NET Core",
    { timeout: 10000 });

  console.log(res1 + "\n");

  const chat = new ChatOpenAI({ temperature: 0.9 });

  const res2 = await chat.call([
      new SystemChatMessage("You are a bot that answers wrong"),
      new HumanChatMessage(
        "How old is earth?"
      ),
    ]);
    

  console.log(res2);
  console.log("\n");
    
  const template = "Generate a {sentiment} tweet about {topic}"
  const prompt = new PromptTemplate({
      template: template,
      inputVariables: ["sentiment", "topic"],
  });

  // check the prompt
  const fullPrompt = await prompt.format({ sentiment: "serious", topic: "the weather" });
  console.log(fullPrompt + "\n");

  // create a chain
  const chain = new LLMChain({llm: llm, prompt: prompt});
  const res3 = await chain.call({ sentiment: "funny", topic: "the weather" });

  console.log(res3);

  // use prompt layer; see https://promptlayer.com
  // only works with standard completions, not with chat
  const model = new PromptLayerOpenAI({
    temperature: 0.9,
  })

  const res4 = await model.call(
    "What's a good name for a Kubernetes cluster?"
  );

  console.log(res4);

})();
  


