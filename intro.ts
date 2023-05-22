import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAI } from "langchain/llms/openai";
import { PromptTemplate } from "langchain/prompts";
import { HumanChatMessage, SystemChatMessage } from "langchain/schema";
import 'dotenv/config'

(async function main() {
  const llm = new OpenAI();

  const res1 = await llm.call("Generate a funny tweet about .NET Core");

  console.log(res1 + "\n");

  const chat = new ChatOpenAI({ temperature: 0.9 });

  const res2 = await chat.call([
      new SystemChatMessage("You are a bot that answers wrong"),
      new HumanChatMessage(
        "How old is earth?"
      ),
    ]);
    

  console.log(res2 + "\n");
    
  const template = "Generate a {sentiment} tweet about {topic}"
  const prompt = new PromptTemplate({
      template: template,
      inputVariables: ["sentiment", "topic"],
  });

  // check the prompt
  const fullPrompt = await prompt.format({ sentiment: "serious", topic: "the weather" });
  console.log(fullPrompt + "\n");
})();
  


