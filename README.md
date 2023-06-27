# Langchain with TypeScript

Instructions below use yarn. Use `brew install yarn` if you have HomeBrew installed.

```bash
yarn global add ts-node typescript '@types/node'
yarn add langchain
yarn add openai
yarn install dotenv
```

To run a .ts file (there are other ways)

```bash
ts-node intro.ts
```

Note: wrap code in the IIFE (Immediately-invoked Function Expression) to allow await

To work with OpenAI APIs, include an .env file with:

`OPENAI_API_KEY=YOUR_OPENAI_API_KEY`

For PromptLayer:

`PROMPTLAYER_API_KEY=pl_6c75db7d655d77bd4726d93b0a919bf0`


## intro.ts

Some background on the code in intro.ts:
- Uses the OpenAI API and the models provided by OpenAI: text-davinci-003 and gpt-3.5-turbo
- You can use Azure OpenAI instead of OpenAI directly:

```typescript
const model = new OpenAI({
  temperature: 0.9,
  azureOpenAIApiKey: "YOUR-API-KEY",
  azureOpenAIApiInstanceName: "YOUR-INSTANCE-NAME",
  azureOpenAIApiDeploymentName: "YOUR-DEPLOYMENT-NAME",
  azureOpenAIApiVersion: "YOUR-API-VERSION",
});
```

You can use other LLMs with LangChain:
- HuggingFace
- Cohere
- ...


Although we just use the `call` method, there are others:
- `generate`: generate multiple completions; use model.generate with an array of prompts; returns an LLMResult object
- `getNumTokens`: get number of tokens of the input (e.g., `model.getNumTokens("Some input")`)

Other capabilities:
- streaming responses
- caching: in-memory and Redis
- adding a timeout: LangChain waits indefinitely; add a timeout option to the call method
- subscribing to events: pass callbacks to the LLM for custom logging (see https://js.langchain.com/docs/production/callbacks/)


## chat.ts

⚠️ **Note**: we use OpenAI chat models but LangChain also supports ChatAnthropic

intro.ts already used a chat model. In this example:
- switched model to `gpt-4` (ensure you have access)
- turn on streaming
- use a callback in the `call` method to write the output to the screen progressively

Note that chat models work with one or more messages of a specific type:
- SystemChatMessage
- HumanChatMessage
- AIChatMessage: in the code, when we print the response, it's an AIChatMessage

You can pass multiple messages in an array, as shown in the code. Note that the system message will influence the conversation in the beginning but it is not as strong as you think. Conversation with the LLM can eventually override the system message. In our case, the LLM will answer more truthfully.

Just like with a normal completion (e.g., with text-davinci-003), you can use `generate` to generate multiple chat completions. Just pass an array of instructions, where each instruction is an array of SystemChatMessage, HumanChatMessage, etc...

### Using prompts with chat

Use `chatPromptTemplate` to create a template to use with chat models:

```typescript
const botPrompt = ChatPromptTemplate.fromPromptMessages(
    [
      SystemMessagePromptTemplate.fromTemplate("You are a bot that answers {mode}"),
      HumanMessagePromptTemplate.fromTemplate("{question}"),
    ]
  );
```

The simplest way to use this prompt with your defined model is with a chain. So if you have a model like:

```typescript
const chat = new ChatOpenAI({ modelName: "gpt-4", streaming: true });
```

You can build a chain with the prompt & llm and then call the chain. Grab the text property from the output to only grab the response from the chat model:

```typescript
const chain = new LLMChain({
    prompt: botPrompt,
    llm: chat,
  });

const res2 = await chain.call({
mode: "wrong",
question: "How old is Earth?",
});

console.log(res2.text);
```

The above would print something like: `Earth is 37 years old.` 😂

### Memory

The last piece of code in chat.ts introduces `memory`. We use the simplest form of memory: `BufferMemory`.

Note that to use BufferMemory with a chat model, use returnMessages is true to keep the history as a series of messages. The messages need to be inserted into the prompt.

Instead of `BufferMemory`, you can use `BufferWindowMemory` which uses a window of size k to only surface the last back-and-forths.

It would be created as follows:

```typescript
const memory = new BufferWindowMemory({ k: 1 });
```

For longer conversations, storing all interactions as is can consume a lot of tokens. The `ConversationSummaryMemory` summarizes the conversation and stores the summmary in memory. The summary can then be injected in the prompt.

There are several other memory types:
- DynamoDB-backed chat memory: instead of memory, store memory presistently across chat sessions in AWS DynamoDB
- Redis-backed chat memory: use Redis to store memory (e.g., Azure Redis Cache)
- VectorStore-backed memory: stores memory in a vector database and retrieve the top-k conversation snippets that are then inserted in the prompt; this does not keep the order of conversation
- Entity memory: remembers given facts about entities in a conversation
- Motorhead memory: Motorhead is a memory server implementation in Rust; see https://github.com/getmetal/motorhead

## chains.ts

This code shows two things:
- LLMChain: a chain of prompt and llm
- SimpleSequentialChain: join multiple single-input/single-output chains into one chain

The example first runs a chain that generates a movie synopsis. The output of that chain is used in a second chain that reviews the movie.

If you have multiple chains with more than one input or output keys, you should use `SequentialChain` instead of `SimpleSequentialChain`.


## Document QA

One of the use cases of using LLMs is asking questions about your own data. LangChain has a couple of chains that allow you to do that:

- StuffDocumentsChain: inject input documents into the prompt as context; ok for small number of docs
- MapReduceChain: has a preprocessing step to select relevant sections of each document until the total number of tokens in less than the maximum allowed by the model; suitable for largers docs
- RefineDocumentsChain: iterates over the input docs one by one, updating an itermediate answer with each iteration; suitable for QA over large number of docs

In `qa.ts`, we start with a simple use of `StuffDocumentsChain` where we create a few short documents in code. In the real world, we would first retrieve the documents but that's for later.

These are the docs:

```typescript
const docs = [
    new Document({ pageContent: "Geert Baeke knows some Kubernetes." }),
    new Document({ pageContent: "John Doe is a front-end developer." }),
  ];
```

Because we set verbose to true, we can see the prompt that LangChain uses under the hood:

```text
"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nGeert Baeke knows some Kubernetes.\n\nJohn Doe is a front-end developer.\n\nQuestion: Who's the developer?\nHelpful Answer:"
```

Above, it's clear the documents were **stuffed** in the prompt. The prompt is defined in the LangChain source code here (Python version): https://github.com/hwchase17/langchain/blob/master/langchain/chains/question_answering/stuff_prompt.py

Sometimes, the prompts seem to provide irrelevant content with your content at the end. They are used to provide examples to the LLM. For example, the prompt for the `loadQAMapReduceChain` contains this (excerpt):

```text
Given the following extracted parts of a long document and a question, create a final answer. \nIf you don't know the answer, just say that you don't know. Don't try to make up an answer.\n\nQUESTION: Which state/country's law governs the interpretation of the contract?\n=========\nContent: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.\n\nContent: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.\n\nContent: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,\n=========\nFINAL ANSWER: This Agreement is governed by English law.
...
...
QUESTION: Who is the developer?\n=========\nGeert Baeke knows some Kubernetes.\n\nJohn Doe is a front-end developer.\n=========\nFINAL ANSWER:"
```

**Note:** for these simple documents, stuffing them is all you need


### Document QA with retrieval

Code is in `qa-retrieve.ts`.

If you have stored documents, you can retrieve them and use them in combination with the QA chains. Instead of simply loading documents from the file system, we will crank it up a notch and use the Pinecone vector database.

Ensure that you have run: `yarn add @pinecone-database/pinecone`

In your .env, have the following:

```bash
PINECONE_API_KEY=YOUR_KEY
PINECONE_ENVIRONMENT=YOUR_ENV
PINECONE_INDEX=YOUR_INDEX
```

In the code, the following happens:
- create a Pinecone client
- initialize the client: requires apiKey and environment
- set the index to use with `client.Index(index)` where index is the name of the index from the PINECONE_INDEX environment variable
- define a `vectorStore`: a vector store is just a database for storing documents and their embeddings; all vector stores in Pinecone use the VectorStore interface to make it easy to add documents to these stores and retrieve documents
- in the code, we use `PineconeStore.fromExistingIndex`. This method requires that you set the type of embeddings that you use (here `OpenAIEmbeddings`). The options define the index to use and sets the key in the database that contains the text of the document. Note that that key is chosen at upload time. It could be `text` but also something else. For my index, it is `text`.
- define a model (text-davince-003 here, not a chat model)
- define the chain, in this case of type `VectorDBQAChain` from an existing llm (fromLLM method). Required parameters are:
    - model
    - vectorStore
    - options: top k search results to return; returnSourceDocuments set to true so we can easily dump the found documents
- call the chain with a query parameter
- log the result and the source documents

The `chain.call` does a lot of stuff under the hood:
- it embeds the query: this does a call to the OpenAI embeddings model (small cost)
- it performs a similarity search via the vector store, in this case Pinecone
- it retrieves the actual text of the documents (the embeddings, which are just vectors, are just used for searching)
- it builds a prompt behind the scenes and stuffs the documents in the prompts 
- it returns the result

If the query is `Can Redis be used as a vector database?` it will embed the query and do the Pinecone search. The first document that is found actually contains the answer:

```text
See Pinecone and OpenAI magic: A guide to finding your long lost blog posts with vectorized search and ChatGPT – baeke.info for more information.\n' +
        'We can replace Pinecone with Redis, a popular open-source, in-memory data store that can be used as a database, cache, and message broker. Redis is well-suited for this task as it can also store vector representations of our blog posts and has the capability to perform vector queries efficiently. \n' +
        'You can easily run Redis with Docker for local development. In addition, Redis is available in Azure, although you will need the Enterprise version. Only Azure Cache for Redis Enterprise supports the RediSearch functionality and that’s what we need here! Note that the Enterprise version is quite costly.\n' +
        'By leveraging Redis for vector storage and querying, we can harness its high performance, flexibility, and reliability in our solution while maintaining the core functionality of quickly querying and retrieving the most relevant blog post content using vectorized search and similarity queries
```

The LLM basically anwsers our question based on the text above (although there are many more documents in the prompt):

```text
Yes, Redis can be used as a vector database. By leveraging Redis for vector storage and querying, we can harness its high performance, flexibility, and reliability in our solution while maintaining the core functionality of quickly querying and retrieving the most relevant blog post content using vectorized search and similarity queries.
```
