# Langchain with TypeScript

Instructions below use yarn. Use `brew install yarn` if you have HomeBrew installed.

```bash
yarn global add ts-node typescript '@types/node'
yarn add langchain
yarn add openai
yarn install dotenv
```

To run a .ts file:

```bash
ts-node intro.ts
```

Note: wrap code in the IIFE (Immediately-invoked Function Expression) to easily allow await

