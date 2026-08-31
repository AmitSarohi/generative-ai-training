import ollama from "ollama";

const messages = [
  {
    role: "system",
    content: "You are a senior AI trainer. Explain with examples.",
  },
];

while (true) {
  const readline = await import("node:readline/promises");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const msg = await rl.question("You: ");

  if (msg === "quit") {
    rl.close();
    break;
  }

  messages.push({
    role: "user",
    content: msg,
  });

  const response = await ollama.chat({
    model: "llama3.2",
    messages,
    stream: true,
    options: {
      temperature: 0.2, // Lower value = more deterministic/focused (0.0 to 1.0+)
      num_predict: 200, // Limits the max number of tokens/words in the response length
    },
  });

  // Loop through each chunk as it arrives from local Ollama
  for await (const chunk of response) {
    process.stdout.write(chunk.message.content);
  }

  //   const answer = response.message.content;

  //   console.log("\nAssistant:", answer);

  //   messages.push({
  //     role: "assistant",
  //     content: answer,
  //   });

  rl.close();
}
