// OpenAI client handler
export default async function OpenaiHandler({ prompt }: { prompt: string }) {
    console.log("trying to call openai")
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`,
        },
        body: JSON.stringify({
          model: "gpt-4.1-nano",
          messages: [{ role: "user", content: prompt }],
        }),
      });
  
      const data = await response.json();
      return data;
    } catch (err) {
      throw new Error(err instanceof Error ? err.message : 'Unknown error');
    }
  }