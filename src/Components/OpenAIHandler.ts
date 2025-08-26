// OpenAI client handler
export default async function OpenaiHandler({ prompt }: { prompt: string }) {
    // console.log("trying to call openai", import.meta.env.VITE_OPENAI_API_KEY)
   
    const openaiApiKey = process.env.OPENAI_API_KEY;
    if (!openaiApiKey) {
      throw new Error('Missing OpenAI API key. Please set VITE_OPENAI_API_KEY in your .env.local file.')
    }
    
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${openaiApiKey}`,
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