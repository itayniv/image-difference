export default async function handler(req: any, res: any) {
    if (req.method !== 'POST') return res.status(405).end();
  
    const { prompt } = req.body ?? {};
    const openaiApiKey = process.env.OPENAI_API_KEY;
    if (!openaiApiKey) {
      return res.status(500).json({
        error: 'Missing OPENAI_API_KEY (set it in Vercel → Project → Settings → Environment Variables).',
      });
    }
  
    try {
      const r = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${openaiApiKey}`,
        },
        body: JSON.stringify({
          model: 'gpt-4.1-nano', // use a valid model name
          messages: [{ role: 'user', content: prompt }],
        }),
      });
  
      const data = await r.json();
      return res.status(r.ok ? 200 : r.status).json(data);
    } catch (err: any) {
      return res.status(500).json({ error: err?.message ?? 'Unknown error' });
    }
  }