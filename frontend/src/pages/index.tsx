import { useState } from 'react';
import { askQuestion } from '../lib/api';
export default function Home() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const handleAsk = async () => {
    if (!query.trim()) return;
    setLoading(true);
    const res = await askQuestion(query);
    setAnswer(res);
    setLoading(false);
  };
  return (
    <main className="p-8 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">社内RAG QAボット</h1>
      <textarea className="w-full p-2 border rounded mb-4" rows={4} value={query} onChange={e => setQuery(e.target.value)} placeholder="質問を入力してください" />
      <button onClick={handleAsk} className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50" disabled={loading}>{loading ? '送信中…' : '質問する'}</button>
      {answer && (<div className="mt-6 p-4 border rounded bg-gray-50"><h2 className="font-semibold mb-2">回答</h2><p>{answer}</p></div>)}
    </main>
  );
}
