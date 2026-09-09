import { useState } from "react"

function App() {
  const [source, setSource] = useState("")
  const [language, setLanguage] = useState("english")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState("")
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [asking, setAsking] = useState(false)
  
  /* first function*/
  const analyzeVideo = async () => {
    if (!source.trim()) {
      setError("Please enter a YouTube URL or file path.")
      return
    }

    setLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source: source,
          language: language,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed.")
      }

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }


  /* second function */

  const askQuestion = async () => {
  if (!question.trim()) {
    return
  }

  setAsking(true)
  setAnswer("")

  try {
    const response = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      throw new Error(data.detail || "Could not get an answer.")
    }

    setAnswer(data.answer)

  } catch (err) {
    setAnswer(`Error: ${err.message}`)
  } finally {
    setAsking(false)
  }
}
  

  return (
    <div className="min-h-screen bg-slate-950 text-white">

      {/* Header */}
      <header className="border-b border-slate-800">
        <div className="mx-auto max-w-7xl px-6 py-5">
          <h1 className="text-2xl font-bold text-cyan-400">
            AI Video Intelligence
          </h1>

          <p className="mt-1 text-sm text-slate-400">
            Turn meetings and videos into actionable insights
          </p>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-7xl px-6 py-10">

        {/* Input section */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

          <h2 className="text-xl font-semibold">
            Analyze a Video
          </h2>

          <p className="mt-2 text-sm text-slate-400">
            Paste a YouTube URL or provide a local audio/video path.
          </p>

          <div className="mt-6 flex flex-col gap-4 md:flex-row">

            <input
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="https://youtube.com/..."
              className="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
            />

            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="rounded-xl border border-slate-700 bg-slate-950 px-4 py-3"
            >
              <option value="english">English</option>
              <option value="hinglish">Hinglish</option>
            </select>

            <button
              onClick={analyzeVideo}
              disabled={loading}
              className="rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Analyzing..." : "Analyze Video"}
            </button>

          </div>

          {error && (
            <p className="mt-4 rounded-lg bg-red-950 p-3 text-sm text-red-300">
              {error}
            </p>
          )}

        </section>

        {/* Results */}
        {result && (
          <div className="mt-8 space-y-6">

            {/* Title */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <p className="text-sm text-cyan-400">
                Meeting Title
              </p>

              <h2 className="mt-2 text-3xl font-bold">
                {result.title}
              </h2>
            </section>

            {/* Summary */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h2 className="text-xl font-semibold">
                Summary
              </h2>

              <div className="mt-4 whitespace-pre-wrap text-slate-300">
                {result.summary}
              </div>
            </section>

            {/* Three cards */}
            <div className="grid gap-6 md:grid-cols-3">

              <ResultCard
                title="Action Items"
                content={result.action_items}
              />

              <ResultCard
                title="Key Decisions"
                content={result.key_decisions}
              />

              <ResultCard
                title="Open Questions"
                content={result.open_questions}
              />

            </div>

            {/* Transcript */}
            <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

              <h2 className="text-xl font-semibold">
                Transcript
              </h2>

              <div className="mt-4 max-h-96 overflow-y-auto whitespace-pre-wrap rounded-xl bg-slate-950 p-5 text-sm leading-7 text-slate-300">
                {result.transcript}
             
              </div>



            </section>

            {/* RAG Chat */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

            <h2 className="text-xl font-semibold">
              Ask About This Video
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Ask questions based on the analyzed transcript.
            </p>

            <div className="mt-5 flex gap-3">

              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    askQuestion()
                  }
                }}
                placeholder="What did the speaker say about...?"
                className="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
              />

              <button
                onClick={askQuestion}
                disabled={asking}
                className="rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {asking ? "Asking..." : "Ask"}
              </button>

            </div>

            {answer && (
              <div className="mt-6 rounded-xl border border-slate-800 bg-slate-950 p-5">

                <p className="text-sm font-semibold text-cyan-400">
                  AI Assistant
                </p>

                <p className="mt-3 whitespace-pre-wrap leading-7 text-slate-300">
                  {answer}
                </p>

              </div>
            )}

          </section>
          </div>
        )}

      </main>
    </div>
  )
}


function ResultCard({ title, content }) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">

      <h2 className="text-lg font-semibold">
        {title}
      </h2>

      <div className="mt-4 whitespace-pre-wrap text-sm leading-6 text-slate-400">
        {content}
      </div>

    </section>
  )
}


export default App