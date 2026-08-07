// FILE: frontend/src/pages/ChatPage.jsx
import { useEffect, useRef, useState } from "react"
import { startConversation, sendMessage, generatePlan, savePlan } from "../api/client"
import BotAvatar from "../components/BotAvatar"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

function mediaSrc(url) {
  if (!url) return null
  if (url.startsWith("/static/")) return `${API}${url}`
  if (url.startsWith("http")) return url
  return null
}

function ExerciseMediaStrip({ exercises }) {
  if (!exercises?.length) return null
  return (
    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
      {exercises.map((ex, idx) => (
        <ChatExerciseCard key={`${ex.name}-${idx}`} exercise={ex} />
      ))}
    </div>
  )
}

function ChatExerciseCard({ exercise }) {
  const [open, setOpen] = useState(false)
  const [failedGif, setFailedGif] = useState(false)
  const [failedImg, setFailedImg] = useState(false)
  const gif = !failedGif ? mediaSrc(exercise.gif_url) : null
  const img = !failedImg ? mediaSrc(exercise.image_url) : null
  const src = gif || img
  const instructions = (exercise.instructions || exercise.description || "").trim()

  return (
    <div className="rounded-xl overflow-hidden border border-gray-100 bg-white">
      {src ? (
        <img
          src={src}
          alt={exercise.name}
          className="w-full h-28 object-cover bg-gray-50"
          onError={() => {
            if (gif && !failedGif) setFailedGif(true)
            else setFailedImg(true)
          }}
        />
      ) : (
        <div className="w-full h-20 bg-gray-50 flex items-center justify-center text-2xl">💪</div>
      )}
      <div className="px-2.5 py-2">
        <p className="text-xs font-semibold text-gray-800 capitalize leading-tight">{exercise.name}</p>
        <p className="text-[10px] text-gray-500 capitalize mt-0.5">
          {[exercise.target_muscle || exercise.body_part, exercise.equipment].filter(Boolean).join(" · ")}
        </p>
        {instructions ? (
          <>
            <button
              type="button"
              onClick={() => setOpen(v => !v)}
              className="mt-1.5 text-[11px] font-medium text-brand-600 hover:text-brand-700"
            >
              {open ? "Hide how-to ↑" : "How to do it ↓"}
            </button>
            {open && (
              <p className="mt-1.5 text-[11px] text-gray-600 leading-relaxed border-t border-gray-50 pt-1.5">
                {instructions}
              </p>
            )}
          </>
        ) : (
          <p className="mt-1 text-[10px] text-gray-400 italic">No written steps in catalog for this move.</p>
        )}
      </div>
    </div>
  )
}

export default function ChatPage({
  onPlanReady, userEmail, setUserEmail,
  messages, setMessages, threadId, setThreadId,
  chatStage, setChatStage, chatDone,
}) {
  const [input,       setInput]       = useState("")
  const [loading,     setLoading]     = useState(false)
  const [generating,  setGenerating]  = useState(false)
  const [emailPrompt, setEmailPrompt] = useState(false)
  const [nameVal,     setNameVal]     = useState("")
  const [emailVal,    setEmailVal]    = useState("")
  const [pendingData, setPendingData] = useState(null)
  const bottomRef = useRef(null)
  const initStarted = useRef(false)

  useEffect(() => {
    if (initStarted.current || messages.length > 0) return
    initStarted.current = true
    init()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading, generating])

  async function init() {
    setLoading(true)
    try {
      const { data } = await startConversation()
      setThreadId(data.thread_id)
      setChatStage(data.stage)
      setMessages([{ role: "assistant", text: data.message }])
    } catch {
      setMessages([{
        role: "assistant",
        text: "Couldn't reach the server. Start the backend with: cd backend && uvicorn main:app --reload --port 8000 — then refresh this page.",
      }])
      initStarted.current = false
    } finally {
      setLoading(false)
    }
  }

  function addBot(text, exercises = null) {
    setMessages(prev => [...prev, { role: "assistant", text, exercises: exercises || undefined }])
  }
  function addUser(text) { setMessages(prev => [...prev, { role: "user", text }]) }

  async function handleSend() {
    if (!input.trim() || loading || !threadId) return
    const msg = input.trim()
    setInput("")
    addUser(msg)
    setLoading(true)
    try {
      const { data } = await sendMessage(threadId, msg)
      setChatStage(data.stage)
      addBot(data.message, data.exercises)
      // Agentic: keep chatting after profile is complete — don't force the email form.
      // Offer save only when a plan actually arrives.
      if (data.plan) {
        setPendingData(data)
        setTimeout(() => onPlanReady(data.plan, null, userEmail), 400)
      }
    } catch {
      addBot("Something went wrong — please try again.")
    } finally {
      setLoading(false)
    }
  }

  async function handleGenerate() {
    if (!emailVal.trim() || !nameVal.trim()) return
    const email = emailVal.trim()
    const name  = nameVal.trim()
    setUserEmail(email)
    setEmailPrompt(false)
    setGenerating(true)
    addBot(`Saving your plan for ${name}…`)
    try {
      let plan = pendingData?.plan
      if (!plan) {
        const { data: planData } = await generatePlan({
          profile:     pendingData.profile,
          sql_filters: pendingData.sql_filters,
          rag_filters: pendingData.rag_filters,
          user_email:  email,
          user_name:   name,
        })
        plan = planData.plan
      }
      const { data: saved } = await savePlan({
        user_email: email,
        user_name:  name,
        goal:       pendingData.profile?.goal || "general_fitness",
        plan,
      })
      if (!saved?.success && !saved?.plan_id) {
        addBot(`Save failed: ${saved?.message || saved?.detail || saved?.error || "database rejected the plan."}`)
        return
      }
      const emailNote = saved.email_sent
        ? " We've also sent a copy to your inbox."
        : saved.email_note
          ? ` (${saved.email_note})`
          : ""
      const day0 = (plan?.week_plan || [])[0]
      const preview = (day0?.exercises || [])
        .filter(ex => ex.gif_url || ex.image_url)
        .slice(0, 6)
      addBot(`Saved — open the Plan tab anytime.${emailNote}`, preview)
      setTimeout(() => onPlanReady(plan, saved.plan_id, email), 600)
    } catch (e) {
      const detail = e?.response?.data?.detail || e.message || "Unknown error"
      addBot(`Save failed: ${detail}`)
    } finally {
      setGenerating(false)
    }
  }

  function renderText(text) {
    return text
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br/>")
  }

  const busy = loading || generating

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            {m.role === "assistant" && (
              <div className="mr-2 mt-0.5">
                <BotAvatar />
              </div>
            )}
            <div className={`${m.role === "user" ? "chat-bubble-user" : "chat-bubble-bot"} max-w-[85%]`}>
              <p className="text-sm leading-relaxed"
                 dangerouslySetInnerHTML={{ __html: renderText(m.text) }} />
              {m.role === "assistant" && m.exercises?.length > 0 && (
                <ExerciseMediaStrip exercises={m.exercises} />
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="mr-2"><BotAvatar /></div>
            <div className="chat-bubble-bot">
              <div className="flex gap-1 items-center h-4">
                {[0, 150, 300].map(d => (
                  <span key={d} className="w-2 h-2 bg-brand-400 rounded-full animate-bounce"
                        style={{ animationDelay: `${d}ms` }} />
                ))}
              </div>
            </div>
          </div>
        )}

        {generating && (
          <div className="flex justify-center py-6">
            <div className="flex flex-col items-center gap-2">
              <div className="w-8 h-8 border-2 border-brand-600 border-t-transparent rounded-full animate-spin" />
              <p className="text-xs text-gray-500">Building your plan…</p>
            </div>
          </div>
        )}

        {chatDone && !emailPrompt && !generating && (
          <div className="text-center py-2 space-y-1">
            <p className="text-xs text-gray-400">Plan ready — open the Plan tab anytime</p>
            <p className="text-xs text-brand-600">Keep chatting to ask questions, swap exercises, or refine your plan</p>
            <button
              type="button"
              className="text-xs font-medium text-brand-700 underline"
              onClick={() => setEmailPrompt(true)}
            >
              Save &amp; email this plan
            </button>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {emailPrompt && (
        <div className="mx-4 mb-3 card border-brand-200 bg-brand-50">
          <p className="text-sm font-semibold text-brand-700 mb-1">
            Save your plan
          </p>
          <p className="text-xs text-gray-500 mb-3">
            Optional — enter details to save and email. Or keep chatting without saving.
          </p>
          <input
            className="w-full border border-gray-200 rounded-xl px-4 py-2.5 text-sm mb-2
                       focus:outline-none focus:ring-2 focus:ring-brand-300 bg-white"
            placeholder="Your name"
            value={nameVal}
            onChange={e => setNameVal(e.target.value)}
          />
          <input
            className="w-full border border-gray-200 rounded-xl px-4 py-2.5 text-sm mb-3
                       focus:outline-none focus:ring-2 focus:ring-brand-300 bg-white"
            placeholder="Email address"
            type="email"
            value={emailVal}
            onChange={e => setEmailVal(e.target.value)}
          />
          <div className="flex gap-2">
            <button
              onClick={() => setEmailPrompt(false)}
              className="flex-1 py-2.5 rounded-xl text-sm font-medium border border-gray-200 text-gray-600 bg-white"
            >
              Keep chatting
            </button>
            <button
              onClick={handleGenerate}
              disabled={!emailVal.trim() || !nameVal.trim()}
              className="btn-primary flex-1"
            >
              Save &amp; email
            </button>
          </div>
        </div>
      )}

      {/* Always keep the agent input available — plan ready must not end the conversation */}
      {!emailPrompt && (
        <div className="px-4 pb-4 pt-2 border-t border-gray-100">
          <div className="flex gap-2">
            <input
              className="flex-1 border border-gray-200 rounded-2xl px-4 py-3 text-sm
                         focus:outline-none focus:ring-2 focus:ring-brand-300"
              placeholder={chatDone
                ? "Ask about exercises, diet, swaps, reminders…"
                : "Type your message…"}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleSend()}
              disabled={busy}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || busy}
              className="btn-primary px-4"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
