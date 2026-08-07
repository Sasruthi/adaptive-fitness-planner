// FILE: frontend/src/App.jsx
// Chat state lifted here so it persists across tab switches
import { useState, useRef } from "react"
import ChatPage     from "./pages/ChatPage"
import PlanPage     from "./pages/PlanPage"
import ProgressPage from "./pages/ProgressPage"
import BotAvatar    from "./components/BotAvatar"

const NAV = [
  { id:"chat",     icon:"💬", label:"Chat"     },
  { id:"plan",     icon:"📅", label:"Plan"     },
  { id:"progress", icon:"📊", label:"Progress" },
]

export default function App() {
  const [page,      setPage]      = useState("chat")
  const [plan,      setPlan]      = useState(null)
  const [planId,    setPlanId]    = useState(null)
  const [userEmail, setUserEmail] = useState(null)

  // Chat state lifted here so it survives tab switches
  const [messages,   setMessages]   = useState([])
  const [threadId,   setThreadId]   = useState(null)
  const [chatStage,  setChatStage]  = useState("idle")
  const [chatDone,   setChatDone]   = useState(false)

  function handlePlanReady(planData, savedPlanId, email) {
    setPlan(planData)
    if (savedPlanId != null) setPlanId(savedPlanId)
    if (email) setUserEmail(email)
    setChatDone(true)
    // Stay on Chat so the agent conversation can continue; Plan tab is one click away
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col max-w-lg mx-auto relative">
      <header className="bg-white border-b border-gray-100 px-4 py-3 flex items-center gap-3 sticky top-0 z-10 shadow-sm">
        <BotAvatar className="w-9 h-9" />
        <div>
          <h1 className="font-bold text-gray-800 text-sm leading-tight">Adaptive Fitness Planner</h1>
          <p className="text-xs text-gray-500">Your personal workout & nutrition coach</p>
        </div>
        {plan && (
          <div className="ml-auto flex items-center gap-1.5">
            <span className="w-2 h-2 bg-brand-500 rounded-full animate-pulse" />
            <span className="text-xs text-brand-600 font-medium">Plan active</span>
          </div>
        )}
      </header>

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Always rendered but hidden — preserves chat state */}
        <div className={page === "chat" ? "flex flex-col flex-1 overflow-hidden" : "hidden"}>
          <ChatPage
            onPlanReady={handlePlanReady}
            userEmail={userEmail}
            setUserEmail={setUserEmail}
            messages={messages}
            setMessages={setMessages}
            threadId={threadId}
            setThreadId={setThreadId}
            chatStage={chatStage}
            setChatStage={setChatStage}
            chatDone={chatDone}
          />
        </div>
        {page === "plan"     && <PlanPage plan={plan} planId={planId} userEmail={userEmail} />}
        {page === "progress" && <ProgressPage userEmail={userEmail} plan={plan} />}
      </main>

      <nav className="bg-white border-t border-gray-100 flex sticky bottom-0 z-10">
        {NAV.map(n => (
          <button key={n.id} onClick={() => setPage(n.id)}
            className={`flex-1 flex flex-col items-center py-3 gap-0.5 transition-all relative ${page===n.id ? "text-brand-600" : "text-gray-400 hover:text-gray-600"}`}>
            <span className="text-xl">{n.icon}</span>
            <span className="text-[10px] font-medium">{n.label}</span>
            {page===n.id && <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-6 h-0.5 bg-brand-600 rounded-full" />}
          </button>
        ))}
      </nav>
    </div>
  )
}
