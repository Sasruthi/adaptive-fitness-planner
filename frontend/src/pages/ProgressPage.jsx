// FILE: frontend/src/pages/ProgressPage.jsx
import { useState, useEffect } from "react"
import { getProgress, sendReminder } from "../api/client"
import ProgressStats from "../components/ProgressStats"

export default function ProgressPage({ userEmail, plan }) {
  const [stats,     setStats]     = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [days,      setDays]      = useState(7)
  const [reminder,  setReminder]  = useState({ sending: false, sent: false, error: null })

  useEffect(() => {
    if (userEmail) fetchStats()
  }, [userEmail, days])

  async function fetchStats() {
    setLoading(true)
    try {
      const { data } = await getProgress(userEmail, days)
      setStats(data)
    } catch {
      setStats(null)
    } finally {
      setLoading(false)
    }
  }

  async function handleReminder() {
    if (!plan || !userEmail) return
    setReminder({ sending: true, sent: false, error: null })
    const todayPlan = plan.week_plan?.find(d => d.type === "workout")
    try {
      const { data } = await sendReminder({
        to_email:       userEmail,
        user_name:      userEmail.split("@")[0],
        plan_day_label: todayPlan?.label || "Today's Workout",
        exercises:      (todayPlan?.exercises || []).map(e => e.name).slice(0, 5),
        reminder_type:  "workout",
      })
      if (data.success) {
        setReminder({ sending: false, sent: true, error: null })
      } else {
        setReminder({
          sending: false,
          sent: false,
          error: data.message || data.error || "Could not send email — check SMTP settings.",
        })
      }
    } catch {
      setReminder({ sending: false, sent: false, error: "Failed to send. Is the backend running?" })
    }
  }

  if (!userEmail) return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-8 gap-3">
      <span className="text-5xl">📊</span>
      <p className="text-gray-500 text-sm">Complete the chat and generate a plan to see your progress here.</p>
    </div>
  )

  return (
    <div className="flex flex-col h-full overflow-y-auto px-4 py-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-bold text-gray-800">Your Progress</h2>
          <p className="text-xs text-gray-400">{userEmail}</p>
        </div>
        <div className="flex gap-1">
          {[7, 14, 30].map(d => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className={`px-3 py-1.5 text-xs rounded-lg font-medium transition-all ${
                days === d ? "bg-brand-600 text-white" : "bg-gray-100 text-gray-600"
              }`}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : stats ? (
        <ProgressStats stats={stats} />
      ) : (
        <p className="text-center text-gray-400 text-sm py-8">No workout data yet. Start logging!</p>
      )}

      {/* Send Reminder */}
      <div className="card">
        <p className="font-semibold text-sm text-gray-700 mb-3">📧 Send Workout Reminder</p>
        <p className="text-xs text-gray-400 mb-3">
          Get a reminder email with today's exercises sent to {userEmail}.
        </p>
        <button
          onClick={handleReminder}
          disabled={reminder.sending || reminder.sent}
          className="btn-primary w-full"
        >
          {reminder.sending ? "Sending…" : reminder.sent ? "✓ Reminder Sent!" : "Send Reminder"}
        </button>
        {reminder.error && (
          <p className="mt-2 text-xs text-red-500">{reminder.error}</p>
        )}
        {reminder.sent && (
          <p className="mt-2 text-xs text-brand-600">
            Sent — check your inbox (and spam folder).
          </p>
        )}
      </div>

      {/* Refresh */}
      <button onClick={fetchStats} className="btn-secondary w-full text-sm">
        ↻ Refresh Stats
      </button>
    </div>
  )
}
