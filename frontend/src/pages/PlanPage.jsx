// FILE: frontend/src/pages/PlanPage.jsx
import { useState } from "react"
import DayCard  from "../components/DayCard"
import DietPlan from "../components/DietPlan"
import { logWorkout } from "../api/client"

const TABS = [
  { id:"week",   label:"📅 Workout" },
  { id:"diet",   label:"🥗 Nutrition" },
  { id:"safety", label:"🛡️ Safety & Sources" },
]

export default function PlanPage({ plan, planId, userEmail }) {
  const dietOnly = Boolean(
    plan?.plan_mode === "diet_only" ||
    plan?.stats?.diet_only ||
    plan?.profile_summary?.plan_mode === "diet_only"
  )
  const yogaOnly = Boolean(
    plan?.plan_mode === "yoga_only" ||
    plan?.profile_summary?.plan_mode === "yoga_only"
  )
  const [tab,           setTab]           = useState(dietOnly ? "diet" : "week")
  const [doneExercises, setDoneExercises] = useState(new Set())
  const [toast,         setToast]         = useState(null)

  if (!plan) return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-8 gap-4">
      <span className="text-5xl">💬</span>
      <p className="text-gray-500 text-sm">Complete the chat to generate your personalised plan.</p>
    </div>
  )

  const { week_plan=[], diet_plan, safety_notes=[], citations=[], weekly_tips=[], profile_summary={} } = plan
  const tabs = dietOnly
    ? TABS.filter(t => t.id !== "week")
    : TABS.map(t => (
        t.id === "week" && yogaOnly ? { ...t, label: "🧘 Yoga" } : t
      ))

  async function handleLogWorkout(exercise, day) {
    if (doneExercises.has(exercise.name)) return
    setDoneExercises(prev => new Set([...prev, exercise.name]))
    if (userEmail && planId) {
      try {
        await logWorkout({ user_email:userEmail, plan_id:planId,
          exercise_names:[exercise.name], notes:`Logged: ${day.label}` })
        setToast(`✅ "${exercise.name}" logged!`)
        setTimeout(() => setToast(null), 2500)
      } catch {}
    }
  }

  const totalExercises = week_plan.reduce((a,d)=>a+d.exercises?.length||0,0)
  const doneCount      = doneExercises.size

  return (
    <div className="flex flex-col h-full">
      {/* Profile summary bar */}
      <div className="mx-4 mt-3 card py-3 px-4 bg-brand-50 border-brand-200">
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {profile_summary.goal && <Chip label="Goal" value={profile_summary.goal.replace(/_/g," ")} />}
          {profile_summary.target_body_parts?.length>0 && <Chip label="Focus" value={profile_summary.target_body_parts.join(", ")} />}
          {profile_summary.fitness_level && <Chip label="Level" value={profile_summary.fitness_level} />}
          {profile_summary.time_per_day_minutes && <Chip label="Time" value={`${profile_summary.time_per_day_minutes} min/day`} />}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 px-4 mt-3 border-b border-gray-100">
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            className={`px-3 py-2 text-xs font-medium rounded-t-lg border-b-2 transition-all ${
              tab===t.id ? "border-brand-600 text-brand-700 bg-brand-50"
                         : "border-transparent text-gray-500 hover:text-gray-700"}`}>
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {toast && (
          <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 bg-brand-700 text-white text-sm px-4 py-2 rounded-xl shadow-lg">
            {toast}
          </div>
        )}

        {/* Week plan */}
        {tab==="week" && (
          <>
            {doneCount > 0 && (
              <div className="card bg-brand-50 border-brand-200 flex items-center justify-between py-3">
                <span className="text-sm text-brand-700">Progress</span>
                <span className="text-sm font-bold text-brand-700">{doneCount}/{totalExercises} exercises done</span>
              </div>
            )}
            {week_plan.map((day,i)=>(
              <DayCard key={i} day={day} onLogWorkout={handleLogWorkout} doneExercises={doneExercises} />
            ))}
            {weekly_tips.length>0 && (
              <div className="card bg-yellow-50 border-yellow-100">
                <p className="font-semibold text-sm text-yellow-700 mb-2">💡 Weekly Tips</p>
                <ul className="space-y-1.5">
                  {weekly_tips.map((t,i)=>(
                    <li key={i} className="text-sm text-gray-600 flex gap-1.5">
                      <span className="text-yellow-500 flex-shrink-0">•</span>{t}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {/* Nutrition */}
        {tab==="diet" && <DietPlan diet={diet_plan} />}

        {/* Safety + Sources (NO tier labels) */}
        {tab==="safety" && (
          <div className="space-y-3">
            {safety_notes.length>0 && (
              <>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide px-1">Safety Notes</p>
                {safety_notes.map((note,i)=>{
                  const flag = typeof note === "string" ? "note" : (note.flag || "note")
                  const text = typeof note === "string" ? note : (note.note || note.text || "")
                  const citation = typeof note === "string" ? null : note.citation
                  return (
                  <div key={i} className="card border-l-4 border-amber-400">
                    <p className="text-xs font-semibold text-amber-700 mb-1 capitalize">
                      {String(flag).replace(/_/g," ")}
                    </p>
                    <p className="text-sm text-gray-700">{text}</p>
                    {citation && (
                      <p className="text-xs text-gray-400 mt-2 italic">Source: {citation}</p>
                    )}
                  </div>
                  )
                })}
              </>
            )}

            {citations.length>0 && (
              <>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide px-1 mt-4">
                  Guidelines Used
                </p>
                <div className="space-y-2">
                  {citations.map((c,i)=>(
                    <div key={i} className="card py-3">
                      <p className="text-sm font-medium text-gray-700">{c.source}</p>
                      {c.page && <p className="text-xs text-gray-400 mt-0.5">Page {c.page}</p>}
                      <p className="text-xs text-gray-500 mt-1">{c.used_for}</p>
                    </div>
                  ))}
                </div>
              </>
            )}

            {safety_notes.length===0 && citations.length===0 && (
              <p className="text-center text-gray-400 text-sm py-8">No safety flags for this profile.</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function Chip({ label, value }) {
  if (!value) return null
  return (
    <div className="flex items-center gap-1">
      <span className="text-xs text-gray-400">{label}:</span>
      <span className="text-xs font-medium text-brand-700 capitalize">{value}</span>
    </div>
  )
}
