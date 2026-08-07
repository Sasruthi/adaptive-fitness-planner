// FILE: frontend/src/components/DayCard.jsx
import { useState } from "react"
import ExerciseCard from "./ExerciseCard"

export default function DayCard({ day, onLogWorkout, doneExercises = new Set() }) {
  const [expanded, setExpanded] = useState(day.day === 1)

  const isRest = day.type === "rest"
  const total  = day.exercises?.length || 0
  const done   = day.exercises?.filter(ex => doneExercises.has(ex.name)).length || 0

  return (
    <div className={`card ${isRest ? "bg-gray-50" : "bg-white"}`}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold flex-shrink-0 ${
            isRest ? "bg-gray-200 text-gray-500" : "bg-brand-600 text-white"
          }`}>
            {isRest ? "😴" : day.day}
          </div>
          <div>
            <p className="font-semibold text-gray-800 text-sm">{day.label}</p>
            <p className="text-xs text-gray-400">
              {isRest ? "Rest & recovery" : `${day.duration_minutes} min · ${total} exercises`}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {!isRest && total > 0 && (
            <span className={`tag ${done === total ? "bg-brand-100 text-brand-700" : "bg-gray-100 text-gray-500"}`}>
              {done}/{total}
            </span>
          )}
          <span className="text-gray-300 text-lg">{expanded ? "▲" : "▼"}</span>
        </div>
      </button>

      {/* Day note */}
      {day.notes && (
        <p className="mt-2 text-xs text-gray-500 italic px-1">{day.notes}</p>
      )}

      {/* Exercises */}
      {expanded && !isRest && day.exercises?.length > 0 && (
        <div className="mt-4 space-y-3">
          {day.exercises.map((ex, i) => (
            <ExerciseCard
              key={i}
              exercise={ex}
              isDone={doneExercises.has(ex.name)}
              onDone={onLogWorkout ? (ex) => onLogWorkout(ex, day) : null}
            />
          ))}
        </div>
      )}

      {expanded && isRest && (
        <div className="mt-4 flex items-center gap-3 text-sm text-gray-500 bg-white rounded-xl p-4 border border-gray-100">
          <span className="text-2xl">🧘</span>
          <p>Light walking, stretching, or yoga. Stay hydrated and let your muscles recover.</p>
        </div>
      )}
    </div>
  )
}
