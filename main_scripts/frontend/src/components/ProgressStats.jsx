// FILE: frontend/src/components/ProgressStats.jsx
export default function ProgressStats({ stats }) {
  if (!stats) return null

  const days   = Object.entries(stats.daily_breakdown || {})
  const streak = stats.current_streak || 0

  return (
    <div className="space-y-4">
      {/* Top stats row */}
      <div className="grid grid-cols-3 gap-3">
        <StatBox icon="🔥" label="Streak" value={`${streak}d`} highlight={streak > 0} />
        <StatBox icon="📅" label="Sessions" value={stats.total_sessions} />
        <StatBox icon="💪" label="Exercises" value={stats.total_exercises} />
      </div>

      {/* Completion rate */}
      <div className="card">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm font-semibold text-gray-700">
            This Week ({stats.period_days} days)
          </p>
          <span className="text-brand-600 font-bold">{stats.completion_rate}</span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-500"
            style={{ width: stats.completion_rate }}
          />
        </div>
        <p className="mt-3 text-sm text-gray-500 italic">{stats.motivation}</p>
      </div>

      {/* Daily breakdown */}
      {days.length > 0 && (
        <div className="card">
          <p className="text-sm font-semibold text-gray-700 mb-3">Daily Activity</p>
          <div className="space-y-2">
            {days.map(([date, exercises]) => (
              <div key={date} className="flex items-start gap-3">
                <div className="flex-shrink-0 w-16 text-xs text-gray-400 pt-0.5">{date}</div>
                <div className="flex flex-wrap gap-1">
                  {exercises.map((ex, i) => (
                    <span key={i} className="tag bg-brand-100 text-brand-700 capitalize text-xs">
                      {ex.length > 20 ? ex.slice(0, 20) + "…" : ex}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function StatBox({ icon, label, value, highlight }) {
  return (
    <div className={`card text-center ${highlight ? "bg-brand-50 border-brand-200" : ""}`}>
      <div className="text-2xl">{icon}</div>
      <div className={`text-xl font-bold mt-1 ${highlight ? "text-brand-700" : "text-gray-800"}`}>
        {value}
      </div>
      <div className="text-xs text-gray-400">{label}</div>
    </div>
  )
}
