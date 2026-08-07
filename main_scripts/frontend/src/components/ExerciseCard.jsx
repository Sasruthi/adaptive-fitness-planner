// FILE: frontend/src/components/ExerciseCard.jsx
import { useState } from "react"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

function getImageSrc(url) {
  if (!url) return null
  if (url.startsWith("/static/")) return `${API}${url}`
  if (url.includes("raw.githubusercontent.com")) return url
  // External http(s) — load directly (no gif-proxy route in this API)
  if (url.startsWith("http")) return url
  return null
}

function ExerciseMedia({ gif_url, image_url, video_url, name }) {
  const [failedGif, setFailedGif] = useState(false)
  const [failedImg, setFailedImg] = useState(false)

  if (video_url && video_url.includes("youtube.com")) {
    const videoId = video_url.split("v=")[1]?.split("&")[0]
    if (videoId) {
      return (
        <div className="flex-shrink-0 w-20 h-20 rounded-xl overflow-hidden bg-gray-100">
          <iframe
            width="100%"
            height="100%"
            src={`https://www.youtube.com/embed/${videoId}`}
            title={name}
            frameBorder="0"
            allowFullScreen
            className="w-full h-full"
          />
        </div>
      )
    }
  }

  const gifSrc = !failedGif ? getImageSrc(gif_url) : null
  const imgSrc = !failedImg ? getImageSrc(image_url) : null
  const src = gifSrc || imgSrc
  if (src) {
    return (
      <div className="flex-shrink-0 w-20 h-20 rounded-xl overflow-hidden bg-gray-100 flex items-center justify-center">
        <img
          src={src}
          alt={name}
          className="w-full h-full object-cover"
          onError={() => {
            if (gifSrc && !failedGif) setFailedGif(true)
            else setFailedImg(true)
          }}
        />
      </div>
    )
  }

  return (
    <div className="flex-shrink-0 w-20 h-20 rounded-xl overflow-hidden bg-gray-100 flex items-center justify-center">
      <span className="text-2xl">💪</span>
    </div>
  )
}

export default function ExerciseCard({ exercise, onDone, isDone = false }) {
  const [showInstr, setShowInstr] = useState(false)
  const hasMedia = Boolean(exercise.video_url || exercise.gif_url || exercise.image_url)

  return (
    <div className={`card transition-all border-2 ${isDone ? "border-brand-500 bg-brand-50" : "border-gray-100"}`}>
      <div className="flex gap-3">

        {hasMedia ? (
          <ExerciseMedia
            gif_url={exercise.gif_url}
            image_url={exercise.image_url}
            video_url={exercise.video_url}
            name={exercise.name}
          />
        ) : (
          <div className="flex-shrink-0 w-20 h-20 rounded-xl overflow-hidden bg-gray-100 flex items-center justify-center">
            <div className="flex flex-col items-center gap-1 px-1">
              <span className="text-2xl">💪</span>
              <span className="text-[9px] text-gray-400 text-center leading-tight capitalize">
                {(exercise.target_muscle || exercise.body_part || "").split(",")[0]}
              </span>
            </div>
          </div>
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <h4 className="font-semibold text-gray-800 text-sm leading-tight capitalize">
              {exercise.name}
            </h4>
            {isDone && <span className="text-brand-500 text-base flex-shrink-0">✓</span>}
          </div>

          <div className="flex flex-wrap gap-1.5 mt-1.5">
            <span className="tag bg-brand-100 text-brand-700">
              {exercise.sets} × {exercise.reps}
            </span>
            {exercise.rest_seconds && (
              <span className="tag bg-gray-100 text-gray-500">{exercise.rest_seconds}s rest</span>
            )}
            {exercise.difficulty && (
              <span className="tag bg-blue-50 text-blue-600 capitalize">{exercise.difficulty}</span>
            )}
          </div>

          {exercise.target_muscle && (
            <p className="text-xs text-gray-400 mt-1 capitalize">🎯 {exercise.target_muscle}</p>
          )}

          {exercise.modification && (
            <div className="mt-1.5 text-xs bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1 text-amber-700">
              ⚠️ {exercise.modification}
            </div>
          )}

          {exercise.instructions && (
            <button
              onClick={() => setShowInstr(!showInstr)}
              className="mt-1.5 text-xs text-brand-600 hover:text-brand-700 font-medium"
            >
              {showInstr ? "Hide ↑" : "How to do it ↓"}
            </button>
          )}
        </div>
      </div>

      {showInstr && exercise.instructions && (
        <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-600 leading-relaxed">
          {exercise.instructions}
        </div>
      )}

      {onDone && (
        <button
          onClick={() => onDone(exercise)}
          disabled={isDone}
          className={`mt-3 w-full py-2 rounded-xl text-sm font-medium transition-all ${
            isDone
              ? "bg-brand-100 text-brand-700 cursor-default"
              : "bg-gray-50 hover:bg-brand-50 text-gray-600 hover:text-brand-600 border border-gray-100"
          }`}
        >
          {isDone ? "✓ Completed" : "Mark as done"}
        </button>
      )}
    </div>
  )
}
