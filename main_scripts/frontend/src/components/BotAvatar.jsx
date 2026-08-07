/** Small dumbbell icon for assistant chat bubbles and header */
export default function BotAvatar({ className = "w-7 h-7" }) {
  return (
    <div
      className={`${className} rounded-full bg-brand-600 flex items-center justify-center
                  text-white flex-shrink-0`}
      aria-hidden
    >
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5">
        <path d="M6.5 10.5h-2a1.5 1.5 0 0 0 0 3h2v-3zm11 0v3h2a1.5 1.5 0 0 0 0-3h-2zm-8 0v3h7v-3h-7zm-5-3a3.5 3.5 0 0 0 0 7h1.5v-7H4.5zm15 0h-1.5v7H18a3.5 3.5 0 0 0 0-7zM9 7.5h6v9H9v-9z" />
      </svg>
    </div>
  )
}
