// FILE: frontend/src/api/client.js
import axios from "axios"

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
})

export const startConversation = ()            => api.post("/api/conversation/start")
export const sendMessage = (tid, msg)          => api.post("/api/conversation/message", { thread_id: tid, user_message: msg })
export const generatePlan = (body)             => api.post("/api/plan/generate", body)
export const savePlan = (body)                 => api.post("/api/plan/save", body)
export const getUserPlan = (email)             => api.get(`/api/plan/${encodeURIComponent(email)}`)
export const calcCalories = (body)             => api.post("/api/plan/calories", body)
export const logWorkout = (body)               => api.post("/api/workout/log", body)
export const getProgress = (email, days = 7)   => api.get(`/api/workout/progress/${encodeURIComponent(email)}?days=${days}`)
export const sendReminder = (body)             => api.post("/api/workout/reminder", body)
export const searchExercises = (body)          => api.post("/api/exercises/search", body)
export const getTaxonomy = ()                  => api.get("/api/exercises/taxonomy")

export default api
