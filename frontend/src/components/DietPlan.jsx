// FILE: frontend/src/components/DietPlan.jsx
const MEAL_ICONS = { breakfast:"🌅", mid_morning_snack:"🍎", lunch:"🍱", evening_snack:"🫖", dinner:"🌙" }
const MEAL_LABELS = { breakfast:"Breakfast", mid_morning_snack:"Mid-Morning Snack", lunch:"Lunch", evening_snack:"Evening Snack", dinner:"Dinner" }

/** Coerce LLM-ish values into a string list (arrays, single string, or dict values). */
function asStringList(value) {
  if (value == null) return []
  if (Array.isArray(value)) {
    return value
      .map(v => (typeof v === "string" ? v : (v && (v.name || v.dish || v.text)) || null))
      .filter(Boolean)
  }
  if (typeof value === "string") {
    const t = value.trim()
    return t ? [t] : []
  }
  if (typeof value === "object") {
    return Object.values(value).flatMap(asStringList)
  }
  return []
}

/** Normalize meals whether the model returned a list or a meal-name keyed object. */
function normalizeMeals(meals) {
  if (!meals) return []
  if (Array.isArray(meals)) {
    return meals.filter(m => m && typeof m === "object")
  }
  if (typeof meals === "object") {
    return Object.entries(meals).map(([key, val]) => {
      if (val && typeof val === "object") {
        return { meal: val.meal || key, ...val }
      }
      return { meal: key, suggestions: asStringList(val) }
    })
  }
  return []
}

function MacroBadge({ label, value, unit, color }) {
  if (value == null || value === "") return null
  return (
    <div className={`flex flex-col items-center px-3 py-1.5 rounded-xl ${color}`}>
      <span className="text-xs font-bold">{value}{unit}</span>
      <span className="text-[10px] opacity-70">{label}</span>
    </div>
  )
}

export default function DietPlan({ diet }) {
  if (!diet || typeof diet !== "object") {
    return (
      <p className="text-center text-gray-400 text-sm py-8">
        No nutrition plan in this response. Ask the chat to regenerate your plan.
      </p>
    )
  }

  const meals = normalizeMeals(diet.meals)
  const indiaTips = asStringList(diet.india_specific_tips)
  const avoid = asStringList(diet.foods_to_avoid)
  const bmr = diet.bmr || diet.calorie_target?.bmr
  const tdee = diet.tdee || diet.calorie_target?.tdee
  const targetKcal = diet.target_calories || diet.calorie_target?.target_calories
  const check = diet.calorie_target_check
  const hasMeta = Boolean(bmr || tdee || targetKcal || diet.daily_calories_estimate || diet.hydration)

  if (!meals.length && !hasMeta && !indiaTips.length && !avoid.length) {
    return (
      <p className="text-center text-gray-400 text-sm py-8">
        Nutrition details are missing for this plan. In chat, ask to regenerate the plan
        (meal enrichment may have failed).
      </p>
    )
  }

  return (
    <div className="space-y-4">

      {/* BMR / TDEE / target (from Mifflin–St Jeor when age+sex+weight+height known) */}
      {(bmr || tdee || targetKcal) && (
        <div className="card border-gray-100">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide mb-2">
            Metabolic targets (Mifflin–St Jeor)
            {diet.calories_estimated ? " — approximate" : ""}
          </p>
          {diet.calories_estimated && (
            <p className="text-xs text-amber-700 mb-2">
              {diet.calorie_estimate_note ||
                "Height and/or weight were estimated from India age/sex averages. Share measured values for a tighter calorie target."}
              {diet.assumed_height_cm != null && diet.assumed_weight_kg != null && (
                <> (used ≈{diet.assumed_height_cm} cm / ≈{diet.assumed_weight_kg} kg)</>
              )}
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            {bmr != null && (
              <MacroBadge label="BMR" value={bmr} unit=" kcal" color="bg-gray-100 text-gray-700" />
            )}
            {tdee != null && (
              <MacroBadge label="TDEE" value={tdee} unit=" kcal" color="bg-slate-100 text-slate-700" />
            )}
            {targetKcal != null && (
              <MacroBadge label="Goal target" value={targetKcal} unit=" kcal" color="bg-brand-100 text-brand-700" />
            )}
          </div>
          {check && (
            <p className="text-xs text-gray-500 mt-2">
              Meals sum ≈ {check.actual_calories_from_meals} kcal
              {check.within_tolerance
                ? " (within target range)"
                : ` (Δ ${check.delta > 0 ? "+" : ""}${check.delta} vs target)`}
            </p>
          )}
        </div>
      )}

      {/* Daily targets from verified meals */}
      {diet.daily_calories_estimate != null && diet.daily_calories_estimate !== "" && (
        <div className="card bg-brand-50 border-brand-200">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-xs text-brand-600 font-medium uppercase tracking-wide">Meals total (INDB)</p>
              <p className="text-3xl font-bold text-brand-700">
                {diet.daily_calories_estimate} <span className="text-base font-normal">kcal</span>
              </p>
              {diet.calorie_note && <p className="text-xs text-brand-600 opacity-75 mt-0.5">{diet.calorie_note}</p>}
            </div>
            <span className="text-4xl">🔥</span>
          </div>
          {diet.macros && (
            <div className="flex gap-2 mt-2">
              <MacroBadge label="Protein" value={diet.macros.protein_g} unit="g" color="bg-blue-100 text-blue-700" />
              <MacroBadge label="Carbs"   value={diet.macros.carbs_g}   unit="g" color="bg-yellow-100 text-yellow-700" />
              <MacroBadge label="Fat"     value={diet.macros.fat_g}      unit="g" color="bg-orange-100 text-orange-700" />
            </div>
          )}
        </div>
      )}

      {!meals.length && (
        <p className="text-sm text-amber-700 bg-amber-50 border border-amber-100 rounded-xl px-3 py-2">
          No meal slots were returned for this plan. Ask the chat to regenerate so enrichment can refill meals.
        </p>
      )}

      {/* Meals */}
      {meals.map((meal,i) => {
        const suggestions = asStringList(meal.suggestions)
        return (
        <div key={i} className="card">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-xl">{MEAL_ICONS[meal.meal]||"🍽️"}</span>
              <div>
                <p className="font-semibold text-gray-800 text-sm">{MEAL_LABELS[meal.meal]||meal.meal||`Meal ${i+1}`}</p>
                {meal.timing && <p className="text-xs text-gray-400">{meal.timing}</p>}
              </div>
            </div>
            {meal.calories != null && meal.calories !== "" && (
              <span className="tag bg-gray-100 text-gray-500 text-xs">~{meal.calories} kcal</span>
            )}
          </div>

          {/* Nutrition breakdown per meal */}
          {(meal.protein_g||meal.carbs_g||meal.carb_g||meal.fat_g) && (
            <div className="flex gap-2 mb-2">
              {meal.protein_g != null && <span className="tag bg-blue-50 text-blue-600">P: {meal.protein_g}g</span>}
              {(meal.carbs_g != null || meal.carb_g != null) && (
                <span className="tag bg-yellow-50 text-yellow-600">C: {meal.carbs_g ?? meal.carb_g}g</span>
              )}
              {meal.fat_g != null && <span className="tag bg-orange-50 text-orange-600">F: {meal.fat_g}g</span>}
            </div>
          )}
          {meal.verified === false && meal.nutrient_note && (
            <p className="text-xs text-amber-600 mb-2">{meal.nutrient_note}</p>
          )}
          {meal.verified === true && meal.matched_food && (
            <p className="text-[10px] text-gray-400 mb-1">INDB: {meal.matched_food}</p>
          )}

          <ul className="space-y-1">
            {suggestions.map((s,j)=>(
              <li key={j} className="text-sm text-gray-600 flex items-start gap-1.5">
                <span className="text-brand-500 mt-0.5 flex-shrink-0">•</span>{s}
              </li>
            ))}
          </ul>
          {meal.notes && <p className="mt-2 text-xs text-gray-400 italic">{meal.notes}</p>}
        </div>
        )
      })}

      {/* India tips */}
      {indiaTips.length>0 && (
        <div className="card bg-orange-50 border-orange-100">
          <p className="font-semibold text-sm text-orange-700 mb-2">Tips</p>
          <ul className="space-y-1.5">
            {indiaTips.map((t,i)=>(
              <li key={i} className="text-sm text-gray-600 flex gap-1.5"><span className="text-orange-400 flex-shrink-0">✦</span>{t}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Avoid */}
      {avoid.length>0 && (
        <div className="card bg-red-50 border-red-100">
          <p className="font-semibold text-sm text-red-700 mb-2">⚠️ Avoid These</p>
          <ul className="space-y-1">
            {avoid.map((f,i)=>(
              <li key={i} className="text-sm text-red-600 flex gap-1.5"><span className="flex-shrink-0">✕</span>{f}</li>
            ))}
          </ul>
        </div>
      )}

      {diet.hydration && (
        <div className="card flex items-center gap-3">
          <span className="text-2xl">💧</span>
          <p className="text-sm text-gray-600">{diet.hydration}</p>
        </div>
      )}
    </div>
  )
}
