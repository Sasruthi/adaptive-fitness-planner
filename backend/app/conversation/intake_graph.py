"""
FILE LOCATION: backend/app/conversation/intake_graph.py

Natural conversational intake:
- LLM extracts ALL slots from every message simultaneously
- Keyword fallback validates each extraction
- Questions are short and natural — no examples, no counters
- Background RAG triggers immediately on any health context
- Corrections work naturally mid-conversation
"""

import os, json, re, threading
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_sessions: Dict[str, dict] = {}

GOALS = ["lose_fat","build_muscle","improve_strength","improve_flexibility",
         "improve_endurance","general_fitness","rehabilitation","stress_relief"]
BODY_PARTS = ["neck","shoulders","chest","back","upper arms","lower arms",
              "waist","upper legs","lower legs","cardio"]
EQUIPMENT_OPTIONS = ["body only","dumbbell","barbell","kettlebells","bands",
                     "cable","machine","exercise ball","foam roll","none"]
FITNESS_LEVELS = ["beginner","intermediate","expert"]
KNOWN_FLAGS = ["high_bp","low_bp","diabetes","knee_injury","back_injury",
               "shoulder_injury","wrist_injury","ankle_injury","heart_condition",
               "asthma","osteoporosis","acidity","pregnancy","obesity","none"]
NONE_HEALTH_PHRASES = {
    "none", "no", "nope", "nah", "nothing", "all clear", "no conditions",
    "i'm fine", "im fine", "healthy", "clear", "n/a", "na", "not really", "no issues",
}

# ── LLM factory ───────────────────────────────────────────────────────────────
def get_llm():
    p = os.getenv("LLM_PROVIDER","groq").lower()
    if p == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),
                        api_key=os.getenv("GROQ_API_KEY"), temperature=0.1, max_tokens=1000)
    elif p == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL","llama3.1"), temperature=0.1)
    elif p == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=os.getenv("MISTRAL_MODEL","mistral-small-latest"),
                             api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.1)
    else:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-4o"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION","2024-02-01"),
            temperature=0.1, max_tokens=1000)

def ask(system, user):
    return get_llm().invoke([SystemMessage(content=system),
                              HumanMessage(content=user)]).content.strip()

def extract_json(text):
    text = re.sub(r'```json\s*','',text)
    text = re.sub(r'```\s*','',text)
    try: return json.loads(text.strip())
    except: pass
    for m in sorted(re.findall(r'\{.*?\}', text, re.DOTALL), key=len, reverse=True):
        try: return json.loads(m)
        except: continue
    return {}


# Words that should NEVER be treated as health conditions
NOISE_WORDS = {
    "hi","hello","hey","okay","ok","yes","no","nope","sure","thanks","thank",
    "you","the","and","for","with","have","has","had","get","got","want",
    "need","like","just","also","well","good","bad","fine","great","nice",
    "please","sorry","none","nothing","clear","fit","healthy","normal","all",
    "that","this","there","here","some","any","from","into","about","more",
    "my","me","i","a","an","is","it","do","be","am","was","can","will",
    "workout","plan","exercise","fitness","body","goal","help","make","give",
}

def is_medical_term(term: str) -> bool:
    """Only accept terms that look like actual medical/health conditions."""
    term = term.lower().strip()
    if len(term) < 4: return False
    if term in NOISE_WORDS: return False
    medical_patterns = [
        "pain","ache","injury","syndrome","disorder","disease","condition",
        "problem","issue","weakness","deficiency","imbalance","pressure",
        "diabetes","thyroid","cortisol","pcod","pcos","fibro","lupus",
        "arthritis","migraine","epilepsy","hormonal","hormone","cholesterol",
        "asthma","cardiac","gastri","reflux","anxiety","depression","insomnia",
        "stress","panic","ptsd","adhd",
    ]
    return any(p in term for p in medical_patterns)

def is_greeting(text: str) -> bool:
    """Detect if the message is just a greeting with no fitness info."""
    stripped = text.lower().strip().rstrip("!.,?")
    greetings = {"hi","hello","hey","hiya","howdy","greetings","sup","yo",
                 "good morning","good evening","good afternoon","good day"}
    return stripped in greetings or (len(stripped.split()) <= 2 and stripped in greetings)

EXTRACTION_PROMPT = f"""You extract fitness profile information from a user message.
Extract only what is clearly stated. Return null/[] for anything not mentioned.

Valid goals: {GOALS}
GOAL mappings: lose/reduce/slim/tone/fat/weight/belly/cut → lose_fat | muscle/bulk/gain/mass → build_muscle | strength/lift/strong/power/strengthen → improve_strength | flex/stretch/mobil/yoga → improve_flexibility | stamina/endur/cardio/run/marathon → improve_endurance | stress/relax/mental/calm → stress_relief | rehab/recover/injur/physio → rehabilitation | fit/healthy/overall/general → general_fitness

Valid body parts: {BODY_PARTS}
BODY PART mappings: belly/stomach/tummy/abs/gut/core/waist/muffin → waist | arms/biceps/triceps/arm fat → upper arms | legs/thighs/quads/hips → upper legs | calves/shins → lower legs | shoulders/deltoids → shoulders | chest/pecs → chest | back/spine/lats → back | full body/everything → [waist,upper legs,upper arms,chest,back] | cardio/heart → cardio

GENDER: female/woman/girl/she/her → female | male/man/boy/he/him → male | non-binary/they → non-binary

HEALTH flags: {KNOWN_FLAGS}
high_bp: BP/hypertension/blood pressure | diabetes: sugar/diabetic | knee_injury: knee pain/ACL | back_injury: back pain/disc/spondylitis/cervical | shoulder_injury: shoulder pain/rotator | wrist_injury: wrist pain/carpal | ankle_injury: ankle/plantar | heart_condition: heart/cardiac | asthma: asthma/breathing | acidity: acidity/GERD/gastritis | pregnancy: pregnant | obesity: obese/overweight | none: none/no/clear/fit/healthy
Custom (not in list): cortisol/PCOD/PCOS/thyroid/fibromyalgia/lupus/arthritis/hormonal/anxiety/depression/stress → custom_conditions

EQUIPMENT: {EQUIPMENT_OPTIONS}
none/nothing/bodyweight/home only → none | dumbbell → dumbbell | barbell/bar → barbell | kettlebell → kettlebells | band/TRX/resistance → bands | cable → cable | machine → machine | full gym → [dumbbell,barbell,cable,machine]

TIME: 1 hour/60 min → 60 | 45 → 45 | half hour/30 → 30 | 15 → 15

Return ONLY this JSON:
{{"goal":null,"target_body_parts":[],"age":null,"gender":null,"height_cm":null,"weight_kg":null,"known_health_flags":[],"custom_conditions":[],"available_equipment":[],"fitness_level":null,"time_per_day_minutes":null}}"""

def extract_all_slots(user_msg):
    try:
        extracted = extract_json(ask(EXTRACTION_PROMPT, user_msg))
    except Exception:
        extracted = {}

    msg = user_msg.lower()

    if not extracted.get("goal"):
        if any(w in msg for w in ["fat","weight","slim","belly","tone","calori","reduc","thin","cut","lose"]): extracted["goal"]="lose_fat"
        elif any(w in msg for w in ["muscle","bulk","mass","gain","bigger"]): extracted["goal"]="build_muscle"
        elif any(w in msg for w in ["strength","strong","lift","power","strengthen"]): extracted["goal"]="improve_strength"
        elif any(w in msg for w in ["flex","stretch","mobil","yoga"]): extracted["goal"]="improve_flexibility"
        elif any(w in msg for w in ["stamina","endur","cardio","run","marathon"]): extracted["goal"]="improve_endurance"
        elif any(w in msg for w in ["stress","relax","mental","calm"]): extracted["goal"]="stress_relief"
        elif any(w in msg for w in ["rehab","recover","injur","physio"]): extracted["goal"]="rehabilitation"

    if not extracted.get("gender"):
        if any(w in msg for w in ["female","woman","girl"," she "," her ","lady"]): extracted["gender"]="female"
        elif any(w in msg for w in ["male","man","boy"," he "," him ","guy"]): extracted["gender"]="male"

    if not extracted.get("age"):
        for n in re.findall(r'\b(\d{1,3})\b', user_msg):
            if 10<=int(n)<=100: extracted["age"]=int(n); break

    stripped = msg.strip().lower()
    if stripped in NONE_HEALTH_PHRASES:
        extracted["known_health_flags"] = ["none"]
    elif not extracted.get("known_health_flags"):
        flags=[]
        kw={"high_bp":["bp","blood pressure","hypertension"],"low_bp":["low bp","hypotension"],
            "diabetes":["sugar","diabet"],"knee_injury":["knee","acl"],
            "back_injury":["back pain","slip disc","spondyl","cervical","herniated"],
            "shoulder_injury":["shoulder pain","rotator","frozen shoulder"],
            "wrist_injury":["wrist pain","carpal"],"ankle_injury":["ankle pain","plantar"],
            "heart_condition":["heart","cardiac"],"asthma":["asthma","breathing"],
            "acidity":["acidity","acid reflux","gerd","gastri"],"pregnancy":["pregnant"],
            "obesity":["obese","obesity"]}
        for flag,keywords in kw.items():
            if any(k in msg for k in keywords): flags.append(flag)
        if flags: extracted["known_health_flags"]=flags

    if not extracted.get("custom_conditions"):
        custom=[t for t in ["cortisol","pcod","pcos","thyroid","fibromyalgia",
                             "lupus","arthritis","migraine","hormonal","hormone",
                             "cholesterol","kidney","liver","epilepsy","autism",
                             "scoliosis","hernia","vertigo","tinnitus","ibs",
                             "anxiety","depression","stress","insomnia","panic","ptsd","adhd"]
                if t in msg]
        if custom:
            extracted["custom_conditions"]=custom
        elif (stripped not in NONE_HEALTH_PHRASES and len(stripped.split()) <= 2
              and stripped.isalpha() and is_medical_term(stripped)):
            extracted["custom_conditions"]=[stripped]
    else:
        VALID_MEDICAL_TERMS = {
            "cortisol","pcod","pcos","thyroid","fibromyalgia","lupus",
            "arthritis","migraine","hormonal","hormone","cholesterol",
            "kidney","liver","epilepsy","scoliosis","hernia","vertigo",
            "tinnitus","ibs","celiac","crohn","vitiligo","psoriasis",
            "hypothyroid","hyperthyroid","anaemia","anemia","gout",
            "parkinson","alzheimer","autism","adhd","bipolar",
            "hypothyroidism","hyperthyroidism","anxiety","depression",
            "stress","insomnia","panic","ptsd",
        }
        cleaned = [
            c.lower().strip() for c in extracted["custom_conditions"]
            if len(c.strip()) > 3
            and any(term in c.lower() for term in VALID_MEDICAL_TERMS)
        ]
        extracted["custom_conditions"] = cleaned

    if not extracted.get("available_equipment"):
        has_equip=any(w in msg for w in ["dumbbell","barbell","kettlebell","band","cable","machine"])
        if not has_equip and any(w in msg for w in ["no equipment","nothing","no gym",
                                                      "bodyweight","body weight","just myself"]):
            extracted["available_equipment"]=["none"]
        elif not has_equip and "home" in msg and "gym" not in msg:
            extracted["available_equipment"]=["none"]
        else:
            equip=[]
            if any(w in msg for w in ["dumbbell","dumbbells"]): equip.append("dumbbell")
            if "barbell" in msg or ("bar" in msg and "barbell" not in msg and "no bar" not in msg): equip.append("barbell")
            if "kettlebell" in msg: equip.append("kettlebells")
            if any(w in msg for w in ["band","resistance band","trx"]): equip.append("bands")
            if "cable" in msg: equip.append("cable")
            if "machine" in msg and "no" not in msg[:msg.index("machine")+1]: equip.append("machine")
            if "full gym" in msg: equip=["dumbbell","barbell","cable","machine"]
            if equip: extracted["available_equipment"]=equip

    if not extracted.get("fitness_level"):
        if any(w in msg for w in ["begin","new to","start","never","novice","zero","couch","unfit","no exp"]): extracted["fitness_level"]="beginner"
        elif any(w in msg for w in ["expert","advanc","athlete","compet","elite","very fit"]): extracted["fitness_level"]="expert"
        elif any(w in msg for w in ["inter","moderate","sometimes","occasio","regular","medium"]): extracted["fitness_level"]="intermediate"

    if not extracted.get("time_per_day_minutes"):
        if any(w in msg for w in ["1 hour","one hour","an hour","60 min","60 minutes"]): extracted["time_per_day_minutes"]=60
        elif "45" in msg: extracted["time_per_day_minutes"]=45
        elif any(w in msg for w in ["half hour","30 min","30 minutes","half an hour"]): extracted["time_per_day_minutes"]=30
        elif "15" in msg: extracted["time_per_day_minutes"]=15

    return extracted


# ── Slot applier ──────────────────────────────────────────────────────────────
def apply_slots(state, ex):
    if ex.get("goal") and not state.get("goal") and ex["goal"] in GOALS:
        state["goal"] = ex["goal"]
    if ex.get("target_body_parts") and not state.get("target_body_parts"):
        valid = [p for p in ex["target_body_parts"] if p in BODY_PARTS]
        if valid: state["target_body_parts"] = valid
    if ex.get("age") and not state.get("age"):
        try: state["age"] = int(ex["age"])
        except: pass
    if ex.get("gender") and not state.get("gender"):
        state["gender"] = ex["gender"]
    if ex.get("height_cm") and not state.get("height_cm"):
        try: state["height_cm"] = float(ex["height_cm"])
        except: pass
    if ex.get("weight_kg") and not state.get("weight_kg"):
        try: state["weight_kg"] = float(ex["weight_kg"])
        except: pass
    new_flags = [f for f in ex.get("known_health_flags",[]) if f in KNOWN_FLAGS]
    if new_flags:
        existing = state.get("health_flags",[])
        merged = list(set(existing+new_flags)-{"none"})
        state["health_flags"] = merged if merged else ["none"]
    elif "none" in ex.get("known_health_flags",[]):
        if not state.get("health_flags"):
            state["health_flags"] = ["none"]
    new_custom = ex.get("custom_conditions",[])
    new_custom = [c for c in new_custom if is_medical_term(c)]
    if new_custom:
        state["custom_health_notes"] = list(set(state.get("custom_health_notes",[])+new_custom))
        if not state.get("health_flags"):
            state["health_flags"] = ["none"]
    if ex.get("available_equipment") and not state.get("available_equipment"):
        valid = [e for e in ex["available_equipment"] if e in EQUIPMENT_OPTIONS]
        if valid: state["available_equipment"] = valid
    if ex.get("fitness_level") and not state.get("fitness_level"):
        if ex["fitness_level"] in FITNESS_LEVELS: state["fitness_level"] = ex["fitness_level"]
    if ex.get("time_per_day_minutes") and not state.get("time_per_day_minutes"):
        if ex["time_per_day_minutes"] in [15,30,45,60]:
            state["time_per_day_minutes"] = ex["time_per_day_minutes"]


# ── Missing slot detector ─────────────────────────────────────────────────────
def health_is_complete(state):
    """Health step done when we have known flags and/or custom notes (e.g. anxiety)."""
    return bool(state.get("health_flags")) or bool(state.get("custom_health_notes"))


def get_missing(state):
    missing = []
    if not state.get("goal"):                  missing.append("goal")
    if not state.get("target_body_parts"):     missing.append("target_body_parts")
    if not state.get("age"):                   missing.append("age")
    if not state.get("gender"):                missing.append("gender")
    if not health_is_complete(state):          missing.append("health_flags")
    if not state.get("available_equipment"):   missing.append("available_equipment")
    if not state.get("fitness_level"):         missing.append("fitness_level")
    if not state.get("time_per_day_minutes"):  missing.append("time_per_day_minutes")
    return missing


# ── Natural questions — no examples, no counters ──────────────────────────────
QUESTIONS = {
    "goal":                 "What's your main fitness goal?",
    "target_body_parts":    "Which part of your body would you like to focus on?",
    "age":                  "How old are you?",
    "gender":               "What's your gender?",
    "health_flags":         "Do you have any health conditions or physical issues I should know about? Feel free to say none if all clear.",
    "available_equipment":  "What equipment do you have available for workouts?",
    "fitness_level":        "How would you describe your current fitness level?",
    "time_per_day_minutes": "How much time can you give per day for working out?",
}

COMBINED_QUESTIONS = {
    frozenset(["age","gender"]): "What's your age and gender?",
    frozenset(["age","gender","health_flags"]): "What's your age, gender, and do you have any health conditions?",
    frozenset(["available_equipment","fitness_level"]): "What equipment do you have, and how would you describe your fitness level?",
    frozenset(["available_equipment","fitness_level","time_per_day_minutes"]): "What equipment do you have, what's your fitness level, and how much time can you spare per day?",
    frozenset(["fitness_level","time_per_day_minutes"]): "What's your fitness level, and how much time per day can you dedicate?",
}

def build_question(missing):
    missing_set = frozenset(missing[:3])
    for combo, question in COMBINED_QUESTIONS.items():
        if combo.issubset(missing_set):
            return question
    return QUESTIONS.get(missing[0], "Could you tell me a bit more?")


# ── Acknowledgement builder ───────────────────────────────────────────────────
def build_ack(state, just_filled):
    if not just_filled:
        return ""

    parts = []
    if "goal" in just_filled and state.get("goal"):
        parts.append(state["goal"].replace("_"," "))
    if "target_body_parts" in just_filled and state.get("target_body_parts"):
        parts.append(", ".join(state["target_body_parts"]))
    if "age" in just_filled and state.get("age"):
        parts.append(f"age {state['age']}")
    if "gender" in just_filled and state.get("gender"):
        parts.append(state["gender"])
    if "health_flags" in just_filled:
        flags  = [f.replace("_"," ") for f in state.get("health_flags",[]) if f != "none"]
        custom = state.get("custom_health_notes",[])
        all_h  = flags + custom
        if all_h:   parts.append(", ".join(all_h))
        else:       parts.append("no conditions — noted")
    if "available_equipment" in just_filled and state.get("available_equipment"):
        parts.append(", ".join(state["available_equipment"]))
    if "fitness_level" in just_filled and state.get("fitness_level"):
        parts.append(state["fitness_level"])
    if "time_per_day_minutes" in just_filled and state.get("time_per_day_minutes"):
        parts.append(f"{state['time_per_day_minutes']} min/day")

    if not parts:
        return ""

    if len(just_filled) >= 3:
        return f"Got it — {', '.join(parts[:3])}.\n\n"
    elif len(just_filled) == 1 and len(parts) == 1:
        return f"Got it.\n\n"
    else:
        return f"Got it — {', '.join(parts)}.\n\n"


# ── Background RAG ────────────────────────────────────────────────────────────
def _run_background_rag(state, flags, custom, goal, parts):
    try:
        from app.services.rag_retrieval import retrieve_multi_query
        queries = []
        for f in flags:
            if f != "none":
                queries.append(f"{f.replace('_',' ')} exercise safety India")
                queries.append(f"diet {f.replace('_',' ')} India nutrition")
        for c in custom:
            queries.append(f"{c} exercise India fitness")
            queries.append(f"{c} diet nutrition India")
        if goal:
            queries.append(f"{goal.replace('_',' ')} India fitness plan")
        if parts:
            queries.append(f"exercise {' '.join(parts)} India")
        if not queries: return
        chunks = retrieve_multi_query(queries, {"trust_tier__in":["Tier 1","Tier 2"]},
                                       top_k_per_query=3)
        state["preloaded_rag_chunks"] = chunks
        print(f"[BackgroundRAG] {len(chunks)} chunks loaded for: {queries[0]}")
    except Exception as e:
        print(f"[BackgroundRAG] {e}")

def trigger_background_rag(state):
    flags  = state.get("health_flags", [])
    custom = state.get("custom_health_notes", [])
    goal   = state.get("goal")
    parts  = state.get("target_body_parts", [])
    has_ctx = (flags and flags!=["none"]) or custom or goal or parts
    if has_ctx and not state.get("rag_triggered"):
        state["rag_triggered"] = True
        threading.Thread(
            target=_run_background_rag,
            args=(state, flags, custom, goal, parts),
            daemon=True
        ).start()


# ── Profile summary ───────────────────────────────────────────────────────────
def build_summary(state):
    goal   = (state.get("goal") or "?").replace("_"," ").title()
    parts  = ", ".join(state.get("target_body_parts") or [])
    age    = state.get("age","?")
    gender = state.get("gender","?")
    flags  = state.get("health_flags") or ["none"]
    custom = state.get("custom_health_notes") or []
    equip  = ", ".join(state.get("available_equipment") or ["none"])
    level  = state.get("fitness_level","?")
    time   = state.get("time_per_day_minutes","?")

    health_str = ", ".join([f.replace("_"," ") for f in flags if f!="none"])
    if custom: health_str += (", " if health_str else "") + ", ".join(custom)
    if not health_str: health_str = "none"

    rag_note = ""
    chunks = state.get("preloaded_rag_chunks",[])
    if chunks:
        rag_note = f"\n\nI've already pulled {len(chunks)} relevant guideline passages for your plan."

    return (
        f"Here's what I have for you:\n\n"
        f"**Goal:** {goal}\n"
        f"**Focus:** {parts}\n"
        f"**Age / Gender:** {age} / {gender}\n"
        f"**Health:** {health_str}\n"
        f"**Equipment:** {equip}\n"
        f"**Level:** {level}\n"
        f"**Time per day:** {time} minutes"
        f"{rag_note}\n\n"
        f"Does this look right? Say yes to generate your plan, or correct anything."
    )


# ── Main processor ────────────────────────────────────────────────────────────
def process_message_intelligently(state, user_msg):

    if is_greeting(user_msg):
        missing = get_missing(state)
        if missing:
            return "Hey! " + build_question(missing)
        return build_summary(state)

    if state.get("stage") == "confirm_profile":
        msg = user_msg.lower()
        if any(w in msg for w in ["yes","correct","looks good","ok","perfect","fine",
                                   "right","confirm","proceed","generate","sure","go","start"]):
            state["stage"] = "done"
            return "__PLAN_READY__"
        ex = extract_all_slots(user_msg)
        for k,v in [("goal", ex.get("goal")),
                     ("target_body_parts", [p for p in ex.get("target_body_parts",[]) if p in BODY_PARTS]),
                     ("fitness_level", ex.get("fitness_level")),
                     ("time_per_day_minutes", ex.get("time_per_day_minutes"))]:
            if v: state[k] = v
        if ex.get("age"): state["age"] = int(ex["age"])
        if ex.get("gender"): state["gender"] = ex["gender"]
        new_flags = [f for f in ex.get("known_health_flags",[]) if f in KNOWN_FLAGS]
        if new_flags: state["health_flags"] = new_flags
        new_equip = [e for e in ex.get("available_equipment",[]) if e in EQUIPMENT_OPTIONS]
        if new_equip: state["available_equipment"] = new_equip
        return build_summary(state)

    before = {k: state.get(k) for k in ["goal","target_body_parts","age","gender",
                                          "health_flags","available_equipment",
                                          "fitness_level","time_per_day_minutes"]}
    ex = extract_all_slots(user_msg)
    apply_slots(state, ex)

    just_filled = []
    for slot in ["goal","target_body_parts","age","gender","health_flags",
                 "available_equipment","fitness_level","time_per_day_minutes"]:
        val = state.get(slot)
        was = before.get(slot)
        if val and (not was or was != val):
            just_filled.append(slot)
    if health_is_complete(state) and not health_is_complete(before):
        if "health_flags" not in just_filled:
            just_filled.append("health_flags")

    trigger_background_rag(state)

    missing = get_missing(state)

    if not missing:
        from app.conversation.state import build_sql_filters, build_rag_filters
        state["sql_filters"] = build_sql_filters(state)
        state["rag_filters"]  = build_rag_filters(state)
        state["stage"] = "confirm_profile"
        ack = build_ack(state, just_filled)
        return ack + build_summary(state)

    ack = build_ack(state, just_filled)
    question = build_question(missing)
    return ack + question


# ── Public API ────────────────────────────────────────────────────────────────
def start_conversation(thread_id):
    state = {
        "thread_id": thread_id, "stage": "collecting",
        "chat_history": [], "_custom_noted": False,
        "goal": None, "target_body_parts": [], "age": None, "gender": None,
        "height_cm": None, "weight_kg": None, "health_flags": [],
        "custom_health_notes": [], "available_equipment": [], "fitness_level": None,
        "time_per_day_minutes": None, "sql_filters": {}, "rag_filters": {},
        "preloaded_rag_chunks": [], "rag_triggered": False,
    }
    reply = ("Hi! I'm your Adaptive Fitness Planner. "
             "I'll build you a personalised workout and diet plan. "
             "You can share everything at once or we can go step by step — whatever works for you.\n\n"
             "What's your main fitness goal?")
    state["chat_history"].append({"role":"assistant","content":reply})
    _sessions[thread_id] = state
    return {"thread_id": thread_id, "stage": "collecting", "message": reply}


def process_user_message(user_message, thread_id):
    state = _sessions.get(thread_id)
    if not state:
        return start_conversation(thread_id)
    state["chat_history"].append({"role":"user","content":user_message})
    reply = process_message_intelligently(state, user_message)
    plan_ready = (reply == "__PLAN_READY__")
    if plan_ready:
        reply = "Generating your personalised plan now..."
    state["chat_history"].append({"role":"assistant","content":reply})
    _sessions[thread_id] = state
    is_done = plan_ready or state.get("stage") == "done"
    return {
        "thread_id":             thread_id,
        "stage":                 state.get("stage","collecting"),
        "message":               reply,
        "slots_complete":        is_done,
        "sql_filters":           state.get("sql_filters",{}),
        "rag_filters":           state.get("rag_filters",{}),
        "preloaded_rag_chunks":  state.get("preloaded_rag_chunks",[]),
        "profile": {
            "goal":                 state.get("goal"),
            "target_body_parts":    state.get("target_body_parts"),
            "age":                  state.get("age"),
            "sex":                  state.get("gender"),
            "gender":               state.get("gender"),
            "height_cm":            state.get("height_cm"),
            "weight_kg":            state.get("weight_kg"),
            "health_flags":         state.get("health_flags"),
            "custom_health_notes":  state.get("custom_health_notes"),
            "available_equipment":  state.get("available_equipment"),
            "fitness_level":        state.get("fitness_level"),
            "time_per_day_minutes": state.get("time_per_day_minutes"),
            "preloaded_rag_chunks": state.get("preloaded_rag_chunks",[]),
        } if is_done else {}
    }

def slots_complete(state):
    return not get_missing(state)
