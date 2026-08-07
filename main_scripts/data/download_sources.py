from pathlib import Path
import urllib.request
import json
import ssl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "adaptive-fitness-planner-data"
RAW = BASE / "raw"
MANIFESTS = BASE / "manifests"

folders = [
    RAW / "india_guidelines",
    RAW / "global_generic",
    RAW / "structured",
    RAW / "authored",
    MANIFESTS,
]

for f in folders:
    f.mkdir(parents=True, exist_ok=True)

sources = [
    {
        "id": "SRC001",
        "name": "ICMR-NIN Dietary Guidelines for Indians 2024",
        "url": "https://nin.res.in/dietaryguidelines/pdfjs/locale/DGI_2024.pdf",
        "path": RAW / "india_guidelines" / "icmr_nin_dietary_guidelines_indians_2024.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "grounding",
    },
    {
        "id": "SRC002",
        "name": "NIN Dietary Guidelines Website Copy",
        "url": "https://www.nin.res.in/downloads/DietaryGuidelinesforNINwebsite.pdf",
        "path": RAW / "india_guidelines" / "nin_dietary_guidelines_website_copy.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "backup_grounding",
    },
    {
        "id": "SRC003",
        "name": "Fit India Fitness Protocols 18-65",
        "url": "https://cimp.ac.in/wp-content/uploads/2024/01/FitIndia.pdf",
        "path": RAW / "india_guidelines" / "fit_india_fitness_protocols_18_65.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "grounding",
    },
    {
        "id": "SRC004",
        "name": "NIN DGI Booklet English",
        "url": "https://www.nin.res.in/downloads/DGI_Booklet_English_CMYK.pdf",
        "path": RAW / "india_guidelines" / "nin_dgi_booklet_english.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "grounding",
    },
    {
        "id": "SRC005",
        "name": "ICMR Nutrient Requirements Press Release",
        "url": "https://www.icmr.gov.in/icmrobject/custom_data/1702892982_icmr_press_release_recommended_dietary_allowances_for_indians.pdf",
        "path": RAW / "india_guidelines" / "icmr_nutrient_requirements_press_release.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "nutrient_reference",
    },
    {
        "id": "SRC006",
        "name": "FSSAI Eat Right India Handbook",
        "url": "https://eatrightindia.gov.in/eatsmartcity/images/media/Eat_Right_India_Handbook_19_08_2020.pdf",
        "path": RAW / "india_guidelines" / "fssai_eat_right_india_handbook.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "diet_behavior",
    },
    {
        "id": "SRC007",
        "name": "FSSAI Do You Eat Right",
        "url": "https://www.fssai.gov.in/upload/knowledge_hub/852185f89a7fc009c5Book_Do_You_Eat_Right_16_10_2020.pdf",
        "path": RAW / "india_guidelines" / "fssai_do_you_eat_right.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "diet_behavior",
    },
    {
        "id": "SRC008",
        "name": "NIN Nutrition Lifestyle and Immunity",
        "url": "https://www.nin.res.in/downloads/Nutrition_Lifestyle_and_Immunity.pdf",
        "path": RAW / "india_guidelines" / "nin_nutrition_lifestyle_immunity.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "lifestyle",
    },
    {
        "id": "SRC009",
        "name": "Common Yoga Protocol",
        "url": "https://www.mea.gov.in/images/pdf/common-yoga-protocol.pdf",
        "path": RAW / "india_guidelines" / "common_yoga_protocol.pdf",
        "tier": "Tier 1",
        "type": "pdf",
        "use": "activity_guidance",
    },
    {
        "id": "SRC010",
        "name": "WHO Physical Activity and Sedentary Behaviour Guidelines 2020",
        "url": "https://iris.who.int/server/api/core/bitstreams/faa83413-d89e-4be9-bb01-b24671aef7ca/content",
        "path": RAW / "global_generic" / "who_physical_activity_sedentary_guidelines_2020.pdf",
        "tier": "Tier 2",
        "type": "pdf",
        "use": "secondary_generic",
    },
    {
        "id": "SRC011",
        "name": "free-exercise-db exercises.json",
        "url": "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/dist/exercises.json",
        "path": RAW / "structured" / "free_exercise_db_exercises.json",
        "tier": "Tier 3",
        "type": "json",
        "use": "structured_exercises",
    },
    {
        "id": "SRC012",
        "name": "hasaneyldrm exercises-dataset",
        "url": "https://github.com/hasaneyldrm/exercises-dataset",
        "path": RAW / "structured" / "hasaneyldrm_exercises_dataset_repo.txt",
        "tier": "Tier 3",
        "type": "repo",
        "use": "structured_exercises",
    },
    {
        "id": "SRC013",
        "name": "wrkout exercises.json",
        "url": "https://github.com/wrkout/exercises.json",
        "path": RAW / "structured" / "wrkout_exercises_repo.txt",
        "tier": "Tier 3",
        "type": "repo",
        "use": "structured_exercises",
    },
    {
        "id": "SRC014",
        "name": "exercemus exercises",
        "url": "https://github.com/exercemus/exercises",
        "path": RAW / "structured" / "exercemus_exercises_repo.txt",
        "tier": "Tier 3",
        "type": "repo",
        "use": "structured_exercises",
    },
    {
        "id": "SRC015",
        "name": "longhaul-fitness exercises",
        "url": "https://github.com/longhaul-fitness/exercises",
        "path": RAW / "structured" / "longhaul_fitness_exercises_repo.txt",
        "tier": "Tier 3",
        "type": "repo",
        "use": "structured_exercises",
    },
]

ctx = ssl.create_default_context()
rows = ["source_id,source_name,source_url,trust_tier,source_type,intended_use,status"]

for s in sources:
    try:
        if s["type"] == "repo":
            s["path"].write_text(s["url"], encoding="utf-8")
            status = "repo_url_saved"
        else:
            urllib.request.urlretrieve(s["url"], s["path"])
            status = "downloaded"
    except Exception as e:
        s["path"].write_text(f"ERROR: {s['url']}\n{e}", encoding="utf-8")
        status = "failed"

    rows.append(
        f'{s["id"]},"{s["name"]}","{s["url"]}","{s["tier"]}","{s["type"]}","{s["use"]}","{status}"'
    )

(MANIFESTS / "source_manifest.csv").write_text("\n".join(rows), encoding="utf-8")

print(f"Done. Corpus saved in: {BASE.resolve()}")
print(f"Manifest saved in: {(MANIFESTS / 'source_manifest.csv').resolve()}")