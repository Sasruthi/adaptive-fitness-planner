"""
FILE LOCATION: backend/app/scripts/load_nutrition_db.py

One-time loader: INDB.xlsx (Indian Nutrient Databank, "Nutrient Data"
sheet, 1014 recipes) -> data/fitness.db, table `nutrition_items`.

Source: https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-
        (open-access; derived from ICMR-NIN IFCT 2017/2004)

Usage:
    python -m app.scripts.load_nutrition_db --xlsx /path/to/INDB.xlsx
"""

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.models.models import Base
from app.models.nutrition_models import NutritionItem

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fitness.db"

COLUMN_MAP = {
    "food_code": "food_code",
    "food_name": "food_name",
    "energy_kcal": "energy_kcal",
    "protein_g": "protein_g",
    "carb_g": "carb_g",
    "fat_g": "fat_g",
    "fibre_g": "fibre_g",
    "freesugar_g": "freesugar_g",
    "servings_unit": "servings_unit",
    "unit_serving_energy_kcal": "serving_energy_kcal",
    "unit_serving_protein_g":   "serving_protein_g",
    "unit_serving_carb_g":      "serving_carb_g",
    "unit_serving_fat_g":       "serving_fat_g",
}


def load(xlsx_path: str):
    df = pd.read_excel(xlsx_path, sheet_name="Nutrient Data")
    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    df = df.dropna(subset=["food_name"])
    df = df.where(pd.notnull(df), None)

    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[NutritionItem.__table__])
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        inserted, skipped = 0, 0
        for _, row in df.iterrows():
            exists = session.query(NutritionItem).filter_by(food_code=row["food_code"]).first()
            if exists:
                skipped += 1
                continue
            session.add(NutritionItem(**row.to_dict()))
            inserted += 1
        session.commit()
        print(f"[NutritionDB] Inserted {inserted} recipes, skipped {skipped} duplicates. "
              f"DB: {DB_PATH}")
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="Path to INDB.xlsx")
    args = parser.parse_args()
    load(args.xlsx)
