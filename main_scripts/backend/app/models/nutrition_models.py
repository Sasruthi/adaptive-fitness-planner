"""
FILE LOCATION: backend/app/models/nutrition_models.py

Nutrition catalog — structured data, mirrors the Exercise model pattern.
Sourced from the Indian Nutrient Databank (INDB), an open-access resource
derived from ICMR-NIN Indian Food Composition Tables (IFCT) 2017/2004:
  Vijayakumar A, Dubasi HB, Awasthi A, Jaacks LM. Development of an Indian
  Food Composition Database. Curr Dev Nutr. 2024.
  https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-

1,014 commonly consumed Indian recipes with verified per-100g and
per-serving macros — used to GROUND diet plan numbers instead of letting
the LLM invent calories/macros.
"""

from sqlalchemy import Column, Integer, String, Float
from app.models.models import Base  # same declarative Base as Exercise, User, Plan


class NutritionItem(Base):
    __tablename__ = "nutrition_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    food_code = Column(String(20), unique=True, index=True)   # e.g. "ASC001", "BFP346", "OSR142"
    food_name = Column(String(255), nullable=False, index=True)
    source = Column(String(100), default="INDB (ICMR-NIN IFCT 2017/2004 derived)")

    # Per 100g
    energy_kcal = Column(Float)
    protein_g   = Column(Float)
    carb_g      = Column(Float)
    fat_g       = Column(Float)
    fibre_g     = Column(Float)
    freesugar_g = Column(Float)

    # Per typical serving (a recipe-appropriate unit, e.g. "tea cup", "bowl")
    servings_unit           = Column(String(50))
    serving_energy_kcal     = Column(Float)
    serving_protein_g       = Column(Float)
    serving_carb_g          = Column(Float)
    serving_fat_g           = Column(Float)

    def __repr__(self):
        return f"<NutritionItem {self.food_code} {self.food_name}>"
