"""Unit tests for profile validation / safety gates (no LLM, no Qdrant)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.conversation.profile_store import Profile, BODY_PARTS


def test_unsafe_without_health_flags():
    p = Profile(goal="lose_fat", age=28, gender="female",
                target_body_parts=["waist"], available_equipment=["body only"],
                fitness_level="beginner", time_per_day_minutes=30)
    assert not p.is_safe_to_plan()
    assert "health_flags" in p.missing_fields()


def test_none_health_flag_unlocks_plan():
    p = Profile(
        goal="lose_fat", age=28, gender="female",
        target_body_parts=["upper arms"], available_equipment=["body only"],
        fitness_level="beginner", time_per_day_minutes=30,
        health_flags=["none"],
    )
    assert p.is_safe_to_plan()


def test_full_body_expands():
    p = Profile()
    p.merge(target_body_parts=["full body"])
    assert set(BODY_PARTS).issubset(set(p.target_body_parts))


def test_equipment_and_core_aliases():
    p = Profile()
    p.merge(
        target_body_parts=["core"],
        available_equipment=["no equipment"],
    )
    assert "waist" in p.target_body_parts
    assert "body only" in p.available_equipment


def test_diet_only_skips_exercise_slots():
    p = Profile(
        goal="lose_fat", age=30, gender="male",
        health_flags=["none"], plan_mode="diet_only",
    )
    assert p.is_safe_to_plan()
    assert "target_body_parts" not in p.missing_fields()


if __name__ == "__main__":
    test_unsafe_without_health_flags()
    test_none_health_flag_unlocks_plan()
    test_full_body_expands()
    test_equipment_and_core_aliases()
    test_diet_only_skips_exercise_slots()
    print("test_profile_store: all passed")
