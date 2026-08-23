import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dnd_crafting_app" / "data" / "recipes.json"
OUT = ROOT / "crafting_catalog_export.csv"

with SRC.open("r", encoding="utf-8") as f:
    recipes = json.load(f)

headers = [
    "legacy_row_id",
    "legacy_profession",
    "legacy_secondary_profession",
    "legacy_tier",
    "legacy_rarity",
    "legacy_name",
    "legacy_category",
    "legacy_craft_type",
    "legacy_description",
    "legacy_use",
    "legacy_output_qty",
    "legacy_base_price_gp",
    "legacy_vendor_price_gp",
    "legacy_sale_price_gp",
    "legacy_component_1",
    "legacy_component_1_qty",
    "legacy_component_1_source",
    "legacy_component_2",
    "legacy_component_2_qty",
    "legacy_component_2_source",
    "legacy_component_3",
    "legacy_component_3_qty",
    "legacy_component_3_source",
    "legacy_components_json",
]

with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for i, recipe in enumerate(recipes, start=1):
        comps = recipe.get("components") or []
        row = {
            "legacy_row_id": i,
            "legacy_profession": recipe.get("profession"),
            "legacy_secondary_profession": recipe.get("secondary_profession"),
            "legacy_tier": recipe.get("tier"),
            "legacy_rarity": recipe.get("rarity"),
            "legacy_name": recipe.get("name"),
            "legacy_category": recipe.get("category"),
            "legacy_craft_type": recipe.get("craft_type"),
            "legacy_description": recipe.get("description"),
            "legacy_use": recipe.get("use"),
            "legacy_output_qty": recipe.get("output_qty"),
            "legacy_base_price_gp": recipe.get("base_price_gp"),
            "legacy_vendor_price_gp": recipe.get("vendor_price_gp"),
            "legacy_sale_price_gp": recipe.get("sale_price_gp"),
            "legacy_components_json": json.dumps(comps, ensure_ascii=False, separators=(",", ":")),
        }
        for idx in range(3):
            comp = comps[idx] if idx < len(comps) else {}
            n = idx + 1
            row[f"legacy_component_{n}"] = comp.get("name")
            row[f"legacy_component_{n}_qty"] = comp.get("qty")
            row[f"legacy_component_{n}_source"] = comp.get("gathering_profession")
        writer.writerow(row)

print(f"Exported {len(recipes)} recipes to {OUT}")
