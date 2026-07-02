# merge_results.py
import json

nano  = json.load(open("results/map_nano.json"))
small = json.load(open("results/map_small.json"))

merged = {**nano, **small}

with open("results/map_results.json", "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("✅ 합치기 완료")
print("모델:", list(merged.keys()))