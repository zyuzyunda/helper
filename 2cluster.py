from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os


# Путь к файлу
GROUPED_SKILLS_PATH = "/Users/macbook/Projects/helper/app/data/grouped_skills_named.json"

# Загрузка
with open(GROUPED_SKILLS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
group_names = list(data.keys())

# Эмбеддинги
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(group_names, show_progress_bar=True)

# Масштабирование и UMAP
scaler = StandardScaler()
scaled = scaler.fit_transform(embeddings)
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
proj = umap_model.fit_transform(scaled)

# Кластеризация
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
labels = clusterer.fit_predict(proj)

# Визуализация
plt.figure(figsize=(12, 6))
plt.title("Верхнеуровневая кластеризация групп навыков (UMAP + HDBSCAN)")
plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab20', s=8)
plt.colorbar(label="Cluster ID")
plt.grid(True)
plt.tight_layout()
plt.show()

# Печать кластеров
cluster_map = defaultdict(list)
for name, label in zip(group_names, labels):
    cluster_map[label].append(name)

for cluster_id, names in cluster_map.items():
    print(f"\n🧠 Кластер {cluster_id} — {len(names)} групп:")
    for name in names[:10]:
        print(f"  • {name}")
    if len(names) > 10:
        print(f"  ...и ещё {len(names) - 10} групп")

print("\n🔍 Начинаем вложенную кластеризацию внутри верхних кластеров...")

for cluster_id, group_titles in cluster_map.items():
    print(f"\n📦 Вложенная кластеризация: верхний кластер {cluster_id} ({len(group_titles)} групп)")

    # Собираем все навыки из входящих групп
    raw_skills = []
    for group in group_titles:
        raw_skills.extend(data.get(group, []))
    
    # Убираем дубли
    raw_skills = list(set(raw_skills))
    if len(raw_skills) < 5:
        print(f"  ⚠️ Слишком мало навыков для кластеризации ({len(raw_skills)}). Пропускаем.")
        continue

    # Эмбеддинги
    skill_emb = model.encode(raw_skills, show_progress_bar=False)

    # UMAP
    scaled_skills = scaler.fit_transform(skill_emb)
    umap_skills = umap_model.fit_transform(scaled_skills)

    # HDBSCAN внутри группы
    inner_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    inner_labels = inner_clusterer.fit_predict(umap_skills)

    # Сбор вложенных кластеров
    inner_map = defaultdict(list)
    for skill, label in zip(raw_skills, inner_labels):
        inner_map[label].append(skill)

    # Вывод
    for inner_id, skills in inner_map.items():
        print(f"\n  🧩 Подкластер {inner_id} — {len(skills)} навыков:")
        for s in skills[:8]:
            print(f"   • {s}")
        if len(skills) > 8:
            print(f"   ...и ещё {len(skills) - 8}")


import os

# Путь для выгрузки
output_json_path = "./cluster_with_subclusters.json"
output_viz_dir = "./viz_clusters"
os.makedirs(output_viz_dir, exist_ok=True)

# Сборка структуры для графа знаний
cluster_results = {}

for cluster_id, group_titles in cluster_map.items():
    cluster_key = str(cluster_id)

    raw_skills = []
    for group in group_titles:
        raw_skills.extend(data.get(group, []))

    raw_skills = list(set(raw_skills))
    if len(raw_skills) < 5:
        continue

    skill_emb = model.encode(raw_skills, show_progress_bar=False)
    scaled_skills = scaler.fit_transform(skill_emb)
    umap_skills = umap_model.fit_transform(scaled_skills)
    inner_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    inner_labels = inner_clusterer.fit_predict(umap_skills)

    # Группировка навыков по подкластерам
    subclusters = defaultdict(list)
    for skill, label in zip(raw_skills, inner_labels):
        subclusters[str(label)].append(skill)

    # Сохраняем структуру
    # Финальная структура без старых групп
    cluster_results[cluster_key] = {
        "cluster_name": f"Cluster {cluster_key}",  # можно будет переименовать
        "subclusters": {
            str(sub_id): sorted(skills)
            for sub_id, skills in subclusters.items()
            if sub_id != -1 and len(skills) > 0
        }
    }


    # Визуализация
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(umap_skills[:, 0], umap_skills[:, 1], c=inner_labels, cmap="tab10", s=10)
    plt.title(f"Подкластеры внутри кластера {cluster_id}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_viz_dir, f"cluster_{cluster_id}.png"))
    plt.close()

# Сохраняем JSON
with open("/Users/macbook/Projects/helper/app/data/final_clusters.json", "w", encoding="utf-8") as f:
    json.dump(cluster_results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Результаты сохранены в {output_json_path}")
print(f"🖼️ Визуализации сохранены в {output_viz_dir}")
