import pandas as pd
import json
import os
import re
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz, process

# Пути
DATA_DIR = "/Users/macbook/Projects/helper/app/data"
ENTITY_PATH = os.path.join(DATA_DIR, "HHRU_extracted_outputs_partial.csv")
DF_PATH = os.path.join(DATA_DIR, "HHRU_Объединённый_датасет.csv")
CLUSTERS_PATH = os.path.join(DATA_DIR, "final_clusters_deduped.json")

# Загрузка
entity_df = pd.read_csv(ENTITY_PATH)
df = pd.read_csv(DF_PATH)
entity_df = entity_df[entity_df['extracted_entities'].str.strip() != '{}'].reset_index(drop=True)
df = df.iloc[:len(entity_df)].reset_index(drop=True)
entity_df['extracted_entities'] = entity_df['extracted_entities'].str.replace("\\n|```|json| in | of |degree", "", regex=True)

# Загрузка структуры кластеров
with open(CLUSTERS_PATH, 'r', encoding='utf-8') as f:
    clusters_data = json.load(f)

# Очистка навыков
def fuzzy_clean_skills(skills_list, threshold=85):
    cleaned_skills, mapping = [], {}
    for skill in skills_list:
        norm = ' '.join(skill.lower().strip().split())
        if cleaned_skills:
            result = process.extractOne(norm, cleaned_skills, scorer=fuzz.token_sort_ratio)
            if result:
                match, score, _ = result
                mapping[skill] = match if score >= threshold else norm
                if score < threshold:
                    cleaned_skills.append(norm)
            else:
                mapping[skill] = norm
                cleaned_skills.append(norm)
        else:
            cleaned_skills.append(norm)
            mapping[skill] = norm
    return mapping

# TF-IDF
def compute_tfidf_weights(skill_docs):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(skill_docs)
    feature_names = vectorizer.get_feature_names_out()
    weights = []
    for i in range(tfidf_matrix.shape[0]):
        tfidf_scores = dict(zip(feature_names, tfidf_matrix[i].toarray()[0]))
        weights.append(tfidf_scores)
    return weights

# Сбор всех скиллов и валидных индексов
skill_docs = []
all_skills_flat = []
valid_indices = []

for idx in range(len(entity_df)):
    raw = entity_df.loc[idx, 'extracted_entities']
    try:
        entities = eval(re.sub(r"(\w+)'s", r"\1s", raw.lower()).replace('null', 'None'))
    except:
        continue

    skills = []
    for items in entities.values():
        for item in items:
            if isinstance(item, str):
                skills.append(item)
                all_skills_flat.append(item)

    if skills:
        skill_docs.append(skills)
        valid_indices.append(idx)

# Очистка и веса
skill_mapping = fuzzy_clean_skills(all_skills_flat, threshold=70)
cleaned_docs = [[skill_mapping[s] for s in doc] for doc in skill_docs]
tfidf_weights = compute_tfidf_weights(cleaned_docs)

# Маппинг навыков на кластеры
skill_to_cluster = {}
for cluster_id, cluster in clusters_data.items():
    for sub_id, skills in cluster['subclusters'].items():
        for s in skills:
            skill_to_cluster[s.lower()] = (cluster_id, sub_id)

# Построение графа
G = nx.DiGraph()
print("🧠 Строим глобальный граф...")

for i, idx in enumerate(valid_indices):
    prof = df.loc[idx, 'search_keyword']
    G.add_node(prof, label=prof, type="profession")
    skills = cleaned_docs[i]
    weights = tfidf_weights[i]

    for s in skills:
        norm = s.lower()
        cluster_info = skill_to_cluster.get(norm)
        if not cluster_info:
            continue
        cluster_id, sub_id = cluster_info
        cluster_node = f"Кластер {cluster_id}"
        sub_node = f"Подкластер {sub_id}"

        G.add_node(cluster_node, label=cluster_node, type="cluster")
        G.add_node(sub_node, label=sub_node, type="subcluster")
        G.add_node(norm, label=norm, type="skill", weight=weights.get(norm, 0.1))

        G.add_edge(prof, cluster_node)
        G.add_edge(cluster_node, sub_node)
        G.add_edge(sub_node, norm, weight=weights.get(norm, 0.1))

# Визуализация
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.3, seed=42)
colors = []
for node, attr in G.nodes(data=True):
    if attr['type'] == 'profession':
        colors.append('red')
    elif attr['type'] == 'cluster':
        colors.append('orange')
    elif attr['type'] == 'subcluster':
        colors.append('skyblue')
    else:
        colors.append('lightgreen')

nx.draw_networkx_nodes(G, pos, node_color=colors, alpha=0.7, node_size=80)
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_labels(G, pos, font_size=6)
plt.title("Глобальный граф: профессии → кластеры → подкластеры → навыки")
plt.axis('off')
plt.tight_layout()
plt.show()
