import streamlit as st
import plotly.graph_objects as go
import io
import pickle
import networkx as nx
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
import pandas as pd
from app.utils import (
    read_resume_pdf, read_txt, load_skills_json,
    extract_entities_with_skills, extract_emails, extract_phone_number,
    extract_education, extract_career_titles, extract_locations,
    has_photo_in_resume, extract_name, clean_skills, final_clean_skills,
    estimate_experience_years, map_experience_to_category,
    process_vacancy, compare_resume_to_vacancy, calculate_combined_similarity,
    calculate_tfidf_similarity, generate_resume_recommendations, generate_roadmap_for_profession
)
# --- Загружаем граф ---
GRAPH_PATH = "/Users/macbook/Projects/helper/app/data/job_graphs_with_meta.pkl"

with open(GRAPH_PATH, "rb") as f:
    job_graphs = pickle.load(f)

# --- Объединяем ---
def combine_graphs(graphs):
    G = nx.Graph()
    for g in graphs.values():
        G.add_nodes_from(g.nodes(data=True))
        G.add_edges_from(g.edges(data=True))
    return G

G_filtered = combine_graphs(job_graphs)

# Проверяем все узлы и выбираем только с type в ['skill', 'skills']
top_skills = sorted(
    [
        (node, G_filtered.nodes[node].get('weight', 0))
        for node in G_filtered.nodes
        if G_filtered.nodes[node].get('type', '').lower() in ['skill', 'skills']
    ],
    key=lambda x: x[1],
    reverse=True
)[:20]


df_top_skills = pd.DataFrame(top_skills, columns=["skill", "tfidf_weight"])

fig_bar = px.bar(
    df_top_skills,
    x="skill", y="tfidf_weight",
    labels={"skill": "Навык", "tfidf_weight": "TF-IDF вес"},
    title="Топ-20 навыков по TF-IDF весу",
)
fig_bar.update_layout(xaxis={'categoryorder':'total descending'})


# --- Pyvis визуализация графа ---
def generate_full_graph(G_filtered):
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
    for node, data in G_filtered.nodes(data=True):
        node_type = data.get('type', 'skill')
        color = {
            'profession': 'skyblue',
            'skill': 'lightgreen',
            'education': 'lightcoral',
            'experience': 'violet'
        }.get(node_type, 'grey')
        shape = {
            'profession': 'dot',
            'skill': 'box',
            'education': 'box',
            'experience': 'triangle'
        }.get(node_type, 'dot')
        size = 15 + G_filtered.degree(node) * 2
        net.add_node(node, label=node, color=color, size=size, shape=shape)
    for source, target, data in G_filtered.edges(data=True):
        label = data.get('label', '')
        net.add_edge(source, target, title=label)
    net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.9)
    return net

# Временный HTML-файл для Pyvis
with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
    tmp_path = f.name
    net = generate_full_graph(G_filtered)
    net.save_graph(tmp_path)

# === Настройки страницы ===
st.set_page_config(page_title="Карьерный помощник", layout="wide")
tab1, tab2, tab3 = st.tabs(["🔍 Сравнение резюме и вакансии", "🧠 Граф знаний и дашборд", "🗺️ Дорожная карта развития"])

# === Основная функция ===
def main():
    with tab1:
        st.title("💼 Карьерный помощник")
        st.write("Загрузите своё резюме и описание вакансии для анализа.")

        # === Загрузка данных ===
        skills_path = "app/data/skills.json"
        all_skills = load_skills_json(skills_path)

        resume_file = st.file_uploader("📄 Загрузите резюме (PDF)", type="pdf")
        summary_file = st.file_uploader("📝 Загрузите краткое описание вакансии (TXT)", type="txt")
        vacancy_file = st.file_uploader("📃 Загрузите полное описание вакансии (TXT)", type="txt")

        if resume_file and summary_file and vacancy_file:
            # Чтение файлов
            resume_text = read_resume_pdf(resume_file)
            summary_text = read_txt(summary_file)
            vacancy_text = read_txt(vacancy_file)

            # Обработка резюме
            resume_entities = extract_entities_with_skills(resume_text, all_skills)
            emails = extract_emails(resume_text)
            phone_number = extract_phone_number(resume_text)
            education_info = extract_education(resume_text)
            career_titles = extract_career_titles(resume_text)
            found_locations = extract_locations(resume_text)
            name = extract_name(resume_text)
            cleaned_skills = clean_skills(resume_entities['SKILLS'], all_skills)
            final_skills = final_clean_skills(cleaned_skills, [])
            experience_years = estimate_experience_years(resume_text)
            experience_category = map_experience_to_category(experience_years)
            photo = has_photo_in_resume(resume_file)

            # Сборка начальной структуры
            structured_resume = {
                "name": name,
                "email": emails[0] if emails else None,
                "phone": phone_number,
                "skills": final_skills,
                "education": list(education_info),
                "locations": list(found_locations),
                "career_titles": list(career_titles),
                "experience_category": experience_category,
                "has_photo": photo
            }

            # ===== Блок редактирования атрибутов резюме =====

            st.subheader("📝 Проверьте извлеченные данные из вашего резюме:")
            st.markdown(f"**👤 Имя:** {structured_resume['name']}")
            st.markdown(f"**📧 Email:** {structured_resume['email']}")
            st.markdown(f"**📞 Телефон:** {structured_resume['phone']}")
            st.markdown("<div style='background-color:#e0f7fa;padding:10px;border-radius:10px'>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

            # Разные цветные секции для разных атрибутов
            with st.container():
                st.markdown("### 🎯 Профессии")
                edited_career_titles = st.multiselect(
                    "Отметьте корректные профессии:",
                    options=structured_resume.get("career_titles") or [],
                    default=structured_resume.get("career_titles") or [],
                )

            with st.container():
                st.markdown("### 🏙️ Локации")
                edited_locations = st.multiselect(
                    "Отметьте корректные города:",
                    options=structured_resume.get("locations") or [],
                    default=structured_resume.get("locations") or [],
                )

            with st.container():
                st.markdown("### 📚 Навыки")
                edited_skills = st.multiselect(
                    "Отметьте корректные навыки:",
                    options=structured_resume.get("skills") or [],
                    default=structured_resume.get("skills") or [],
                )

            with st.container():
                st.markdown("### 🎓 Образование")
                edited_education = st.multiselect(
                    "Отметьте корректные образовательные учреждения:",
                    options=structured_resume.get("education") or [],
                    default=structured_resume.get("education") or [],
                )

            with st.container():
                st.markdown("### ⏳ Опыт работы")
                experience_options = ["Нет опыта", "1-3 года", "От 3 до 6 лет", "Более 6 лет"]
                edited_experience = st.selectbox(
                    "Выберите корректный опыт работы:",
                    options=experience_options,
                    index=experience_options.index(structured_resume.get("experience_category")) if structured_resume.get("experience_category") in experience_options else 0
                )

            with st.container():
                st.markdown("### 🖼️ Фото в резюме")
                edited_photo = st.checkbox(
                    "Есть фото в резюме?",
                    value=structured_resume.get("has_photo", False)
                )

            # Кнопка подтверждения
            if st.button("🚀 Продолжить обработку"):
                # Обновляем structured_resume на основе пользовательских правок
                structured_resume["career_titles"] = edited_career_titles
                structured_resume["locations"] = edited_locations
                structured_resume["skills"] = edited_skills
                structured_resume["education"] = edited_education
                structured_resume["experience_category"] = edited_experience
                structured_resume["has_photo"] = edited_photo

                st.success("✅ Данные успешно обновлены! Продолжаем анализ...")

                # Обработка вакансии
                structured_vacancy = process_vacancy(vacancy_text, all_skills)

                # Сравнение резюме и вакансии
                comparison_result = compare_resume_to_vacancy(structured_resume, structured_vacancy)
                skills_match_percent = comparison_result['skill_match_percent']
                tfidf_similarity = calculate_tfidf_similarity(resume_text, summary_text)
                final_similarity = calculate_combined_similarity(skills_match_percent, tfidf_similarity)

                # Блок красивого вывода сравнения
                st.header("📊 Результаты анализа")
                st.subheader("✍️ Схожесть вашего резюме и вакансии")
                # Разделение на две колонки
                col1, col2 = st.columns(2)

                with col1:
                    st.header("")
                    st.metric("📚 Skills Match", f"{skills_match_percent:.2f}%")
                    st.metric("🔍 TF-IDF Similarity", f"{tfidf_similarity * 100:.2f}%")

                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=final_similarity * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "🏆 Итоговая оценка, %"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#333333"},
                            'steps': [
                                {'range': [0, 30], 'color': "#FFCCCC"},   # Красный
                                {'range': [30, 70], 'color': "#FFFF99"},  # Желтый
                                {'range': [70, 100], 'color': "#99FF99"}  # Зеленый
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                recommendations = generate_resume_recommendations(structured_resume)
                if recommendations:
                    st.subheader("📢 Рекомендации по улучшению резюме")
                    for rec in recommendations:
                        st.write(f"• {rec}")
                else:
                    st.success("✅ Ваше резюме выглядит полно и информативно!")
                
                st.subheader("📚 Навыки, которых не хватает")

                # Извлекаем навыки, которых не хватает
                unmatched_skills = comparison_result.get('unmatched_skills', [])

                if unmatched_skills:
                    # Мультиселект для выбора навыков
                    unmatched_skills_selected = st.multiselect(
                        "📚 Навыки, которых не хватает:",
                        unmatched_skills,
                        default=unmatched_skills,
                        key="unmatched_skills"
                    )

                    # Кнопка "Сохранить навыки"
                    if st.button("🚀 Сохранить навыки", key="save_unmatched_skills"):
                        unmatched_skills = unmatched_skills_selected  # Только после нажатия кнопки обновляем данные
                        st.success("✅ Навыки для доработки сохранены!")

                else:
                    st.success("✅ Все ключевые навыки вакансии указаны в резюме!")
    with tab2:
        st.subheader("👀 Дашборд")
        st.markdown("🔎 Фильтры")

        # Собираем списки уникальных значений
        companies = sorted(set(nx.get_node_attributes(G_filtered, "company").values()))
        cities = sorted(set(nx.get_node_attributes(G_filtered, "city").values()))
        experiences = sorted(set(nx.get_node_attributes(G_filtered, "experience_category").values()))
        published_dates = sorted(set(nx.get_node_attributes(G_filtered, "published_at").values()))
        professions = sorted([node for node, d in G_filtered.nodes(data=True) if d.get("type") == "profession"])

        # Компактные фильтры
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_company = st.selectbox("Компания", ["Все"] + companies)
            with col2:
                selected_city = st.selectbox("Город", ["Все"] + cities)
            with col3:
                selected_experience = st.selectbox("Опыт", ["Все"] + experiences)

        with st.container():
            col4, col5 = st.columns(2)
            with col4:
                selected_date = st.selectbox("Дата публикации", ["Все"] + published_dates)
            with col5:
                selected_profession = st.selectbox("Профессия", ["Все"] + professions)


        # Функция фильтрации узлов
        def filter_nodes(node, data):
            if data.get('type') == 'profession':
                if selected_profession != "Все" and node != selected_profession:
                    return False
                if selected_company != "Все" and data.get('company') != selected_company:
                    return False
                if selected_city != "Все" and data.get('city') != selected_city:
                    return False
                if selected_experience != "Все" and data.get('experience_category') != selected_experience:
                    return False
                if selected_date != "Все" and data.get('published_at') != selected_date:
                    return False
            return True

        # Применяем фильтрацию
        filtered_nodes = {n: d for n, d in G_filtered.nodes(data=True) if filter_nodes(n, d)}
        G_subgraph = G_filtered.subgraph(filtered_nodes.keys()).copy()

        col1, col2 = st.columns([2, 1]) 

        with col1:
            st.subheader("🏆 Топ-20 навыков после фильтрации")
            if not df_top_skills.empty:
                fig_bar.update_layout(height=600)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("⚠️ Нет навыков для отображения после фильтрации!")

        with col2:
            with col2:
                st.subheader("📄 Сводка по данным")

                # Общие числа
                st.metric("🔗 Всего узлов", len(G_subgraph.nodes()))
                st.metric("🧩 Всего связей", len(G_subgraph.edges()))

                # Уникальные значения
                professions_count = len({n for n, d in G_subgraph.nodes(data=True) if d.get("type") == "profession"})
                cities_count = len(set(nx.get_node_attributes(G_subgraph, "city").values()))
                companies_count = len(set(nx.get_node_attributes(G_subgraph, "company").values()))
                skills_count = len([n for n, d in G_subgraph.nodes(data=True) if d.get("type") in ["skill", "skills"]])

                st.markdown(f"👔 Кол-во профессий: **{professions_count}**")
                st.markdown(f"🏙️ Кол-во городов: **{cities_count}**")
                st.markdown(f"🏢 Кол-во компаний: **{companies_count}**")
                st.markdown(f"🛠️ Кол-во навыков в графе: **{skills_count}**")

                # Распределение по опыту
                experience_distribution = pd.Series(nx.get_node_attributes(G_subgraph, "experience_category").values())
                if not experience_distribution.empty:
                    st.markdown("📊 Распределение по опыту:")
                    st.dataframe(experience_distribution.value_counts().rename_axis("Категория").reset_index(name="Количество"))

                # Все навыки (по json)
                skills_path = "app/data/skills.json"
                all_skills = load_skills_json(skills_path)
                st.markdown(f"📚 Всего уникальных навыков в словаре: **{len(all_skills)}**")

        st.subheader("👀 Здесь будут еще разные графики")
        # === Строим граф в Pyvis ===
        st.subheader("🌐 Граф знаний")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
            net = generate_full_graph(G_subgraph)
            tmp_path = tmp_file.name
            net.save_graph(tmp_path)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_graph = f.read()

        st.components.v1.html(html_graph, height=800, scrolling=True)

        os.unlink(tmp_path)
    with tab3:
        st.title("🗺️ Дорожная карта профессионального развития")
        st.markdown("Выберите профессию и отметьте, какие навыки у вас уже есть. Или загрузите резюме — и мы сделаем это за вас.")

        # === 1. Выбор профессии ===
        professions = sorted([n for n, d in G_filtered.nodes(data=True) if d.get("type") == "profession"])
        selected_profession = st.selectbox("🔍 Выберите профессию", professions)

        # === 2. Загрузка резюме (необязательно) ===
        resume_file = st.file_uploader("📄 Загрузите резюме (PDF)", type="pdf", key="resume_roadmap")

        user_skills = set()
        if resume_file:
            resume_text = read_resume_pdf(resume_file)
            skills_path = "app/data/skills.json"
            all_skills = load_skills_json(skills_path)
            resume_entities = extract_entities_with_skills(resume_text, all_skills)
            cleaned_skills = clean_skills(resume_entities['SKILLS'], all_skills)
            final_skills = final_clean_skills(cleaned_skills, [])
            user_skills = set(final_skills)
            st.success(f"✅ Извлечено {len(user_skills)} навыков из резюме!")

        # === 3. Навыки, связанные с профессией ===
        st.subheader("📌 Навыки, связанные с выбранной профессией")
        related_skills = []
        for neighbor in G_filtered.neighbors(selected_profession):
            node_data = G_filtered.nodes[neighbor]
            if node_data.get("type") in ["skill", "skills"]:
                related_skills.append({
                    "Навык": neighbor,
                    "TF-IDF вес": node_data.get("weight", 0),
                    "Уже в резюме": neighbor in user_skills
                })

        df = pd.DataFrame(related_skills).sort_values(by="TF-IDF вес", ascending=False)

        # === 4. Отметка галочками (с предзаполнением из резюме) ===
        selected_skills = st.multiselect(
            "✅ Отметьте навыки, которые у вас уже есть:",
            options=df["Навык"].tolist(),
            default=[row["Навык"] for row in related_skills if row["Уже в резюме"]]
        )

        # Добавим колонку "Освоено" по галочкам
        df["Освоено"] = df["Навык"].apply(lambda x: "✅ Да" if x in selected_skills else "—")

        st.subheader("🛤️ Визуальная цепочка навыков")

        sorted_skills = df.sort_values(by="TF-IDF вес", ascending=False)["Навык"].tolist()

        # Рендерим дорожку
        roadmap_html = "<div style='display:flex;flex-wrap:wrap;align-items:center;margin-top:20px;'>"

        for i, skill in enumerate(sorted_skills):
            has_skill = skill in selected_skills
            bg_color = "#d4edda" if has_skill else "#f8f9fa"
            border = "2px solid #28a745" if has_skill else "1px solid #ccc"
            text_color = "#155724" if has_skill else "#333"

            step_html = f"""
            <div style='
                padding:10px 15px;
                margin:5px;
                background:{bg_color};
                border:{border};
                border-radius:20px;
                color:{text_color};
                font-weight:500;
                font-size:14px;
            '>
                {'✅ ' if has_skill else '➕ '}Шаг {i+1}: {skill}
            </div>
            """

            roadmap_html += step_html

            if i < len(sorted_skills) - 1:
                roadmap_html += "<div style='margin:5px;font-size:18px;'>➡️</div>"

        roadmap_html += "</div>"

        st.markdown(roadmap_html, unsafe_allow_html=True)
        st.subheader("🛠 Персональный план развития:")
        st.dataframe(df[["Навык", "TF-IDF вес", "Освоено"]].reset_index(drop=True), width=1000)

        # === 5. Выгрузка в CSV ===
        csv_buffer = io.StringIO()
        df_to_save = df[["Навык", "TF-IDF вес", "Освоено"]]
        df_to_save.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Скачать план развития в CSV",
            data=csv_buffer.getvalue(),
            file_name=f"roadmap_{selected_profession}.csv",
            mime="text/csv"
        )



    

if __name__ == "__main__":
    main()
