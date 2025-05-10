# app/utils.py

import os
import re
import json
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#numpy==1.26.4

nlp = spacy.load('ru_core_news_md')

# Важные короткие навыки (не используется по итогу)
important_short_skills = []


def read_resume_pdf(filepath: str) -> str:
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def load_skills_json(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8') as file:
        skills = json.load(file)
    return skills

def extract_entities_with_skills(text: str, skills_list: list) -> dict:
    doc = nlp(text)
    entities = {'ORG': [], 'DATE': [], 'PERSON': [], 'GPE': [], 'SKILLS': []}

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text.strip())

    lower_text = text.lower()
    for skill in skills_list:
        if skill.lower() in lower_text:
            entities['SKILLS'].append(skill)

    return entities

EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
def extract_emails(resume_text: str) -> list:
    return re.findall(EMAIL_REG, resume_text)

PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
def extract_phone_number(resume_text: str) -> str:
    phone = re.findall(PHONE_REG, resume_text)
    if phone:
        number = ''.join(phone[0])
        if len(number) < 16:
            return number
    return None

def estimate_experience_years(text: str) -> int:
    years = re.findall(r'\b(19[8-9]\d|20[0-2]\d|2023|2024)\b', text)
    years = [int(year) for year in years]

    if not years:
        return None

    min_year = min(years)
    max_year = max(years)

    experience = max_year - min_year
    if experience < 0:
        return None
    return experience

def map_experience_to_category(experience_years: int) -> str:
    if experience_years is None:
        return "Нет опыта"
    elif experience_years < 1:
        return "Нет опыта"
    elif 1 <= experience_years < 3:
        return "1-3 года"
    elif 3 <= experience_years <= 6:
        return "От 3 до 6 лет"
    else:
        return "Более 6 лет"

education_keywords = [
    'университет', 'институт', 'академия', 'колледж' 'college',
    'university', 'faculty', 'факультет'
]
def extract_education(text: str) -> set:
    found_education = set()
    lines = text.lower().split('\n')
    for line in lines:
        for keyword in education_keywords:
            if keyword in line:
                found_education.add(line.strip())
    return found_education

career_keywords = [
    'аналитик', 'бизнес-аналитик', 'data scientist',
    'programmer analyst',  'ml engineer',
    'data engineer', 'data analyst', 'product analyst', 'data science',
    'machine learning engineer', 'data architect', 'продуктовый аналитик'
]
def extract_career_titles(text: str) -> set:
    found_titles = set()
    lines = text.lower().split('\n')
    for line in lines:
        for keyword in career_keywords:
            if keyword in line:
                found_titles.add(keyword)
    return found_titles

russian_cities = [
    'москва', 'санкт-петербург', 'новосибирск', 'екатеринбург', 'казань',
    'нижний новгород', 'челябинск', 'самара', 'омск', 'ростов-на-дону',
    'уфа', 'красноярск', 'пермь', 'воронеж', 'волгоград', 'краснодар',
    'саратов', 'тюмень', 'тольятти', 'ижевск', 'барнаул', 'ульяновск',
    'иркутск', 'владивосток', 'ярославль', 'махачкала', 'томск', 'оренбург',
    'кемерово', 'новокузнецк', 'рязань', 'астрахань', 'пенза', 'липецк',
    'киров', 'чебоксары', 'тула', 'калининград', 'курск'
]
def extract_locations(text: str) -> set:
    found_cities = set()
    lower_text = text.lower()
    for city in russian_cities:
        if city in lower_text:
            found_cities.add(city)
    return found_cities

def has_photo_in_resume(filepath: str) -> bool:
    with pdfplumber.open(filepath) as pdf:
        if not pdf.pages:
            return False
        first_page = pdf.pages[0]
        if first_page.images:
            return True
    return False

def extract_name(text: str) -> str:
    lines = text.strip().split('\n')
    for line in lines[:5]:
        clean_line = line.strip()
        words = clean_line.split()

        if len(words) in [2, 3]:
            if all(word[0].isupper() for word in words if word.isalpha()):
                if not any(char.isdigit() for char in clean_line) and '@' not in clean_line:
                    return clean_line
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ('PER', 'PERSON'):
            if len(ent.text.split()) in [2, 3]:
                return ent.text.strip()
    return None

def clean_skills(skills_list: list, skills_dictionary: list) -> list:
    skills_list_lower = [skill.lower().strip() for skill in skills_list]
    skills_dictionary_lower = [skill.lower().strip() for skill in skills_dictionary]
    unique_skills = list(set(skills_list_lower))
    filtered_skills = [skill for skill in unique_skills if skill in skills_dictionary_lower]
    return filtered_skills

def final_clean_skills(skills_list: list, important_short_skills: list = important_short_skills) -> list:
    cleaned = []
    for skill in skills_list:
        if len(skill) > 2 or skill.lower() in important_short_skills:
            if not (skill.isdigit() and skill.lower() not in important_short_skills):
                cleaned.append(skill)
    return cleaned


def extract_experience_category_and_years(text: str) -> tuple:
    pattern = r"Требуемый опыт работы:\s*(.+)"
    match = re.search(pattern, text)

    experience_category = None
    experience_years = None

    if match:
        experience_category = match.group(1).strip()

        # Ищем числа в найденной строке
        years = re.findall(r'\d+', experience_category)

        if years:
            # Берем минимальное значение (например, 1 из "1–3 года")
            experience_years = int(years[0])

    return experience_category, experience_years

def process_vacancy(vacancy_text: str, skills_list: list) -> dict:
    career_titles = extract_career_titles(vacancy_text)
    found_locations = extract_locations(vacancy_text)

    extracted_skills = extract_entities_with_skills(vacancy_text, skills_list)
    cleaned_skills = clean_skills(extracted_skills['SKILLS'], skills_list)
    final_skills = final_clean_skills(cleaned_skills)

    experience_category, experience_years = extract_experience_category_and_years(vacancy_text)

    structured_vacancy = {
        "skills": final_skills if final_skills else None,
        "locations": list(found_locations) if found_locations else None,
        "career_titles": list(career_titles) if career_titles else None,
        "experience_category": experience_category,
        "experience_years": experience_years
    }

    return structured_vacancy

def read_txt(file):
    """Чтение текста из загруженного файла через Streamlit."""
    return file.read().decode('utf-8')


def process_vacancy(vacancy_text, skills_list):
    career_titles = extract_career_titles(vacancy_text)
    found_locations = extract_locations(vacancy_text)
    extracted_skills = extract_entities_with_skills(vacancy_text, skills_list)
    cleaned_skills = clean_skills(extracted_skills['SKILLS'], skills_list)
    final_skills = final_clean_skills(cleaned_skills, important_short_skills)
    experience_category, experience_years = extract_experience_category_and_years(vacancy_text)

    structured_vacancy = {
        "skills": final_skills if final_skills else None,
        "locations": list(found_locations) if found_locations else None,
        "career_titles": list(career_titles) if career_titles else None,
        "experience_category": experience_category,
        "experience_years": experience_years
    }
    
    return structured_vacancy

def generate_resume_recommendations(structured_resume: dict) -> list:
    recommendations = []

    if not structured_resume.get("has_photo"):
        recommendations.append("📷 Рекомендуется добавить фотографию в резюме — это может повысить доверие работодателя.")

    if not structured_resume.get("name"):
        recommendations.append("📝 Добавьте своё имя в начале резюме для персонализации.")

    if not structured_resume.get("email"):
        recommendations.append("📧 Укажите электронную почту для обратной связи.")

    if not structured_resume.get("phone"):
        recommendations.append("📞 Укажите номер телефона для оперативной связи.")

    if not structured_resume.get("skills"):
        recommendations.append("🛠️ Заполните раздел 'Навыки', чтобы подчеркнуть свои компетенции.")

    if not structured_resume.get("education"):
        recommendations.append("🎓 Укажите образование, особенно если оно профильное.")

    if not structured_resume.get("career_titles"):
        recommendations.append("🏷️ Рекомендуется указать должности/роли, которые вы занимали.")

    if structured_resume.get("experience_years") is None:
        recommendations.append("⌛ Добавьте информацию о своём опыте работы.")

    return recommendations


def compare_resume_to_vacancy(resume_data, vacancy_data):
    """Сравнение навыков, опыта и профессии резюме и вакансии."""
    result = {}

    resume_skills = set(resume_data.get('skills', []))
    vacancy_skills = set(vacancy_data.get('skills', []))

    matched_skills = resume_skills.intersection(vacancy_skills)
    unmatched_skills = vacancy_skills.difference(resume_skills)

    skill_match_percent = (len(matched_skills) / len(vacancy_skills)) * 100 if vacancy_skills else 0

    result['matched_skills'] = list(matched_skills)
    result['unmatched_skills'] = list(unmatched_skills)
    result['skill_match_percent'] = round(skill_match_percent, 2)

    resume_experience = resume_data.get('experience_years')
    vacancy_experience = vacancy_data.get('experience_years')

    if resume_experience is not None and vacancy_experience is not None:
        experience_ok = resume_experience >= vacancy_experience
    else:
        experience_ok = None

    result['experience_match'] = experience_ok
    result['resume_experience_years'] = resume_experience
    result['vacancy_experience_years'] = vacancy_experience

    resume_titles = set(resume_data.get('career_titles') or [])
    vacancy_titles = set(vacancy_data.get('career_titles') or [])

    title_match = bool(resume_titles.intersection(vacancy_titles))

    result['career_title_match'] = title_match

    return result

def calculate_tfidf_similarity(text1, text2):
    """Расчет TF-IDF cosine similarity между двумя текстами."""
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity, 4)

def calculate_combined_similarity(skill_match_percent: float, tfidf_similarity: float,
                                   weight_skills: float = 0.6, weight_tfidf: float = 0.4) -> float:
    """Рассчитывает комбинированную метрику Similarity на основе навыков и TF-IDF."""
    skill_match_score = skill_match_percent / 100  # переводим процент в [0,1]
    combined_score = (skill_match_score * weight_skills) + (tfidf_similarity * weight_tfidf)
    return round(combined_score, 4)
def generate_roadmap_for_profession(profession, user_experience, user_skills, G):
    roadmap = {"Начальный уровень": [], "Промежуточный уровень": [], "Продвинутый уровень": []}
    if profession not in G:
        return roadmap

    for neighbor in G.neighbors(profession):
        node_data = G.nodes[neighbor]
        node_type = node_data.get('type')
        importance = node_data.get('weight', 0)
        experience_required = node_data.get('experience_category', 'Нет опыта')

        if node_type not in ["skill", "education", "experience"]:
            continue

        # Определяем уровень
        if experience_required == "Нет опыта":
            level = "Начальный уровень"
        elif experience_required in ["1-3 года"]:
            level = "Промежуточный уровень"
        else:
            level = "Продвинутый уровень"

        roadmap[level].append({
            "название": neighbor,
            "тип": node_type,
            "важность": importance,
            "уже_есть": neighbor in user_skills
        })

    for level in roadmap:
        roadmap[level] = sorted(roadmap[level], key=lambda x: -x["важность"])
    return roadmap
