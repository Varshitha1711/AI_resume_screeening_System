from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(resume, jd):
    embeddings = model.encode([resume, jd], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score) * 100


def skill_match(resume_skills, jd_skills):
    if len(jd_skills) == 0:
        return 0
    matched = set(resume_skills).intersection(set(jd_skills))
    return len(matched) / len(jd_skills) * 100


def missing_skills(resume_skills, jd_skills):
    return list(set(jd_skills) - set(resume_skills))