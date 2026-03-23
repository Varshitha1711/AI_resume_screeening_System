from src.matcher import compute_similarity

job_roles = {
    "Data Scientist": "python machine learning statistics data analysis pandas numpy",
    "ML Engineer": "python tensorflow pytorch ml deployment docker api",
    "Web Developer": "html css javascript react node",
    "Data Analyst": "excel sql powerbi dashboard"
}

def recommend_jobs(resume_text):
    scores = {}
    for role, desc in job_roles.items():
        scores[role] = compute_similarity(resume_text, desc)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)