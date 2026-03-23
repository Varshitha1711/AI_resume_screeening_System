import json

with open("assets/skills_list.json") as f:
    SKILLS = json.load(f)

def extract_skills(text):
    found = set()
    for skill in SKILLS:
        if skill.lower() in text:
            found.add(skill)
    return list(found)