import streamlit as st
from src.pdf_extractor import extract_text_from_pdf
from src.preprocess import clean_text
from src.skill_extractor import extract_skills
from src.matcher import compute_similarity, skill_match, missing_skills
from src.recommender import recommend_jobs
from src.predict import predict_category

# Page config
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="centered"
)

# Header
st.markdown("<h1 style='text-align: center;'>📄 AI Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Smart Resume Analysis using NLP & Machine Learning</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.info(
    "This system analyzes resumes using AI.\n\n"
    "✔ Resume Classification\n"
    "✔ Job Matching (BERT)\n"
    "✔ Skill Analysis\n"
    "✔ Job Recommendations"
)

st.sidebar.markdown("---")
st.sidebar.write("💡 Tip: Use ML Engineer JD for best results")

# Inputs
st.markdown("### 📤 Upload Resume")
uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

st.markdown("### 📝 Job Description")
job_description = st.text_area("Paste Job Description here...", height=150)

# Analyze Button
if st.button("🚀 Analyze Resume"):

    if uploaded_file and job_description:

        # Progress bar
        progress = st.progress(0)

        with st.spinner("🔍 Analyzing Resume..."):

            # Step 1: Extract text
            resume_text = extract_text_from_pdf(uploaded_file)
            progress.progress(20)

            # Step 2: Clean
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(job_description)
            progress.progress(40)

            # Step 3: Skills
            resume_skills = extract_skills(resume_clean)
            jd_skills = extract_skills(jd_clean)
            progress.progress(60)

            # Step 4: Scores
            semantic_score = compute_similarity(resume_clean, jd_clean)
            skill_percent = skill_match(resume_skills, jd_skills)
            final_score = (0.7 * semantic_score) + (0.3 * skill_percent)
            progress.progress(80)

            # Step 5: Prediction
            predicted_role = predict_category(resume_clean)
            missing = missing_skills(resume_skills, jd_skills)
            job_recs = recommend_jobs(resume_clean)
            progress.progress(100)

        st.success("✅ Analysis Complete!")

        st.markdown("---")

        # 📊 SCORES
        st.subheader("📊 Match Scores")

        col1, col2, col3 = st.columns(3)

        col1.metric("Final Score", f"{final_score:.2f}%")
        col2.metric("Semantic Score", f"{semantic_score:.2f}%")
        col3.metric("Skill Match", f"{skill_percent:.2f}%")

        st.progress(int(final_score))

        # 🎯 Recommendation
        if final_score >= 75:
            recommendation = "✅ Strong Match"
        elif final_score >= 60:
            recommendation = "⚠️ Moderate Match"
        else:
            recommendation = "❌ Low Match"

        st.markdown("### 🎯 Recommendation")
        st.write(recommendation)

        # 🧠 Predicted Role
        st.markdown("### 🧠 Predicted Job Role")
        st.success(predicted_role)

        # ✅ Skills
        st.markdown("### ✅ Extracted Skills")
        if resume_skills:
            st.write(", ".join(resume_skills))
        else:
            st.write("No skills detected")

        # ⚠️ Missing Skills
        st.markdown("### ⚠️ Missing Skills")
        if missing:
            st.write(", ".join(missing))
        else:
            st.write("None 🎉")

        # 💼 Job Recommendations
        st.markdown("### 💼 Top Job Recommendations")
        for job, score in job_recs[:3]:
            st.write(f"🔹 {job} → {score:.2f}% match")

        st.markdown("---")

    else:
        st.warning("⚠️ Please upload a resume and enter job description")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)