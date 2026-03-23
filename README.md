# AI Resume Screening and Job Recommendation System

This project is a machine learning-based system that helps automate resume screening and job matching. It analyzes resumes, compares them with job descriptions, and provides a match score along with useful insights such as skill gaps and role predictions.

## Overview

The system takes a resume (PDF) and a job description as input. It processes the text, extracts skills, and uses both machine learning and NLP techniques to evaluate how well the resume matches the job.

It also predicts the most suitable job role for the candidate and suggests other relevant roles.

## Features

* Resume parsing from PDF files
* Text preprocessing and cleaning
* Skill extraction and matching
* Resume classification using a trained ML model
* Semantic similarity using BERT
* Match score calculation
* Missing skill identification
* Job recommendations
* Simple web interface using Streamlit

## Project Structure

```
ai-resume-screening/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── assets/
│   └── skills_list.json
│
└── src/
    ├── pdf_extractor.py
    ├── preprocess.py
    ├── skill_extractor.py
    ├── matcher.py
    ├── recommender.py
    ├── train_model.py
    └── predict.py
```

## Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* NLTK
* Sentence Transformers (BERT)
* Streamlit

## Setup Instructions

1. Clone the repository

```
git clone https://github.com/your-username/ai-resume-screening-system.git
cd ai-resume-screening-system
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Train the model

```
python src/train_model.py
```

4. Run the application

```
streamlit run app.py
```

## Output

The system provides:

* Final match score
* Semantic similarity score
* Skill match percentage
* Predicted job role
* Missing skills
* Recommendation (strong, moderate, or low match)
* Suggested job roles

## Notes

The model is trained using a structured resume dataset. Since the dataset is not in plain text format, multiple fields such as skills, objectives, and responsibilities are combined to represent a resume during training.

## Future Improvements

* Improve skill extraction using advanced NLP
* Add experience-based scoring
* Generate downloadable reports

