from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def resume_matcher(resume_text, job_description):
    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)


resume_text = """
Computer Science student skilled in Python, Machine Learning,
SQL, Data Analysis, and Flask API development.
"""

job_description = """
Looking for a frontend developer with experience in React.js,
SQL, next.js, and web development.
"""

score = resume_matcher(resume_text, job_description)
print("Resume Match Score:", score, "%")
