# sproutsai-candidate-rec-engine

link to streamlit: https://sproutsai-candidate-rec-engine-nyfbks5dmn2dwxjacsghjg.streamlit.app/ 

here are my thoughts and approaches on the problem:
  - i began by coding the file uploads and gathering data from each resume.
  - then, i used spaCy's NLP to help me extract the name from the resume. i also used regex for extracting emails and phone numbers.
  - next, i used scikit-learn's cosine_similarity function to compare the job description vector and resume vector. i also coded the logic for cosine_similarity (and left it commented!).
  - finally, i used the bart transformed model and pytorch to summarize the candidate's credentials.
  - i created multiple objects for different candidates and sorted the list by highest similarity score.

thoughts:
- the cosine similarity scores seem to be low (but checks out relatively) since the cosine similarity function compares the entire resume to the job description; so, even the irrelevant parts like education history are being compared to the job description, leading to a lower score. i tried to extract the important parts by cutting out education and filtering out key words. however, this could be improved more.
- my favorite part was implementing the bart model since it was fairly new to me.
- i believe the summaries could be better using LangChain framework.

thank you for testing my project! 
