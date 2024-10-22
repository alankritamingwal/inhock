from flask import Flask, render_template, request
import json
import requests
import joblib
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Load embeddings
sent_bertphrase_embeddings = joblib.load('model/questionembedding.dump')
sent_bertphrase_ans_embeddings = joblib.load('model/ansembedding.dump')

# Load stop words
stop_w = stopwords.words('english')

# Load FAQ DataFrame
df = pd.read_csv("model/20200325_counsel_chat.csv", encoding="utf-8")

# Initialize lemmatizer
lmtzr = WordNetLemmatizer()

def get_embeddings(texts):
    url = '5110e4cb8b63.ngrok.io'  # Change to your embedding service URL
    headers = {'content-type': 'application/json'}
    data = {
        "id": 123,
        "texts": texts,
        "is_tokenized": False
    }
    data = json.dumps(data)

    try:
        r = requests.post(f"http://{url}/encode", data=data, headers=headers)
        r.raise_for_status()  # Raise an error for bad responses
        response_json = r.json()
        print(f"Embedding response: {response_json}")  # Debug statement
        return response_json.get('result', [])
    except requests.exceptions.RequestException as e:
        print("Error during request:", e)
        return []  # Return an empty list on error
    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        return []

def clean(column, df, stopwords=False):
    df[column] = df[column].apply(str)
    df[column] = df[column].str.lower().str.split()
    if stopwords:
        df[column] = df[column].apply(lambda x: [item for item in x if item not in stop_w])
    df[column] = df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
    df[column] = df[column].apply(lambda x: " ".join(x))

def retrieve_and_print_faq_answer(question_embedding, sentence_embeddings, FAQdf):
    max_sim = -1
    index_sim = -1
    valid_ans = []

    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        if sim >= max_sim:
            max_sim = sim
            index_sim = index
            valid_ans.append(index_sim)

    max_a_sim = -1
    answer = ""
    for ans in valid_ans:
        answer_text = FAQdf.iloc[ans, 8]  # Answer
        answer_em = sent_bertphrase_ans_embeddings[ans]
        similarity = cosine_similarity(answer_em, question_embedding)[0][0]
        if similarity > max_a_sim:
            max_a_sim = similarity
            answer = answer_text

    if max_a_sim < 0.70:
        return "Could you please elaborate your situation more? I don't really understand."
    return answer

def clean_text(greetings):
    greetings = greetings.lower()
    greetings = ' '.join(word.strip(string.punctuation) for word in greetings.split())
    return lmtzr.lemmatize(greetings)

def predictor(user_text):
    data = [user_text]
    x_try = pd.DataFrame(data, columns=['text'])
    clean('text', x_try, stopwords=True)

    for index, row in x_try.iterrows():
        question = row['text']
        question_embedding = get_embeddings([question])
        print(f"Question: {question}, Embedding: {question_embedding}")  # Debug statement

        if not question_embedding:
            return "I couldn't understand your question. Can you please rephrase it?"

        answer = retrieve_and_print_faq_answer(question_embedding, sent_bertphrase_embeddings, df)
        print(f"Retrieved Answer: {answer}")  # Debug statement
        return answer

greetings = ['hi', 'hey', 'hello', 'heyy', 'good evening', 'good morning', 'good afternoon', 'good', 'fine', 'okay', 'great']
happy_emotions = ['i feel good', 'life is good', 'life is great', "i've had a wonderful day", "i'm doing good"]
goodbyes = ['thank you', 'bye', 'thanks and bye', 'goodbye', 'see ya later']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    print(f"User input: {user_text}")  # Debug statement
    clean_text_input = clean_text(str(user_text))
    blob = TextBlob(user_text)
    polarity = blob.sentiment.polarity

    if clean_text_input in greetings:
        return "Hello! How may I help you today?"
    elif polarity > 0.7:
        return "That's great! Do you still have any questions for me?"
    elif clean_text_input in happy_emotions:
        return "That's great! Do you still have any questions for me?"  
    elif clean_text_input in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"

    topic = predictor(user_text)
    return topic

if __name__ == "__main__":
    app.run(debug=True)
