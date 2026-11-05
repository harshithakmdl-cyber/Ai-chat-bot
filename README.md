# AI Chatbot

A simple Python chatbot that understands what you type and replies using Machine Learning.

## How it works
1. Sample questions ("patterns") are stored in `intents.json`.
2. **TF-IDF** converts text into numbers.
3. **Cosine similarity** finds the closest known question.
4. The bot replies with an answer for that topic. If nothing is close enough, it says it did not understand.

## Run
```bash
pip install -r requirements.txt
python chatbot.py
```

## Add your own topics
Edit `intents.json`: add a new `tag`, some `patterns` and `responses`.

## Tech
Python, Scikit-learn

Author: Harshitha N
