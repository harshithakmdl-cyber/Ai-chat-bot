"""Simple AI chatbot: matches your message to the closest known intent
using TF-IDF (turns words into numbers) and cosine similarity (how close two texts are)."""
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.3  # below this, the bot says it did not understand


class Chatbot:
    def __init__(self, path="intents.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)["intents"]
        self.patterns, self.tags, self.responses = [], [], {}
        for intent in data:
            self.responses[intent["tag"]] = intent["responses"]
            for p in intent["patterns"]:
                self.patterns.append(p.lower())
                self.tags.append(intent["tag"])
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.patterns)

    def reply(self, message):
        vec = self.vectorizer.transform([message.lower()])
        scores = cosine_similarity(vec, self.matrix)[0]
        best = scores.argmax()
        if scores[best] < THRESHOLD:
            return "Sorry, I did not understand that. Type 'help' to see what I can do."
        return random.choice(self.responses[self.tags[best]])


if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot ready! Type 'bye' to exit.")
    while True:
        text = input("You: ").strip()
        if not text:
            continue
        print("Bot:", bot.reply(text))
        if text.lower() in ("bye", "goodbye", "exit", "quit"):
            break
