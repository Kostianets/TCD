import math
import re

# -----------------------------------------------------------
# Implementácia jednoduchého Naive Bayes klasifikátora pre text
# -----------------------------------------------------------
class SimpleNaiveBayesClassifier:
    """
    Jednoduchý Naive Bayes klasifikátor pre textové dáta.
    Používa bag-of-words prístup s Laplaceovým vyhladzovaním.
    """
    def __init__(self, alpha=1.0):
        """
        Parameters
        ----------
        - alpha: parameter vyhladzovania
        - class_counts: počet výskytov jednotlivých tried
        - word_counts: slovníky s počtami slov pre každú triedu
        - total_words: celkový počet slov v každej triede
        - vocab: množina unikátnych slov vo všetkých dokument
        """
        self.alpha = alpha           # parameter vyhladzovania
        self.class_counts = {}       # počet výskytov jednotlivých tried
        self.word_counts = {}        # slovníky s počtami slov pre každú triedu
        self.total_words = {}        # celkový počet slov v každej triede
        self.vocab = set()           # množina unikátnych slov vo všetkých dokumentoch

    # Inside naive_bayes.py, add this method to the SimpleNaiveBayesClassifier class
    def _log_prob(self, word, label):
        if label not in self.word_counts:
            return math.log(self.alpha / (self.total_words.get(label, 0) + self.alpha * len(self.vocab)))
        word_count = self.word_counts[label].get(word, 0)
        total_words_label = self.total_words[label]
        return math.log((word_count + self.alpha) / (total_words_label + self.alpha * len(self.vocab)))

    def tokenize(self, text):
        """
        Tokenizácia textu: prevod na malé písmená, odstránenie interpunkcie a rozdelenie podľa medzier.

        Parameters
        ----------
        - text: vstupný text (komentár)

        Returns
        -------
        - text.split(): zoznam tokenov (slov)
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def fit(self, X, y):
        """
        Natrénuje Naive Bayes klasifikátor.
        
        Parameters
        ----------
        - X: zoznam textov (komentárov)
        - y: zoznam príslušných tried (napr. 0 - netoxický, 1 - toxický)
        """
        for text, label in zip(X, y):
            if label not in self.class_counts:
                self.class_counts[label] = 0
                self.word_counts[label] = {}
                self.total_words[label] = 0
            self.class_counts[label] += 1
            words = self.tokenize(text)
            for word in words:
                self.vocab.add(word)
                self.word_counts[label][word] = self.word_counts[label].get(word, 0) + 1
                self.total_words[label] += 1
        self.total_docs = len(X)
        self.class_priors = {label: count / self.total_docs for label, count in self.class_counts.items()}

    def predict(self, X):
        """
        Vykoná predikciu tried pre zoznam textov.
        
        Parameters
        ----------
        - X: zoznam textov
        
        Returns
        -------
        - predictions: zoznam predikovaných tried
        """
        predictions = []
        for text in X:
            words = self.tokenize(text)
            class_scores = {}
            for label in self.class_counts:
                score = math.log(self.class_priors[label])
                for word in words:
                    word_count = self.word_counts[label].get(word, 0)
                    score += math.log((word_count + self.alpha) / (self.total_words[label] + self.alpha * len(self.vocab)))
                class_scores[label] = score
            predicted = max(class_scores, key=class_scores.get)
            predictions.append(predicted)
        return predictions