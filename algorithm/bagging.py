import random

class BaggingClassifier:
    """
    Implementácia bagging algoritmu pre klasifikáciu.
    Vytvára viacero bootstrap vzoriek z trénovacích dát a pre každú vzorku
    natrénuje kópiu základného klasifikátora. Konečná predikcia je získaná hlasovaním.
    """
    def __init__(self, base_estimator, n_estimators=10, max_samples=None):
        """
        Parameters
        ----------
        - base_estimator: Trieda základného klasifikátora (napr. SimpleNaiveBayesClassifier),
                          ktorá musí mať metódy fit a predict.
        - n_estimators: Počet základných modelov.
        - max_samples: Počet vzoriek použitých pre každú bootstrap vzorku.
                       Ak None, použije sa celý počet trénovacích dát.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples  # ak None, nastavíme v metóde fit
        self.estimators = []  # zoznam natrénovaných základných modelov

    def _clone_estimator(self):
        """
        Vytvorí novú inštanciu základného klasifikátora.
        Predpokladáme, že base_estimator je trieda, ktorú možno volať bez argumentov.

        Returns
        -------
        - self.base_estimator(): nová inštancia základného klasifikátora
        """
        return self.base_estimator()

    # Inside bagging.py, add this method to the BaggingClassifier class
    def get_contributing_words(self, text, positive_class=1):
        words = self.estimators[0].tokenize(text)
        contributing = []
        for word in set(words):
            count = sum(1 for est in self.estimators if est._log_prob(word, positive_class) > est._log_prob(word, 1 - positive_class))
            if count > len(self.estimators) / 2:
                contributing.append(word)
        return contributing

    def fit(self, X, y):
        """
        Natrénuje bagging klasifikátor na trénovacích dátach X a y.

        Parameters
        ----------
        - X: Trénovacie dáta (zoznam textov).
        - y: Príslušné triedy (zoznam labelov).
        """
        n_samples = len(X)
        if self.max_samples is None:
            self.max_samples = n_samples
        
        self.estimators = []  # vyčistenie predchádzajúcich modelov
        for _ in range(self.n_estimators):
            # Vytvorenie bootstrap vzorky (náhodný výber s opakovaním)
            indices = [random.randint(0, n_samples - 1) for _ in range(self.max_samples)]
            X_sample = [X[idx] for idx in indices]
            y_sample = [y[idx] for idx in indices]
            
            # Vytvorenie a trénovanie novej inštancie základného klasifikátora
            estimator = self._clone_estimator()
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def _majority_vote(self, predictions):
        """
        Vráti triedu s najväčším počtom hlasov z daného zoznamu predikcií.
        V prípade remízy vráti prvú z najčastejších tried.

        Parameters
        ----------
        - predictions: zoznam predikcií (tried) od jednotlivých modelov.

        Returns
        -------
        - pred: trieda s najväčším počtom hlasov
        """
        vote_count = {}
        for pred in predictions:
            vote_count[pred] = vote_count.get(pred, 0) + 1
        max_votes = max(vote_count.values())
        for pred in predictions:
            if vote_count[pred] == max_votes:
                return pred

    def predict(self, X):
        """
        Vykoná predikciu na dátach X.
        
        Parameters
        ----------
        - X: dáta, pre ktoré chceme vykonať predikciu (zoznam textov).

        Returns
        -------
        - aggregated_predictions: zoznam predikovaných tried získaných hlasovaním zo všetkých modelov.
        """
        all_predictions = [estimator.predict(X) for estimator in self.estimators]
        all_predictions = list(zip(*all_predictions))
        aggregated_predictions = [self._majority_vote(preds) for preds in all_predictions]
        return aggregated_predictions