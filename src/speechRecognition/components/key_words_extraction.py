from keybert import KeyBERT


class KeyWordsExtractor:
    def __init__(self, text):
        self.text = text

    def recognize_key_words(self):

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(self.text)

        return keywords