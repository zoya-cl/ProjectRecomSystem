import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

class TextPreprocessor:
    """
    Dedicated class for text preprocessing steps.
    Includes:
    1. Lowercasing
    2. Punctuation removal (preserving tech-specific chars)
    3. Tokenization
    4. Stop-word removal
    5. Stemming
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
            
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove special characters (preserving .+# for tech names)
        text = re.sub(r'[^a-z0-9+#.\s]', ' ', text)
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4. Stop-word removal and Stemming
        processed_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 1
        ]
        
        # 5. Rejoin into string
        return " ".join(processed_tokens)

    def preprocess_list(self, texts: list) -> list:
        return [self.preprocess(t) for t in texts]
