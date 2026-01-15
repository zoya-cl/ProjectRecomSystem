#!/usr/bin/env python3
"""
Advanced Text Preprocessor with Domain-Specific Handling

This preprocessor is specifically designed for technical/project recommendation systems.
It preserves important domain-specific terminology while still performing effective 
noise reduction and normalization.

Key improvements over basic preprocessing:
1. Lemmatization instead of stemming (preserves word meaning better)
2. Domain-specific term protection (agriculture, blockchain, etc. won't be modified)
3. Bigram/trigram phrase detection for technical terms
4. Special handling for programming languages and frameworks
5. Number preservation with context
6. Acronym and abbreviation handling
"""

import re
import nltk
from typing import List, Set, Optional
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')


class AdvancedTextPreprocessor:
    """
    Advanced preprocessor with domain-specific handling for technical content.
    """
    
    def __init__(self, preserve_numbers: bool = True):
        """
        Initialize the advanced preprocessor.
        
        Args:
            preserve_numbers: If True, keep numbers with context (e.g., "python3", "5g")
        """
        self.lemmatizer = WordNetLemmatizer()
        self.preserve_numbers = preserve_numbers
        
        # Standard English stopwords, but we'll be selective
        self.base_stopwords = set(stopwords.words('english'))
        
        # Remove some stopwords that might be important in technical contexts
        self.technical_stopwords = self.base_stopwords - {
            'system', 'data', 'network', 'learning', 'processing',
            'management', 'application', 'web', 'mobile', 'cloud'
        }
        
        # Domain-specific terms that should NEVER be stemmed/lemmatized or removed
        self.protected_terms = self._build_protected_terms()
        
        # Programming languages and frameworks
        self.tech_terms = self._build_tech_terms()
        
        # Common bigrams/trigrams in tech
        self.known_phrases = self._build_known_phrases()
        
    def _build_protected_terms(self) -> Set[str]:
        """Build set of domain-specific terms to protect from modification."""
        return {
            # Domains
            'agriculture', 'agricultural', 'farming', 'crops', 'livestock',
            'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'smart contracts',
            'healthcare', 'medical', 'clinical', 'diagnosis', 'patient',
            'finance', 'financial', 'banking', 'trading', 'investment',
            'education', 'educational', 'learning', 'teaching', 'training',
            'ecommerce', 'retail', 'shopping', 'marketplace',
            'iot', 'internet of things', 'sensors', 'embedded',
            'robotics', 'automation', 'autonomous',
            'gaming', 'game', 'entertainment',
            'logistics', 'supply chain', 'inventory', 'warehouse',
            'energy', 'renewable', 'solar', 'wind', 'power',
            'manufacturing', 'production', 'assembly',
            'transportation', 'vehicle', 'automotive',
            'telecommunications', 'networking', 'wireless',
            
            # Technologies
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'cnn', 'rnn', 'lstm', 'transformer',
            'computer vision', 'image processing', 'object detection',
            'nlp', 'natural language processing', 'text mining',
            'data science', 'analytics', 'visualization',
            'cloud computing', 'serverless', 'microservices',
            'devops', 'cicd', 'continuous integration',
            'cybersecurity', 'encryption', 'authentication',
            'database', 'sql', 'nosql', 'mongodb', 'postgresql',
            'api', 'rest', 'graphql', 'websocket',
            'frontend', 'backend', 'fullstack',
            'mobile', 'android', 'ios', 'flutter',
            'web development', 'responsive', 'pwa',
            
            # Methodologies
            'agile', 'scrum', 'kanban', 'waterfall',
            'supervised', 'unsupervised', 'reinforcement',
            'classification', 'regression', 'clustering',
            'recommendation', 'prediction', 'forecasting',
        }
    
    def _build_tech_terms(self) -> Set[str]:
        """Build set of programming languages and frameworks."""
        return {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'julia',
            
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'svelte', 'nextjs', 'nuxt',
            'django', 'flask', 'fastapi', 'spring', 'express',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
            'pandas', 'numpy', 'opencv', 'matplotlib', 'seaborn',
            'node', 'nodejs', 'deno', 'bun',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
            'elasticsearch', 'neo4j', 'firebase', 'supabase',
        }
    
    def _build_known_phrases(self) -> Set[str]:
        """Build set of known technical phrases."""
        return {
            'machine learning', 'deep learning', 'computer vision',
            'natural language processing', 'data science', 'web development',
            'mobile development', 'cloud computing', 'data structure',
            'object oriented', 'functional programming', 'test driven',
            'user interface', 'user experience', 'real time',
            'big data', 'data mining', 'artificial intelligence',
            'neural network', 'supply chain', 'smart contract',
        }
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert treebank POS tag to WordNet POS tag for better lemmatization."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun
    
    def _detect_phrases(self, tokens: List[str]) -> List[str]:
        """
        Detect and combine known technical phrases (bigrams/trigrams).
        
        For example: ['machine', 'learning'] -> ['machine_learning']
        """
        result = []
        i = 0
        
        while i < len(tokens):
            # Try trigram first
            if i + 2 < len(tokens):
                trigram = ' '.join(tokens[i:i+3])
                if trigram in self.known_phrases:
                    result.append(trigram.replace(' ', '_'))
                    i += 3
                    continue
            
            # Try bigram
            if i + 1 < len(tokens):
                bigram = ' '.join(tokens[i:i+2])
                if bigram in self.known_phrases:
                    result.append(bigram.replace(' ', '_'))
                    i += 2
                    continue
            
            # Single token
            result.append(tokens[i])
            i += 1
        
        return result
    
    def _normalize_special_terms(self, text: str) -> str:
        """
        Normalize special technical terms before tokenization.
        
        Examples:
        - "C++" -> "cpp"
        - "C#" -> "csharp"
        - "Node.js" -> "nodejs"
        """
        replacements = {
            r'\bc\+\+\b': 'cpp',
            r'\bc#\b': 'csharp',
            r'\bf#\b': 'fsharp',
            r'\bnode\.js\b': 'nodejs',
            r'\bnext\.js\b': 'nextjs',
            r'\bvue\.js\b': 'vuejs',
            r'\breact\.js\b': 'reactjs',
            r'\b\.net\b': 'dotnet',
            r'\basp\.net\b': 'aspnet',
        }
        
        text_lower = text.lower()
        for pattern, replacement in replacements.items():
            text_lower = re.sub(pattern, replacement, text_lower, flags=re.IGNORECASE)
        
        return text_lower
    
    def preprocess(self, text: str, verbose: bool = False) -> str:
        """
        Preprocess text with advanced domain-aware techniques.
        
        Args:
            text: Input text to preprocess
            verbose: If True, print intermediate steps for debugging
            
        Returns:
            Preprocessed text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        if verbose:
            print(f"Original: {text[:100]}...")
        
        # Step 1: Normalize special technical terms
        text = self._normalize_special_terms(text)
        
        if verbose:
            print(f"After normalization: {text[:100]}...")
        
        # Step 2: Remove URLs, emails, and other noise
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Step 3: Preserve important patterns (version numbers, etc.)
        # Replace with placeholders temporarily
        version_pattern = r'\b\w+\d+(?:\.\d+)*\b'  # e.g., "python3.9", "v2.0"
        versions = re.findall(version_pattern, text)
        for i, version in enumerate(versions):
            text = text.replace(version, f'VERSION{i}PLACEHOLDER')
        
        # Step 4: Clean special characters, but preserve some (#, +, .)
        text = re.sub(r'[^a-z0-9+#.\s_-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Step 5: Tokenization
        tokens = word_tokenize(text)
        
        # Restore version placeholders
        for i, version in enumerate(versions):
            tokens = [version.lower() if t == f'version{i}placeholder' else t for t in tokens]
        
        if verbose:
            print(f"Tokens: {tokens[:20]}...")
        
        # Step 6: Detect and combine known phrases
        tokens = self._detect_phrases(tokens)
        
        # Step 7: POS tagging for better lemmatization
        pos_tags = pos_tag(tokens)
        
        # Step 8: Process each token
        processed_tokens = []
        for word, pos in pos_tags:
            # Skip very short tokens (except important ones like 'ai', 'ml', 'r')
            if len(word) <= 1 and word.lower() not in {'r', 'c'}:
                continue
            
            word_lower = word.lower()
            
            # Keep protected terms as-is
            if word_lower in self.protected_terms or word_lower in self.tech_terms:
                processed_tokens.append(word_lower)
                continue
            
            # Skip stopwords (but not if they're part of a technical term)
            if word_lower in self.technical_stopwords:
                continue
            
            # Lemmatize with POS awareness
            wordnet_pos = self._get_wordnet_pos(pos)
            lemmatized = self.lemmatizer.lemmatize(word_lower, pos=wordnet_pos)
            
            processed_tokens.append(lemmatized)
        
        result = ' '.join(processed_tokens)
        
        if verbose:
            print(f"Final: {result[:100]}...")
        
        return result
    
    def preprocess_list(self, texts: List[str], verbose: bool = False) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of input texts
            verbose: If True, print progress
            
        Returns:
            List of preprocessed texts
        """
        results = []
        for i, text in enumerate(texts):
            if verbose and i % 100 == 0:
                print(f"Processing {i}/{len(texts)}...")
            results.append(self.preprocess(text, verbose=False))
        return results
    
    def analyze_vocabulary(self, texts: List[str], top_n: int = 50) -> dict:
        """
        Analyze vocabulary after preprocessing to verify important terms are preserved.
        
        Args:
            texts: List of texts to analyze
            top_n: Number of top terms to return
            
        Returns:
            Dictionary with vocabulary statistics
        """
        all_tokens = []
        for text in texts:
            processed = self.preprocess(text)
            all_tokens.extend(processed.split())
        
        counter = Counter(all_tokens)
        
        return {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(counter),
            'top_terms': counter.most_common(top_n),
            'protected_terms_found': [term for term in self.protected_terms if counter[term] > 0],
            'tech_terms_found': [term for term in self.tech_terms if counter[term] > 0]
        }


# Test function
if __name__ == "__main__":
    preprocessor = AdvancedTextPreprocessor()
    
    # Test cases
    test_texts = [
        "Agriculture-based IoT system for smart farming",
        "Blockchain implementation for healthcare records",
        "Machine Learning model using Python and TensorFlow",
        "Real-time object detection with OpenCV and C++",
        "Web development using React.js and Node.js",
        "Natural Language Processing for sentiment analysis",
    ]
    
    print("=" * 80)
    print("ADVANCED PREPROCESSOR TEST")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        processed = preprocessor.preprocess(text, verbose=True)
        print(f"Processed: {processed}")
        print("-" * 80)
