#!/usr/bin/env python3
"""
IBM AI Engineering Professional Certificate Capstone Project
Advanced Deep Learning & Computer Vision Platform

This comprehensive AI engineering platform demonstrates competencies from the
IBM AI Engineering Professional Certificate program including:
- Deep learning model development
- Computer vision applications
- Natural language processing
- Model deployment and serving
- AI pipeline automation
- Performance optimization
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
from typing import Dict, List, Any, Optional, Tuple

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
import cv2

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Web Framework
from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class ComputerVisionEngine:
    """Computer vision model development and deployment"""
    
    def __init__(self):
        self.models = {}
        self.model_dir = 'cv_models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def generate_synthetic_image_data(self, n_samples=5000, img_size=(224, 224)):
        """Generate synthetic image classification dataset"""
        logger.info(f"Generating {n_samples} synthetic images...")
        
        # Create synthetic images for different categories
        categories = ['cats', 'dogs', 'birds', 'cars', 'flowers']
        images = []
        labels = []
        
        for i, category in enumerate(categories):
            samples_per_category = n_samples // len(categories)
            
            for j in range(samples_per_category):
                # Generate synthetic image with category-specific patterns
                img = np.random.rand(*img_size, 3)
                
                # Add category-specific features
                if category == 'cats':
                    # Add circular patterns (eyes)
                    center1 = (img_size[0]//3, img_size[1]//3)
                    center2 = (img_size[0]//3, 2*img_size[1]//3)
                    cv2.circle(img, center1, 20, (0.8, 0.8, 0.2), -1)
                    cv2.circle(img, center2, 20, (0.8, 0.8, 0.2), -1)
                    
                elif category == 'dogs':
                    # Add rectangular patterns
                    cv2.rectangle(img, (50, 50), (150, 150), (0.2, 0.8, 0.8), -1)
                    
                elif category == 'birds':
                    # Add triangular patterns (beaks)
                    pts = np.array([[100, 50], [150, 100], [50, 100]], np.int32)
                    cv2.fillPoly(img, [pts], (0.8, 0.2, 0.8))
                    
                elif category == 'cars':
                    # Add horizontal lines (wheels)
                    cv2.line(img, (0, img_size[0]//2), (img_size[1], img_size[0]//2), (0.8, 0.8, 0.8), 10)
                    
                elif category == 'flowers':
                    # Add radial patterns
                    center = (img_size[0]//2, img_size[1]//2)
                    for angle in range(0, 360, 45):
                        x = int(center[0] + 50 * np.cos(np.radians(angle)))
                        y = int(center[1] + 50 * np.sin(np.radians(angle)))
                        cv2.circle(img, (x, y), 10, (0.9, 0.1, 0.5), -1)
                
                # Add noise
                noise = np.random.normal(0, 0.1, img.shape)
                img = np.clip(img + noise, 0, 1)
                
                images.append(img)
                labels.append(i)
        
        return np.array(images), np.array(labels), categories
    
    def build_cnn_model(self, input_shape, num_classes):
        """Build a CNN model for image classification"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_transfer_learning_model(self, input_shape, num_classes, base_model_name='VGG16'):
        """Build a transfer learning model"""
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_image_classifier(self, model_name='custom_cnn'):
        """Train an image classification model"""
        logger.info(f"Training image classifier: {model_name}")
        
        # Generate synthetic data
        X, y, categories = self.generate_synthetic_image_data(3000, (128, 128))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        if model_name == 'transfer_learning':
            model = self.build_transfer_learning_model((128, 128, 3), len(categories), 'MobileNetV2')
        else:
            model = self.build_cnn_model((128, 128, 3), len(categories))
        
        # Training callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=20,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Save model
        model_path = os.path.join(self.model_dir, f'{model_name}.h5')
        model.save(model_path)
        
        # Store model info
        self.models[model_name] = {
            'model': model,
            'categories': categories,
            'accuracy': test_accuracy,
            'path': model_path
        }
        
        logger.info(f"Model {model_name} trained successfully. Accuracy: {test_accuracy:.4f}")
        
        return model, history, test_accuracy

class NLPEngine:
    """Natural Language Processing model development"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.model_dir = 'nlp_models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def generate_synthetic_text_data(self, n_samples=10000):
        """Generate synthetic text classification dataset"""
        logger.info(f"Generating {n_samples} synthetic text samples...")
        
        # Define categories and sample texts
        categories = {
            'technology': [
                'artificial intelligence machine learning deep learning neural networks',
                'software development programming python javascript react',
                'cloud computing aws azure google cloud kubernetes docker',
                'data science analytics big data visualization dashboard',
                'cybersecurity encryption blockchain cryptocurrency bitcoin'
            ],
            'business': [
                'marketing strategy customer acquisition sales revenue growth',
                'financial analysis investment portfolio risk management',
                'project management agile scrum team collaboration',
                'human resources recruitment talent management culture',
                'operations supply chain logistics efficiency optimization'
            ],
            'health': [
                'medical research clinical trials pharmaceutical drug development',
                'healthcare system patient care treatment diagnosis',
                'nutrition diet exercise fitness wellness lifestyle',
                'mental health therapy counseling stress management',
                'public health epidemiology vaccination disease prevention'
            ],
            'education': [
                'online learning e-learning educational technology platform',
                'curriculum development teaching methods pedagogy assessment',
                'student engagement motivation academic performance',
                'higher education university college degree program',
                'professional development training certification skills'
            ]
        }
        
        texts = []
        labels = []
        
        for label, category_texts in categories.items():
            samples_per_category = n_samples // len(categories)
            
            for i in range(samples_per_category):
                # Randomly combine and modify base texts
                base_text = np.random.choice(category_texts)
                words = base_text.split()
                
                # Add some randomness
                if len(words) > 3:
                    # Randomly shuffle some words
                    shuffle_indices = np.random.choice(len(words), size=min(3, len(words)//2), replace=False)
                    for idx in shuffle_indices:
                        if idx < len(words) - 1:
                            words[idx], words[idx + 1] = words[idx + 1], words[idx]
                
                # Add random words from the same category
                if np.random.random() > 0.5:
                    additional_words = np.random.choice(category_texts).split()[:3]
                    words.extend(additional_words)
                
                text = ' '.join(words)
                texts.append(text)
                labels.append(label)
        
        return texts, labels, list(categories.keys())
    
    def preprocess_text(self, texts):
        """Preprocess text data"""
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        processed_texts = []
        
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [lemmatizer.lemmatize(token) for token in tokens 
                     if token.isalpha() and token not in stop_words]
            
            processed_texts.append(' '.join(tokens))
        
        return processed_texts
    
    def train_text_classifier(self, model_name='text_classifier'):
        """Train a text classification model"""
        logger.info(f"Training text classifier: {model_name}")
        
        # Generate synthetic data
        texts, labels, categories = self.generate_synthetic_text_data(8000)
        
        # Preprocess texts
        processed_texts = self.preprocess_text(texts)
        
        # Convert labels to numeric
        label_to_idx = {label: idx for idx, label in enumerate(categories)}
        numeric_labels = [label_to_idx[label] for label in labels]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels
        )
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        accuracy = (y_pred == y_test).mean()
        
        # Store model and vectorizer
        self.models[model_name] = {
            'model': model,
            'vectorizer': vectorizer,
            'categories': categories,
            'label_to_idx': label_to_idx,
            'accuracy': accuracy
        }
        
        logger.info(f"Text classifier {model_name} trained successfully. Accuracy: {accuracy:.4f}")
        
        return model, vectorizer, accuracy

class AIModelRegistry:
    """AI model registry and management"""
    
    def __init__(self, db_path='ai_models.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize AI model registry database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                framework TEXT NOT NULL,
                accuracy REAL,
                parameters TEXT,
                file_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                input_type TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("AI model registry database initialized")
    
    def register_model(self, model_name: str, model_type: str, framework: str, 
                      accuracy: float, parameters: Dict, file_path: str):
        """Register a new AI model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_models 
            (model_name, model_type, framework, accuracy, parameters, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_name, model_type, framework, accuracy, json.dumps(parameters), file_path))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"AI model registered: {model_name} (ID: {model_id})")
        return model_id

class AIServingAPI:
    """AI model serving API"""
    
    def __init__(self, cv_engine: ComputerVisionEngine, nlp_engine: NLPEngine):
        self.cv_engine = cv_engine
        self.nlp_engine = nlp_engine
        self.registry = AIModelRegistry()
    
    def predict_image(self, model_name: str, image_data: str) -> Dict:
        """Make image prediction"""
        try:
            if model_name not in self.cv_engine.models:
                return {'error': f'Model {model_name} not found'}
            
            model_info = self.cv_engine.models[model_name]
            model = model_info['model']
            categories = model_info['categories']
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image = image.resize((128, 128))
            image_array = np.array(image) / 255.0
            
            # Make prediction
            prediction = model.predict(np.expand_dims(image_array, axis=0))
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            result = {
                'prediction': categories[predicted_class],
                'confidence': confidence,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log prediction
            self.log_prediction(model_name, 'image', result['prediction'], confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Image prediction error: {e}")
            return {'error': str(e)}
    
    def predict_text(self, model_name: str, text: str) -> Dict:
        """Make text prediction"""
        try:
            if model_name not in self.nlp_engine.models:
                return {'error': f'Model {model_name} not found'}
            
            model_info = self.nlp_engine.models[model_name]
            model = model_info['model']
            vectorizer = model_info['vectorizer']
            categories = model_info['categories']
            
            # Preprocess text
            processed_text = self.nlp_engine.preprocess_text([text])
            text_vec = vectorizer.transform(processed_text)
            
            # Make prediction
            prediction = model.predict(text_vec)[0]
            confidence = float(np.max(model.predict_proba(text_vec)[0]))
            
            result = {
                'prediction': categories[prediction],
                'confidence': confidence,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log prediction
            self.log_prediction(model_name, 'text', result['prediction'], confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Text prediction error: {e}")
            return {'error': str(e)}
    
    def log_prediction(self, model_name: str, input_type: str, prediction: str, confidence: float):
        """Log prediction for monitoring"""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (model_name, input_type, prediction, confidence)
            VALUES (?, ?, ?, ?)
        ''', (model_name, input_type, prediction, confidence))
        
        conn.commit()
        conn.close()

# Flask API
app = Flask(__name__)

# Initialize AI platform components
cv_engine = ComputerVisionEngine()
nlp_engine = NLPEngine()
serving_api = AIServingAPI(cv_engine, nlp_engine)

@app.route('/')
def dashboard():
    """AI Platform dashboard"""
    return jsonify({
        'service': 'IBM AI Engineering Platform',
        'version': '1.0.0',
        'status': 'running',
        'capabilities': ['computer_vision', 'natural_language_processing'],
        'endpoints': [
            '/api/train/cv',
            '/api/train/nlp',
            '/api/predict/image',
            '/api/predict/text',
            '/api/models',
            '/api/health'
        ]
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cv_models': list(cv_engine.models.keys()),
        'nlp_models': list(nlp_engine.models.keys())
    })

@app.route('/api/train/cv', methods=['POST'])
def train_cv_model():
    """Train a computer vision model"""
    try:
        data = request.json
        model_name = data.get('model_name', 'custom_cnn')
        
        model, history, accuracy = cv_engine.train_image_classifier(model_name)
        
        # Register model
        serving_api.registry.register_model(
            model_name=model_name,
            model_type='computer_vision',
            framework='tensorflow',
            accuracy=accuracy,
            parameters={'epochs': 20, 'batch_size': 32},
            file_path=cv_engine.models[model_name]['path']
        )
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'accuracy': accuracy,
            'model_type': 'computer_vision'
        })
        
    except Exception as e:
        logger.error(f"CV training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/nlp', methods=['POST'])
def train_nlp_model():
    """Train an NLP model"""
    try:
        data = request.json
        model_name = data.get('model_name', 'text_classifier')
        
        model, vectorizer, accuracy = nlp_engine.train_text_classifier(model_name)
        
        # Register model
        serving_api.registry.register_model(
            model_name=model_name,
            model_type='natural_language_processing',
            framework='scikit-learn',
            accuracy=accuracy,
            parameters={'max_features': 5000, 'ngram_range': '(1,2)'},
            file_path=''
        )
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'accuracy': accuracy,
            'model_type': 'natural_language_processing'
        })
        
    except Exception as e:
        logger.error(f"NLP training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """Make image prediction"""
    try:
        data = request.json
        model_name = data.get('model_name', 'custom_cnn')
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({'error': 'image_data required'}), 400
        
        result = serving_api.predict_image(model_name, image_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Image prediction API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/text', methods=['POST'])
def predict_text():
    """Make text prediction"""
    try:
        data = request.json
        model_name = data.get('model_name', 'text_classifier')
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'text required'}), 400
        
        result = serving_api.predict_text(model_name, text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Text prediction API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def list_models():
    """List all trained models"""
    conn = sqlite3.connect(serving_api.registry.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT model_name, model_type, framework, accuracy, created_at
        FROM ai_models ORDER BY created_at DESC
    ''')
    
    models = []
    for row in cursor.fetchall():
        models.append({
            'model_name': row[0],
            'model_type': row[1],
            'framework': row[2],
            'accuracy': row[3],
            'created_at': row[4]
        })
    
    conn.close()
    return jsonify(models)

def run_ai_training_pipeline():
    """Run automated AI training pipeline"""
    logger.info("🤖 Starting IBM AI Engineering Platform Training Pipeline")
    
    # Train computer vision models
    logger.info("Training computer vision models...")
    cv_model, cv_history, cv_accuracy = cv_engine.train_image_classifier('custom_cnn')
    transfer_model, transfer_history, transfer_accuracy = cv_engine.train_image_classifier('transfer_learning')
    
    # Train NLP model
    logger.info("Training NLP model...")
    nlp_model, vectorizer, nlp_accuracy = nlp_engine.train_text_classifier('text_classifier')
    
    logger.info("✅ AI training pipeline completed successfully")
    
    return {
        'cv_accuracy': cv_accuracy,
        'transfer_accuracy': transfer_accuracy,
        'nlp_accuracy': nlp_accuracy
    }

if __name__ == '__main__':
    print("🤖 IBM AI Engineering Professional Certificate Capstone Project")
    print("🧠 Advanced Deep Learning & Computer Vision Platform")
    print("=" * 65)
    
    # Run training pipeline
    results = run_ai_training_pipeline()
    
    print(f"\n🎯 Training Results:")
    print(f"Custom CNN Accuracy: {results['cv_accuracy']:.4f}")
    print(f"Transfer Learning Accuracy: {results['transfer_accuracy']:.4f}")
    print(f"NLP Classifier Accuracy: {results['nlp_accuracy']:.4f}")
    
    # Start API server
    logger.info("Starting AI Platform API on http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=False)

