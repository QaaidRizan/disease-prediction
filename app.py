"""
Complete Flask API for Disease Prediction System
Handles all prediction requests with severity weighting, descriptions, and precautions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for models
model = None
symptom_list = None
severity_dict = None
description_dict = None
precaution_dict = None


def load_models():
    """Load all trained models and dictionaries"""
    global model, symptom_list, severity_dict, description_dict, precaution_dict

    print("=" * 70)
    print("🔄 Loading models...")
    print("=" * 70)

    try:
        model_path = 'models/'

        # Load model
        model = joblib.load(f'{model_path}disease_model.pkl')
        print("✅ Loaded: disease_model.pkl")

        # Load symptom list
        symptom_list = joblib.load(f'{model_path}symptom_list.pkl')
        print(f"✅ Loaded: symptom_list.pkl ({len(symptom_list)} symptoms)")

        # Load severity dictionary
        severity_dict = joblib.load(f'{model_path}severity_dict.pkl')
        print(f"✅ Loaded: severity_dict.pkl ({len(severity_dict)} entries)")

        # Load description dictionary
        description_dict = joblib.load(f'{model_path}description_dict.pkl')
        print(f"✅ Loaded: description_dict.pkl ({len(description_dict)} entries)")

        # Load precaution dictionary
        precaution_dict = joblib.load(f'{model_path}precaution_dict.pkl')
        print(f"✅ Loaded: precaution_dict.pkl ({len(precaution_dict)} entries)")

        print("=" * 70)
        print("✅ All models loaded successfully!")
        print("=" * 70)

        return True

    except Exception as e:
        print("=" * 70)
        print("❌ ERROR loading models!")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("\nPlease make sure you have trained the model first:")
        print("  python models/train_model.py")
        print("=" * 70)
        return False


# Load models on startup
models_loaded = load_models()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Disease Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'models_loaded': models_loaded,
        'endpoints': {
            'health': 'GET /api/health',
            'symptoms': 'GET /api/symptoms',
            'diseases': 'GET /api/diseases',
            'predict': 'POST /api/predict',
            'analyze': 'POST /api/analyze-symptoms',
            'disease_info': 'GET /api/disease/<name>'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'error',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': models_loaded,
        'total_symptoms': len(symptom_list) if symptom_list else 0,
        'total_diseases': len(model.classes_) if model else 0
    })


@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """
    Get all available symptoms with severity weights and display names

    Returns:
        JSON with list of symptoms
    """
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        symptoms_data = []

        for symptom in sorted(symptom_list):
            symptoms_data.append({
                'name': symptom,
                'display_name': symptom.replace('_', ' ').title(),
                'severity': severity_dict.get(symptom, 1),
                'severity_category': categorize_severity(severity_dict.get(symptom, 1))
            })

        return jsonify({
            'success': True,
            'symptoms': symptoms_data,
            'total': len(symptoms_data)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """
    Get all diseases with descriptions

    Returns:
        JSON with list of diseases and their descriptions
    """
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        diseases_data = []

        for disease in sorted(model.classes_):
            diseases_data.append({
                'name': disease,
                'description': description_dict.get(disease, 'No description available'),
                'has_precautions': disease in precaution_dict
            })

        return jsonify({
            'success': True,
            'diseases': diseases_data,
            'total': len(diseases_data)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """
    Predict diseases based on selected symptoms

    Request Body:
        {
            "symptoms": ["symptom1", "symptom2", ...],
            "use_severity": true/false (optional, default: true),
            "top_k": 5 (optional, default: 5)
        }

    Returns:
        JSON with predictions, confidence scores, descriptions, and precautions
    """
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please train the model first.'}), 500

    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        selected_symptoms = data.get('symptoms', [])
        use_severity = data.get('use_severity', True)
        top_k = data.get('top_k', 5)

        # Validate input
        if not selected_symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        if not isinstance(selected_symptoms, list):
            return jsonify({'error': 'Symptoms must be a list'}), 400

        # Clean symptom names
        selected_symptoms = [str(s).strip().lower().replace(' ', '_') for s in selected_symptoms]

        # Create input vector with severity weighting
        input_vector = np.zeros(len(symptom_list))
        total_severity = 0
        valid_symptoms = []
        invalid_symptoms = []

        for symptom in selected_symptoms:
            if symptom in symptom_list:
                idx = symptom_list.index(symptom)
                severity = severity_dict.get(symptom, 1)

                if use_severity:
                    input_vector[idx] = severity
                    total_severity += severity
                else:
                    input_vector[idx] = 1
                    total_severity += 1

                valid_symptoms.append({
                    'name': symptom,
                    'display_name': symptom.replace('_', ' ').title(),
                    'severity': severity,
                    'severity_category': categorize_severity(severity)
                })
            else:
                invalid_symptoms.append(symptom)

        # Check if we have valid symptoms
        if not valid_symptoms:
            return jsonify({
                'error': 'None of the provided symptoms are valid',
                'invalid_symptoms': invalid_symptoms
            }), 400

        # Normalize by total severity
        if total_severity > 0 and use_severity:
            input_vector = input_vector / total_severity

        # Get prediction probabilities
        probabilities = model.predict_proba([input_vector])[0]
        disease_names = model.classes_

        # Create predictions with full details
        predictions = []
        for i in range(len(disease_names)):
            disease = disease_names[i]
            confidence = float(probabilities[i] * 100)

            # Only include diseases with >0.5% confidence
            if confidence > 0.5:
                predictions.append({
                    'disease': disease,
                    'confidence': round(confidence, 2),
                    'confidence_level': get_confidence_level(confidence),
                    'description': description_dict.get(disease, 'No description available'),
                    'precautions': precaution_dict.get(disease, []),
                    'severity_assessment': assess_disease_severity(confidence)
                })

        # Sort by confidence and get top K
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:top_k]

        # Calculate overall risk
        overall_risk = calculate_overall_risk(valid_symptoms, predictions)

        return jsonify({
            'success': True,
            'predictions': predictions,
            'symptom_count': len(valid_symptoms),
            'symptoms_analyzed': valid_symptoms,
            'invalid_symptoms': invalid_symptoms if invalid_symptoms else None,
            'severity_weighted': use_severity,
            'overall_risk': overall_risk,
            'recommendation': get_recommendation(overall_risk, predictions[0]['confidence'] if predictions else 0)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    """
    Analyze symptoms and provide severity scoring

    Request Body:
        {
            "symptoms": ["symptom1", "symptom2", ...]
        }

    Returns:
        JSON with symptom analysis and urgency level
    """
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        data = request.get_json()
        selected_symptoms = data.get('symptoms', [])

        if not selected_symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        # Clean symptom names
        selected_symptoms = [str(s).strip().lower().replace(' ', '_') for s in selected_symptoms]

        # Calculate severity metrics
        total_severity = 0
        max_severity = 0
        symptom_analysis = []
        valid_count = 0

        for symptom in selected_symptoms:
            if symptom in symptom_list:
                severity = severity_dict.get(symptom, 1)
                total_severity += severity
                max_severity = max(max_severity, severity)
                valid_count += 1

                symptom_analysis.append({
                    'symptom': symptom.replace('_', ' ').title(),
                    'severity_weight': severity,
                    'severity_category': categorize_severity(severity)
                })

        if valid_count == 0:
            return jsonify({'error': 'No valid symptoms found'}), 400

        avg_severity = total_severity / valid_count

        return jsonify({
            'success': True,
            'symptom_count': valid_count,
            'total_severity_score': round(total_severity, 2),
            'average_severity': round(avg_severity, 2),
            'max_severity': max_severity,
            'urgency_level': get_urgency_level(avg_severity),
            'urgency_color': get_urgency_color(avg_severity),
            'symptoms': symptom_analysis,
            'recommendation': get_urgency_recommendation(avg_severity)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/disease/<disease_name>', methods=['GET'])
def get_disease_details(disease_name):
    """
    Get detailed information about a specific disease

    Args:
        disease_name: Name of the disease

    Returns:
        JSON with disease details
    """
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        # Check if disease exists
        if disease_name not in model.classes_:
            return jsonify({
                'error': f'Disease "{disease_name}" not found',
                'available_diseases': list(model.classes_)[:10]
            }), 404

        return jsonify({
            'success': True,
            'disease': disease_name,
            'description': description_dict.get(disease_name, 'No description available'),
            'precautions': precaution_dict.get(disease_name, []),
            'precaution_count': len(precaution_dict.get(disease_name, []))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-symptoms', methods=['GET'])
def search_symptoms():
    """
    Search symptoms by keyword

    Query Params:
        q: Search query

    Returns:
        Matching symptoms
    """
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        query = request.args.get('q', '').lower().strip()

        if not query or len(query) < 2:
            return jsonify({'error': 'Query must be at least 2 characters'}), 400

        # Search for matching symptoms
        matching_symptoms = []
        for symptom in symptom_list:
            if query in symptom.lower():
                matching_symptoms.append({
                    'name': symptom,
                    'display_name': symptom.replace('_', ' ').title(),
                    'severity': severity_dict.get(symptom, 1)
                })

        return jsonify({
            'success': True,
            'query': query,
            'results': matching_symptoms[:20],  # Limit to 20 results
            'count': len(matching_symptoms)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def categorize_severity(weight):
    """Categorize symptom severity based on weight"""
    if weight >= 6:
        return 'Severe'
    elif weight >= 4:
        return 'Moderate'
    elif weight >= 2:
        return 'Mild'
    else:
        return 'Minor'


def get_confidence_level(confidence):
    """Categorize prediction confidence"""
    if confidence >= 70:
        return 'Very High'
    elif confidence >= 50:
        return 'High'
    elif confidence >= 30:
        return 'Moderate'
    elif confidence >= 10:
        return 'Low'
    else:
        return 'Very Low'


def assess_disease_severity(confidence):
    """Assess disease severity based on confidence"""
    if confidence >= 70:
        return 'High likelihood - Seek medical attention'
    elif confidence >= 40:
        return 'Moderate likelihood - Consult doctor soon'
    elif confidence >= 20:
        return 'Low likelihood - Monitor symptoms'
    else:
        return 'Very low likelihood'


def get_urgency_level(avg_severity):
    """Determine urgency based on average severity"""
    if avg_severity >= 6:
        return 'Urgent'
    elif avg_severity >= 4:
        return 'Important'
    elif avg_severity >= 2:
        return 'Moderate'
    else:
        return 'Low'


def get_urgency_color(avg_severity):
    """Get color code for urgency level"""
    if avg_severity >= 6:
        return 'red'
    elif avg_severity >= 4:
        return 'orange'
    elif avg_severity >= 2:
        return 'yellow'
    else:
        return 'green'


def get_urgency_recommendation(avg_severity):
    """Get recommendation based on urgency"""
    if avg_severity >= 6:
        return 'Seek immediate medical attention'
    elif avg_severity >= 4:
        return 'Consult a doctor within 24-48 hours'
    elif avg_severity >= 2:
        return 'Monitor symptoms and consult if they worsen'
    else:
        return 'General care and rest recommended'


def calculate_overall_risk(symptoms, predictions):
    """Calculate overall risk score"""
    if not symptoms or not predictions:
        return 'Low'

    avg_severity = sum(s['severity'] for s in symptoms) / len(symptoms)
    top_confidence = predictions[0]['confidence'] if predictions else 0

    risk_score = (avg_severity * 0.4) + (top_confidence * 0.6)

    if risk_score >= 60:
        return 'High'
    elif risk_score >= 40:
        return 'Moderate'
    else:
        return 'Low'


def get_recommendation(risk_level, confidence):
    """Get medical recommendation"""
    if risk_level == 'High' or confidence >= 70:
        return 'Consult a healthcare professional immediately'
    elif risk_level == 'Moderate' or confidence >= 40:
        return 'Schedule an appointment with your doctor soon'
    else:
        return 'Monitor your symptoms and seek help if they worsen'


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("=" * 70)
    print("🏥 DISEASE PREDICTION API SERVER")
    print("=" * 70)
    print("\n📡 Available Endpoints:")
    print("   GET  /                        - API information")
    print("   GET  /api/health              - Health check")
    print("   GET  /api/symptoms            - Get all symptoms")
    print("   GET  /api/diseases            - Get all diseases")
    print("   POST /api/predict             - Predict disease")
    print("   POST /api/analyze-symptoms    - Analyze symptom severity")
    print("   GET  /api/disease/<name>      - Get disease details")
    print("   GET  /api/search-symptoms?q=  - Search symptoms")
    print("\n🚀 Server Configuration:")
    print(f"   Host: 0.0.0.0")
    print(f"   Port: 5000")
    print(f"   Models Loaded: {'✅ Yes' if models_loaded else '❌ No'}")
    print("\n💡 Server starting at: http://localhost:5000")
    print("=" * 70)
    print("\n")

    if not models_loaded:
        print("⚠️  WARNING: Models not loaded!")
        print("   Please train the model first: python models/train_model.py\n")

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )