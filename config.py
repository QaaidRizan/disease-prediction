import os
from pathlib import Path


class Config:
    """Base configuration"""
    # Application settings
    APP_NAME = "Disease Prediction API"
    VERSION = "1.0.0"
    DEBUG = True

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    MODEL_DIR = BASE_DIR / "models"

    # Model files
    MODEL_PATH = MODEL_DIR / "disease_model.pkl"
    SYMPTOM_LIST_PATH = MODEL_DIR / "symptom_list.pkl"
    SEVERITY_DICT_PATH = MODEL_DIR / "severity_dict.pkl"
    DESCRIPTION_DICT_PATH = MODEL_DIR / "description_dict.pkl"
    PRECAUTION_DICT_PATH = MODEL_DIR / "precaution_dict.pkl"

    # Dataset files
    DATASET_PATH = DATA_DIR / "dataset.csv"
    SEVERITY_PATH = DATA_DIR / "Symptom-severity.csv"
    DESCRIPTION_PATH = DATA_DIR / "symptom_Description.csv"
    PRECAUTION_PATH = DATA_DIR / "symptom_precaution.csv"

    # Server settings
    HOST = "0.0.0.0"
    PORT = 5000

    # CORS settings
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]

    # Model parameters
    N_ESTIMATORS = 200
    MAX_DEPTH = 20
    MIN_SAMPLES_SPLIT = 5
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # API settings
    MAX_PREDICTIONS = 5
    MIN_CONFIDENCE = 1.0  # Minimum confidence percentage to show

    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Add production-specific settings here


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}