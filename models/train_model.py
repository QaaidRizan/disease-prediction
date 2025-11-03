"""
Disease Prediction ML Model Training Pipeline
Customized for your specific dataset format
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DiseasePredictionModel:
    def __init__(self, data_path='data/raw/'):
        # Get the absolute path relative to this script's location
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level from models/
        self.data_path = os.path.join(project_root, data_path)
        self.model = None
        self.symptom_columns = []
        self.all_symptoms = set()
        self.severity_dict = {}
        self.description_dict = {}
        self.precaution_dict = {}

    def load_datasets(self):
        """Load all CSV files"""
        print("=" * 70)
        print("📂 Loading datasets...")
        print("=" * 70)

        # 1. Main dataset (your format: Disease, symptom1, symptom2, ...)
        try:
            self.df_main = pd.read_csv(f'{self.data_path}dataset.csv')
            print(f"✅ Main dataset loaded: {self.df_main.shape}")
            print(f"   Columns: {len(self.df_main.columns)}")
            print(f"   First few columns: {list(self.df_main.columns[:5])}")
        except Exception as e:
            print(f"❌ Error loading dataset.csv: {e}")
            raise

        # 2. Symptom severity
        try:
            self.df_severity = pd.read_csv(f'{self.data_path}Symptom-severity.csv')
            print(f"✅ Severity data loaded: {self.df_severity.shape}")
        except Exception as e:
            print(f"❌ Error loading Symptom-severity.csv: {e}")
            raise

        # 3. Disease descriptions (from document provided)
        try:
            # You can save the descriptions as a CSV or create from the text
            self.df_description = pd.read_csv(f'{self.data_path}symptom_Description.csv')
            print(f"✅ Description data loaded: {self.df_description.shape}")
        except Exception as e:
            print(f"⚠️  Description file not found, creating from text...")
            # Create from the document you provided
            self.create_description_dict_from_text()

        # 4. Precautions
        try:
            self.df_precaution = pd.read_csv(f'{self.data_path}symptom_precaution.csv')
            print(f"✅ Precaution data loaded: {self.df_precaution.shape}")
        except Exception as e:
            print(f"❌ Error loading symptom_precaution.csv: {e}")
            raise

    def create_description_dict_from_text(self):
        """Create description dictionary from the provided text"""
        descriptions = {
            'Drug Reaction': "An adverse drug reaction (ADR) is an injury caused by taking medication. ADRs may occur following a single dose or prolonged administration of a drug or result from the combination of two or more drugs.",
            'Malaria': "An infectious disease caused by protozoan parasites from the Plasmodium family that can be transmitted by the bite of the Anopheles mosquito or by a contaminated needle or transfusion.",
            'Allergy': "An allergy is an immune system response to a foreign substance that's not typically harmful to your body. They can include certain foods, pollen, or pet dander.",
            'GERD': "Gastroesophageal reflux disease, or GERD, is a digestive disorder that affects the lower esophageal sphincter (LES), the ring of muscle between the esophagus and stomach.",
            'Chronic cholestasis': "Chronic cholestatic diseases are characterized by defective bile acid transport from the liver to the intestine, which is caused by primary damage to the biliary epithelium.",
            # Add more as needed
        }
        self.description_dict = descriptions

    def preprocess_data(self):
        """Transform your format into binary symptom matrix"""
        print("\n" + "=" * 70)
        print("🔧 Preprocessing data...")
        print("=" * 70)

        # Get unique disease names from first column
        disease_col = self.df_main.columns[0]
        print(f"Disease column: {disease_col}")

        # Collect all unique symptoms from the dataset
        print("\n📊 Collecting all unique symptoms...")
        for col in self.df_main.columns[1:]:  # Skip disease column
            symptoms = self.df_main[col].dropna().unique()
            for symptom in symptoms:
                if symptom and str(symptom).strip():
                    # Clean symptom name
                    clean_symptom = str(symptom).strip().lower().replace(' ', '_')
                    self.all_symptoms.add(clean_symptom)

        self.symptom_columns = sorted(list(self.all_symptoms))
        print(f"✅ Found {len(self.symptom_columns)} unique symptoms")
        print(f"   Sample symptoms: {self.symptom_columns[:5]}")

        # Create binary matrix
        print("\n🔄 Creating binary symptom matrix...")
        binary_data = []
        diseases = []

        for idx, row in self.df_main.iterrows():
            disease = row[disease_col]
            diseases.append(disease)

            # Create binary vector for this row
            symptom_vector = [0] * len(self.symptom_columns)

            # Mark present symptoms as 1
            for col in self.df_main.columns[1:]:
                symptom = row[col]
                if pd.notna(symptom) and str(symptom).strip():
                    clean_symptom = str(symptom).strip().lower().replace(' ', '_')
                    if clean_symptom in self.symptom_columns:
                        idx_symptom = self.symptom_columns.index(clean_symptom)
                        symptom_vector[idx_symptom] = 1

            binary_data.append(symptom_vector)

        # Create DataFrame
        self.df_processed = pd.DataFrame(binary_data, columns=self.symptom_columns)
        self.df_processed['prognosis'] = diseases

        print(f"✅ Binary matrix created: {self.df_processed.shape}")
        print(f"   Diseases: {self.df_processed['prognosis'].nunique()}")

        # Create severity dictionary
        print("\n⚖️  Loading symptom severity weights...")
        if hasattr(self, 'df_severity'):
            severity_col = 'Symptom'
            weight_col = 'weight'

            for _, row in self.df_severity.iterrows():
                symptom = str(row[severity_col]).strip().lower().replace(' ', '_')
                weight = row[weight_col]
                self.severity_dict[symptom] = weight

            print(f"✅ Loaded {len(self.severity_dict)} severity weights")

        # Create description dictionary
        print("\n📝 Loading disease descriptions...")
        if hasattr(self, 'df_description') and not self.df_description.empty:
            desc_col = self.df_description.columns[0]
            desc_text_col = self.df_description.columns[1]

            for _, row in self.df_description.iterrows():
                disease = str(row[desc_col]).strip()
                description = str(row[desc_text_col]).strip()
                self.description_dict[disease] = description

            print(f"✅ Loaded {len(self.description_dict)} descriptions")

        # Create precaution dictionary
        print("\n💊 Loading precautions...")
        if hasattr(self, 'df_precaution'):
            prec_col = self.df_precaution.columns[0]
            precaution_cols = [col for col in self.df_precaution.columns if col != prec_col]

            for _, row in self.df_precaution.iterrows():
                disease = str(row[prec_col]).strip()
                precautions = []
                for col in precaution_cols:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        precautions.append(str(row[col]).strip())
                self.precaution_dict[disease] = precautions

            print(f"✅ Loaded {len(self.precaution_dict)} precaution sets")

    def train_model(self):
        """Train the Random Forest model"""
        print("\n" + "=" * 70)
        print("🤖 Training Machine Learning Model...")
        print("=" * 70)

        # Separate features and target
        X = self.df_processed[self.symptom_columns]
        y = self.df_processed['prognosis']

        print(f"\n📊 Dataset Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   Features (symptoms): {len(self.symptom_columns)}")
        print(f"   Classes (diseases): {y.nunique()}")
        print(f"\n   Disease distribution:")
        for disease, count in y.value_counts().head(10).items():
            print(f"      {disease}: {count} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n📈 Train set: {len(X_train)} samples")
        print(f"   Test set:  {len(X_test)} samples")

        # Train Random Forest
        print("\n🌲 Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        self.model.fit(X_train, y_train)
        print("✅ Model training completed!")

        # Evaluate
        print("\n📊 Model Evaluation:")
        print("-" * 70)

        # Training accuracy
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"   Training Accuracy:   {train_accuracy * 100:.2f}%")

        # Testing accuracy
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"   Testing Accuracy:    {test_accuracy * 100:.2f}%")

        # Cross-validation
        print("\n🔄 Cross-validation (5-fold)...")
        cv_scores = cross_val_score(self.model, X, y, cv=5, n_jobs=-1)
        print(f"   CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Feature importance
        print("\n🔝 Top 15 Important Symptoms:")
        print("-" * 70)
        feature_importance = pd.DataFrame({
            'symptom': self.symptom_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        for i, row in feature_importance.head(15).iterrows():
            severity = self.severity_dict.get(row['symptom'], 'N/A')
            print(f"   {row['symptom']:30s} | Importance: {row['importance']:.4f} | Severity: {severity}")

        return test_accuracy

    def save_models(self, output_path='models/'):
        """Save all models and dictionaries"""
        print("\n" + "=" * 70)
        print("💾 Saving models and data...")
        print("=" * 70)

        # Create absolute path
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, output_path)

        # Create directory
        os.makedirs(output_path, exist_ok=True)

        # Save files
        joblib.dump(self.model, f'{output_path}disease_model.pkl')
        print(f"✅ Saved: disease_model.pkl")

        joblib.dump(self.symptom_columns, f'{output_path}symptom_list.pkl')
        print(f"✅ Saved: symptom_list.pkl ({len(self.symptom_columns)} symptoms)")

        joblib.dump(self.severity_dict, f'{output_path}severity_dict.pkl')
        print(f"✅ Saved: severity_dict.pkl ({len(self.severity_dict)} entries)")

        joblib.dump(self.description_dict, f'{output_path}description_dict.pkl')
        print(f"✅ Saved: description_dict.pkl ({len(self.description_dict)} entries)")

        joblib.dump(self.precaution_dict, f'{output_path}precaution_dict.pkl')
        print(f"✅ Saved: precaution_dict.pkl ({len(self.precaution_dict)} entries)")

        print("\n📦 All files saved successfully!")

    def test_prediction(self, sample_symptoms):
        """Test the model with sample symptoms"""
        print("\n" + "=" * 70)
        print("🧪 Testing Prediction...")
        print("=" * 70)

        print(f"\nInput symptoms: {sample_symptoms}")

        # Create input vector
        input_vector = np.zeros(len(self.symptom_columns))

        for symptom in sample_symptoms:
            clean_symptom = symptom.strip().lower().replace(' ', '_')
            if clean_symptom in self.symptom_columns:
                idx = self.symptom_columns.index(clean_symptom)
                input_vector[idx] = 1
            else:
                print(f"⚠️  Warning: '{symptom}' not found in trained symptoms")

        # Predict
        probabilities = self.model.predict_proba([input_vector])[0]
        disease_names = self.model.classes_

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]

        print("\n🎯 Top 3 Predictions:")
        print("-" * 70)
        for i, idx in enumerate(top_indices, 1):
            disease = disease_names[idx]
            confidence = probabilities[idx] * 100
            description = self.description_dict.get(disease, 'No description available')
            precautions = self.precaution_dict.get(disease, [])

            print(f"\n{i}. {disease}")
            print(f"   Confidence: {confidence:.2f}%")
            print(f"   Description: {description[:100]}...")
            if precautions:
                print(f"   Precautions: {', '.join(precautions[:3])}")


# Main execution
if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("🏥 DISEASE PREDICTION MODEL TRAINING")
    print("=" * 70)
    print()

    try:
        # Initialize
        model_trainer = DiseasePredictionModel(data_path='data/raw/')

        # Load all datasets
        model_trainer.load_datasets()

        # Preprocess
        model_trainer.preprocess_data()

        # Train
        accuracy = model_trainer.train_model()

        # Save
        model_trainer.save_models()

        # Test with sample
        print("\n")
        test_symptoms = ['itching', 'skin_rash', 'stomach_pain']
        model_trainer.test_prediction(test_symptoms)

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n   Final Test Accuracy: {accuracy * 100:.2f}%")
        print(f"   Models saved in: models/")
        print(f"   Ready for deployment!")
        print("\n" + "=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR OCCURRED")
        print("=" * 70)
        print(f"\n{str(e)}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 70)