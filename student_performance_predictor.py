import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    """
    A comprehensive student performance prediction system that uses multiple
    machine learning algorithms to predict student grades based on various factors.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.preprocessor = None
        self.feature_names = None
        self.target_column = None
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different machine learning models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=1.0, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
    
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate realistic student performance data"""
        
        np.random.seed(self.random_state)
        
        # Demographics
        genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        ages = np.random.normal(16, 1.5, n_samples).astype(int)
        ages = np.clip(ages, 14, 19)
        
        # Family background
        parent_education = np.random.choice(
            ['High School', 'Some College', 'Bachelor', 'Master', 'PhD'],
            n_samples, p=[0.3, 0.25, 0.25, 0.15, 0.05]
        )
        
        family_income = np.random.choice(
            ['Low', 'Medium', 'High'],
            n_samples, p=[0.3, 0.5, 0.2]
        )
        
        # Study habits
        study_hours = np.random.exponential(3, n_samples)
        study_hours = np.clip(study_hours, 0, 10)
        
        absences = np.random.poisson(5, n_samples)
        absences = np.clip(absences, 0, 20)
        
        # Previous grades
        previous_grade = np.random.normal(75, 15, n_samples)
        previous_grade = np.clip(previous_grade, 0, 100)
        
        # Extracurricular activities
        extracurricular = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.4, 0.3, 0.1])
        
        # Social factors
        social_support = np.random.choice(
            ['Low', 'Medium', 'High'],
            n_samples, p=[0.2, 0.5, 0.3]
        )
        
        # Health factors
        health_status = np.random.choice(
            ['Poor', 'Good', 'Excellent'],
            n_samples, p=[0.1, 0.6, 0.3]
        )
        
        # Technology access
        internet_access = np.random.choice([0, 1], n_samples, p=[0.15, 0.85])
        
        # Calculate final grade based on realistic relationships
        final_grade = (
            previous_grade * 0.4 +  # Previous performance is strong predictor
            study_hours * 3 +       # Study hours positively impact grades
            -absences * 1.5 +       # Absences negatively impact grades
            extracurricular * 2 +   # Extracurricular activities help
            np.random.normal(0, 8, n_samples)  # Random noise
        )
        
        # Add bonuses/penalties based on categorical variables
        education_bonus = {
            'High School': 0, 'Some College': 2, 'Bachelor': 4,
            'Master': 6, 'PhD': 8
        }
        
        income_bonus = {'Low': -3, 'Medium': 0, 'High': 3}
        support_bonus = {'Low': -2, 'Medium': 0, 'High': 2}
        health_bonus = {'Poor': -5, 'Good': 0, 'Excellent': 3}
        
        for i in range(n_samples):
            final_grade[i] += education_bonus[parent_education[i]]
            final_grade[i] += income_bonus[family_income[i]]
            final_grade[i] += support_bonus[social_support[i]]
            final_grade[i] += health_bonus[health_status[i]]
            final_grade[i] += internet_access[i] * 2
        
        # Clip grades to realistic range
        final_grade = np.clip(final_grade, 0, 100)
        
        # Create DataFrame
        data = {
            'gender': genders,
            'age': ages,
            'parent_education': parent_education,
            'family_income': family_income,
            'study_hours_per_week': study_hours,
            'absences': absences,
            'previous_grade': previous_grade,
            'extracurricular_activities': extracurricular,
            'social_support': social_support,
            'health_status': health_status,
            'internet_access': internet_access,
            'final_grade': final_grade
        }
        
        df = pd.DataFrame(data)
        return df
    
    def load_data(self, filepath: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
        """Load data from file or use provided DataFrame"""
        if df is not None:
            return df
        elif filepath:
            return pd.read_csv(filepath)
        else:
            print("Generating sample data...")
            return self.generate_sample_data()
    
    def explore_data(self, df: pd.DataFrame, target_column: str = 'final_grade'):
        """Perform exploratory data analysis"""
        
        print("=== DATA EXPLORATION ===")
        print(f"Dataset shape: {df.shape}")
        print(f"\nTarget variable: {target_column}")
        print(f"Target statistics:")
        print(df[target_column].describe())
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found.")
        
        # Data types
        print(f"\nData types:")
        print(df.dtypes)
        
        # Create visualizations
        self._create_exploration_plots(df, target_column)
    
    def _create_exploration_plots(self, df: pd.DataFrame, target_column: str):
        """Create exploratory data analysis plots"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Performance Data Exploration', fontsize=16, fontweight='bold')
        
        # Target distribution
        axes[0, 0].hist(df[target_column], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Final Grades')
        axes[0, 0].set_xlabel('Final Grade')
        axes[0, 0].set_ylabel('Frequency')
        
        # Study hours vs Grade
        axes[0, 1].scatter(df['study_hours_per_week'], df[target_column], alpha=0.6, color='green')
        axes[0, 1].set_title('Study Hours vs Final Grade')
        axes[0, 1].set_xlabel('Study Hours per Week')
        axes[0, 1].set_ylabel('Final Grade')
        
        # Previous grade vs Final grade
        axes[0, 2].scatter(df['previous_grade'], df[target_column], alpha=0.6, color='orange')
        axes[0, 2].set_title('Previous Grade vs Final Grade')
        axes[0, 2].set_xlabel('Previous Grade')
        axes[0, 2].set_ylabel('Final Grade')
        
        # Absences vs Grade
        axes[1, 0].scatter(df['absences'], df[target_column], alpha=0.6, color='red')
        axes[1, 0].set_title('Absences vs Final Grade')
        axes[1, 0].set_xlabel('Number of Absences')
        axes[1, 0].set_ylabel('Final Grade')
        
        # Gender distribution
        gender_counts = df['gender'].value_counts()
        axes[1, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Gender Distribution')
        
        # Grade by parent education
        education_order = ['High School', 'Some College', 'Bachelor', 'Master', 'PhD']
        df_plot = df.copy()
        df_plot['parent_education'] = pd.Categorical(df_plot['parent_education'], 
                                                   categories=education_order, 
                                                   ordered=True)
        df_plot.boxplot(column=target_column, by='parent_education', ax=axes[1, 2])
        axes[1, 2].set_title('Final Grade by Parent Education')
        axes[1, 2].set_xlabel('Parent Education Level')
        axes[1, 2].set_ylabel('Final Grade')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'final_grade'):
        """Preprocess the data for machine learning"""
        
        self.target_column = target_column
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Create preprocessing pipelines
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Store feature names for later use
        self.feature_names = numeric_features + [
            f"{cat}_{val}" for cat in categorical_features
            for val in df[cat].unique()[1:]  # Skip first category (dropped)
        ]
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train multiple models and compare their performance"""
        
        print("=== TRAINING MODELS ===")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train and evaluate each model
        results = []
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                      cv=5, scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            model_results = {
                'Model': name,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'CV_RMSE': cv_rmse,
                'CV_STD': cv_std
            }
            
            results.append(model_results)
            self.model_performance[name] = model_results
            
            # Store the trained pipeline
            self.models[name] = pipeline
            
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  CV RMSE: {cv_rmse:.4f} (+/- {cv_std * 2:.4f})")
        
        # Convert results to DataFrame for easy comparison
        self.results_df = pd.DataFrame(results)
        
        # Find best model based on test RMSE
        best_idx = self.results_df['Test_RMSE'].idxmin()
        self.best_model_name = self.results_df.iloc[best_idx]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n=== BEST MODEL: {self.best_model_name} ===")
        print(f"Test RMSE: {self.results_df.iloc[best_idx]['Test_RMSE']:.4f}")
        print(f"Test R²: {self.results_df.iloc[best_idx]['Test_R2']:.4f}")
        
        return X_test, y_test
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_name: str = None):
        """Perform hyperparameter tuning for the best model"""
        
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"\n=== HYPERPARAMETER TUNING FOR {model_name} ===")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            },
            'Ridge Regression': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Support Vector Regression': {
                'regressor__C': [0.1, 1, 10],
                'regressor__gamma': ['scale', 'auto'],
                'regressor__kernel': ['rbf', 'linear']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return
        
        # Get the base pipeline
        base_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', self.models[model_name].named_steps['regressor'])
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update the best model
        self.models[model_name] = grid_search.best_estimator_
        self.best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def plot_model_comparison(self):
        """Plot model comparison results"""
        
        if not hasattr(self, 'results_df'):
            print("No model results available. Train models first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # RMSE comparison
        x_pos = np.arange(len(self.results_df))
        
        axes[0, 0].bar(x_pos, self.results_df['Test_RMSE'], alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Test RMSE')
        axes[0, 0].set_title('Test RMSE by Model')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(self.results_df['Model'], rotation=45, ha='right')
        
        # R² comparison
        axes[0, 1].bar(x_pos, self.results_df['Test_R2'], alpha=0.7, color='lightgreen')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Test R²')
        axes[0, 1].set_title('Test R² by Model')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(self.results_df['Model'], rotation=45, ha='right')
        
        # Cross-validation RMSE with error bars
        axes[1, 0].bar(x_pos, self.results_df['CV_RMSE'], 
                      yerr=self.results_df['CV_STD'], alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('CV RMSE')
        axes[1, 0].set_title('Cross-Validation RMSE by Model')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(self.results_df['Model'], rotation=45, ha='right')
        
        # Train vs Test RMSE
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, self.results_df['Train_RMSE'], 
                      width, label='Train RMSE', alpha=0.7, color='lightcoral')
        axes[1, 1].bar(x_pos + width/2, self.results_df['Test_RMSE'], 
                      width, label='Test RMSE', alpha=0.7, color='lightblue')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Train vs Test RMSE')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(self.results_df['Model'], rotation=45, ha='right')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 15):
        """Plot feature importance for the best model"""
        
        if self.best_model is None:
            print("No trained model available.")
            return
        
        # Get the trained regressor from the pipeline
        regressor = self.best_model.named_steps['regressor']
        
        # Check if the model has feature importance
        if not hasattr(regressor, 'feature_importances_'):
            print(f"{self.best_model_name} doesn't have feature importance.")
            return
        
        # Get feature names after preprocessing
        preprocessor = self.best_model.named_steps['preprocessor']
        feature_names = []
        
        # Get numeric feature names
        numeric_features = preprocessor.transformers_[0][2]
        feature_names.extend(numeric_features)
        
        # Get categorical feature names
        if len(preprocessor.transformers_) > 1:
            cat_transformer = preprocessor.transformers_[1][1]
            cat_feature_names = cat_transformer.get_feature_names_out()
            feature_names.extend(cat_feature_names)
        
        # Get feature importance
        importance = regressor.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Select top features
        top_indices = indices[:top_n]
        top_importance = importance[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importance, color='skyblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model"""
        
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        return self.best_model.predict(X)
    
    def predict_single_student(self, student_data: Dict[str, Any]) -> float:
        """Predict performance for a single student"""
        
        # Convert to DataFrame
        student_df = pd.DataFrame([student_data])
        
        # Make prediction
        prediction = self.predict(student_df)[0]
        
        return prediction
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessor"""
        
        if self.best_model is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'model_performance': self.model_performance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.preprocessor = model_data['preprocessor']
        self.feature_names = model_data['feature_names']
        self.target_column = model_data['target_column']
        self.model_performance = model_data['model_performance']
        
        print(f"Model loaded from {filepath}")  
        print(f"Best model: {self.best_model_name}")

def main():
    """Main execution function"""
    
    print("=== STUDENT PERFORMANCE PREDICTION SYSTEM ===")
    
    # Initialize the predictor
    predictor = StudentPerformancePredictor(random_state=42)
    
    # Load or generate data
    df = predictor.load_data()  # This will generate sample data
    
    # Explore the data
    predictor.explore_data(df, target_column='final_grade')
    
    # Preprocess the data
    X, y = predictor.preprocess_data(df, target_column='final_grade')
    
    # Train multiple models
    X_test, y_test = predictor.train_models(X, y, test_size=0.2)
    
    # Plot model comparison
    predictor.plot_model_comparison()
    
    # Perform hyperparameter tuning for the best model
    predictor.hyperparameter_tuning(X, y)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Make a sample prediction
    sample_student = {
        'gender': 'Female',
        'age': 17,
        'parent_education': 'Bachelor',
        'family_income': 'Medium',
        'study_hours_per_week': 6.5,
        'absences': 3,
        'previous_grade': 82.5,
        'extracurricular_activities': 2,
        'social_support': 'High',
        'health_status': 'Good',
        'internet_access': 1
    }
    
    predicted_grade = predictor.predict_single_student(sample_student)
    print(f"\n=== SAMPLE PREDICTION ===")
    print(f"Student profile: {sample_student}")
    print(f"Predicted final grade: {predicted_grade:.2f}")
    
    # Save the best model
    predictor.save_model('models/student_performance_model.pkl')
    
    print("\n=== TRAINING COMPLETED ===")
    print(f"Best model: {predictor.best_model_name}")
    print(f"Test RMSE: {predictor.model_performance[predictor.best_model_name]['Test_RMSE']:.4f}")
    print(f"Test R²: {predictor.model_performance[predictor.best_model_name]['Test_R2']:.4f}")

if __name__ == "__main__":
    main()