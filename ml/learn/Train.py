import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier

class TrafficModelTrainer:
    def __init__(self):
        # Inițializăm un model Deep Learning (Multi-Layer Perceptron)
        # hidden_layer_sizes=(64, 32) înseamnă 2 straturi ascunse: primul cu 64 neuroni, al doilea cu 32.
        # activation='relu' este standardul în industrie pentru decizii rapide.
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            solver='adam', 
            max_iter=1500, # Am crescut iterațiile pentru că rețelele neuronale învață puțin mai greu
            random_state=42 # Pentru a avea rezultate reproductibile
        )

    def train_from_csv(self, train_csv_path, model_save_path="ML/generated/traffic_model.pkl"):
        print(f"[AI] Se încarcă datele de antrenament din '{train_csv_path}'...")
        
        # Citim datele folosind Pandas
        df_train = pd.read_csv(train_csv_path)
        
        # Separăm vectorul de parametri (X) de eticheta corectă (y)
        X_train = df_train.drop('label', axis=1).values
        y_train = df_train['label'].values
        
        print("[AI] Rețeaua Neuronală se antrenează pe datele primite (Deep Learning)...")
        self.model.fit(X_train, y_train)
        
        # Verificăm acuratețea
        accuracy = self.model.score(X_train, y_train)
        print(f"[AI] Antrenament complet! Acuratețe internă: {accuracy * 100:.2f}%")
        
        # Salvăm rețeaua neuronală antrenată pe disc
        joblib.dump(self.model, model_save_path)
        print(f"[AI] Modelul Deep Learning a fost salvat cu succes în '{model_save_path}'.\n")

if __name__ == "__main__":
    trainer = TrafficModelTrainer()
    # Apelăm funcția dându-i CSV-ul generat anterior
    trainer.train_from_csv("ML/generated/train_dataset.csv")