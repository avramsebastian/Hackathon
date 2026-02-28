import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class TrafficModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=25, 
            max_depth=12,       
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

    def train(self, train_csv_path: str, model_save_path: str):
        print(f"[Model Training] Încărcare date din '{os.path.basename(train_csv_path)}'...")
        df_train = pd.read_csv(train_csv_path)
        
        stop_count = len(df_train[df_train['label'] == 0])
        go_count = len(df_train[df_train['label'] == 1])
        print(f"[Dataset Stats] Eșantioane GO: {go_count} | Eșantioane STOP: {stop_count}")
        
        X = df_train.drop('label', axis=1).values
        y = df_train['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        print(f"[Model Training] Start antrenament pe {X.shape[1]} parametri...")
        self.model.fit(X_train, y_train)
        
        acc_train = self.model.score(X_train, y_train)
        acc_test = self.model.score(X_test, y_test)
        
        print("\n--- RAPORT DE PERFORMANȚĂ ---")
        print(f"Acuratețe Învățare (Train): {acc_train * 100:.2f}%")
        print(f"Acuratețe Validare (Test):  {acc_test * 100:.2f}%\n")
        
        joblib.dump(self.model, model_save_path)
        print(f"[Model Training] Model serializat și salvat în '{os.path.basename(model_save_path)}'.")

if __name__ == "__main__":
    cale_ml = os.path.abspath(os.path.dirname(__file__))
    cale_csv = os.path.join(cale_ml, "..", "generated", "train_dataset.csv")
    cale_model = os.path.join(cale_ml, "..", "generated", "traffic_model.pkl")
    
    trainer = TrafficModelTrainer()
    trainer.train(cale_csv, cale_model)