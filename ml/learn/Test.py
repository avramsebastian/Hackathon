import pandas as pd
import joblib

def test_saved_model(val_csv_path, model_path="ML/generated/traffic_model.pkl", output_file="ML/generated/rezultate_test.txt"):
    print(f"[Test] Se încarcă modelul antrenat din '{model_path}'...")
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("Eroare: Nu am găsit modelul! Te rog rulează Train.py mai întâi.")
        return

    print(f"[Test] Se încarcă datele de validare din '{val_csv_path}'...")
    df_val = pd.read_csv(val_csv_path)
    
    X_val = df_val.drop('label', axis=1).values
    y_val = df_val['label'].values
    
    # Deschidem fișierul în modul scriere ("w")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("==================================================\n")
        file.write("         RAPORT TESTARE MODEL V2X AI\n")
        file.write("==================================================\n\n")

        # 1. Acuratețea generală pe setul de validare
        accuracy = model.score(X_val, y_val)
        file.write(f"[Test] Acuratețe pe date noi (Validare): {accuracy * 100:.2f}%\n\n")
        
        # 2. Extragem Coeficientul pentru cele 30 de situații
        file.write("--- DEMONSTRAȚIE COEFICIENȚI (500 scenarii) ---\n")
        
        for i in range(500):
            features = X_val[i].reshape(1, -1)
            label_corect = y_val[i]
            
            probabilitati = model.predict_proba(features)[0]
            coef_go = probabilitati[1]
            
            raspuns_real = 'GO (Accelerează)' if label_corect == 1 else 'STOP (Frânează)'
            
            file.write(f"\nScenariul {i+1} (Răspunsul corect din trafic era: {raspuns_real}):\n")
            file.write(f" -> Coeficient ACCELERARE generat de AI: {coef_go * 100:.1f}%\n")
            file.write(f" -> Coeficient OPRIRE generat de AI:     {(1 - coef_go) * 100:.1f}%\n")
            
            if coef_go > 0.5:
                file.write(" -> Decizie Model: AI-ul alege să MEARGĂ.\n")
            else:
                file.write(" -> Decizie Model: AI-ul alege să OPREASCĂ.\n")
                
    print(f"\n✅ Gata! Toate rezultatele au fost salvate cu succes în fișierul '{output_file}'.")

if __name__ == "__main__":
    test_saved_model("ML/generated/val_dataset.csv")