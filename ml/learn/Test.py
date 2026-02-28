import pandas as pd
import joblib
import os

def test_saved_model(val_csv_path, model_path, output_file):
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
    
    # Deschidem fișierul în modul scriere
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("==================================================\n")
        file.write("         RAPORT TESTARE MODEL V2X AI\n")
        file.write("==================================================\n\n")

        # 1. Acuratețea generală
        accuracy = model.score(X_val, y_val)
        mesaj_acuratete = f"[Test] Acuratețe pe date noi (Validare): {accuracy * 100:.2f}%\n"
        print(mesaj_acuratete)
        file.write(mesaj_acuratete + "\n")
        
        # 2. Extragem Coeficientul pentru 30 de situații
        file.write("--- DEMONSTRAȚIE COEFICIENȚI (30 scenarii) ---\n")
        for i in range(30):
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
                
    print(f"\n✅ Gata! Rezultatele au fost salvate în '{output_file}'.")

if __name__ == "__main__":
    cale_ml = os.path.abspath(os.path.dirname(__file__))
    
    # Mapăm exact folderul generated
    cale_csv = os.path.join(cale_ml, "..", "generated", "val_dataset.csv")
    cale_model = os.path.join(cale_ml, "..", "generated", "traffic_model.pkl")
    cale_out = os.path.join(cale_ml, "..", "generated", "rezultate_test.txt")
    
    test_saved_model(cale_csv, cale_model, cale_out)