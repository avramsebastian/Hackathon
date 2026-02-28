import pygame
import math
import random
import sys
import time

# ==========================================
# 1. CANALUL V2X
# ==========================================
class CanalV2X:
    def __init__(self):
        self.stari_vehicule = {}

    def actualizeaza_stare(self, id_agent, stare):
        self.stari_vehicule[id_agent] = stare

    def citeste_stari(self):
        return self.stari_vehicule

# ==========================================
# 2. DEFINIREA SENSULUI GIRATORIU
# ==========================================
PUNCTE_ACCES = {
    "EST": 0,
    "SUD": 90,
    "VEST": 180,
    "NORD": 270
}

class AgentGiratoriu:
    def __init__(self, id_agent, intrare, iesire, culoare):
        self.id = id_agent
        self.culoare = culoare
        
        self.unghi = PUNCTE_ACCES[intrare]
        self.unghi_iesire = PUNCTE_ACCES[iesire]
        self.banda = 0 # 0 = Banda Exterioara, 1 = Banda Interioara
        
        self.activa = True
        self.viteza = 0
        self.mesaj = "Analizeaza..."
        
        self.genom = [random.uniform(-10, 10) for _ in range(4)]

    def incarca_creier(self, genom_bun):
        self.genom = genom_bun

    def partajeaza_date(self, canal_v2x):
        if self.activa:
            canal_v2x.actualizeaza_stare(self.id, {
                "unghi": self.unghi,
                "banda": self.banda,
                "viteza": self.viteza
            })

    def ia_decizie(self, date_v2x):
        if not self.activa: return

        dist_pana_la_iesire = (self.unghi_iesire - self.unghi) % 360
        if dist_pana_la_iesire < 15 and self.banda == 0:
            self.activa = False
            self.mesaj = "A iesit din sens!"
            return

        alte_masini = [date for id_v, date in date_v2x.items() if id_v != self.id]
        scoruri = [0, 0, 0, 0]
        
        for actiune in range(4):
            viitor_unghi = self.unghi
            viitoare_banda = self.banda
            
            if actiune == 0: 
                scoruri[actiune] += self.genom[3] 
            elif actiune == 1: 
                viitor_unghi = (self.unghi + 10) % 360
                scoruri[actiune] += self.genom[0] 
            elif actiune == 2 and self.banda == 0: 
                viitoare_banda = 1
                scoruri[actiune] += self.genom[2]
            elif actiune == 3 and self.banda == 1: 
                viitoare_banda = 0
                scoruri[actiune] -= self.genom[2] 
            else:
                scoruri[actiune] = -float('inf') 
                continue

            for alta in alte_masini:
                dist_unghiulara = abs(viitor_unghi - alta["unghi"])
                if dist_unghiulara < 15 and viitoare_banda == alta["banda"]:
                    scoruri[actiune] += self.genom[1] 
                    
            viitoare_dist_iesire = (self.unghi_iesire - viitor_unghi) % 360
            if viitoare_dist_iesire < 45 and viitoare_banda == 1:
                scoruri[actiune] -= 10 

        cea_mai_buna_actiune = scoruri.index(max(scoruri))

        if cea_mai_buna_actiune == 0:
            self.viteza = 0
            self.mesaj = "Franeaza / Cedeaza"
        elif cea_mai_buna_actiune == 1:
            self.viteza = 10 
            self.unghi = (self.unghi + self.viteza) % 360
            self.mesaj = "Avanseaza"
        elif cea_mai_buna_actiune == 2:
            self.banda = 1
            self.mesaj = "Trece pe banda 2"
        elif cea_mai_buna_actiune == 3:
            self.banda = 0
            self.mesaj = "Trece pe banda 1"

# ==========================================
# 3. ANTRENAMENTUL AI (Genetic Algorithm)
# ==========================================
def evalueaza_fitness(genom, date_intrare):
    canal = CanalV2X()
    m1 = AgentGiratoriu("M1", date_intrare['m1_in'], date_intrare['m1_out'], None)
    m2 = AgentGiratoriu("M2", date_intrare['m2_in'], date_intrare['m2_out'], None)
    m1.incarca_creier(genom)
    m2.incarca_creier(genom)
    
    fitness = 0
    pasi = 0
    while (m1.activa or m2.activa) and pasi < 50:
        m1.partajeaza_date(canal)
        m2.partajeaza_date(canal)
        stare = canal.citeste_stari()
        m1.ia_decizie(stare)
        m2.ia_decizie(stare)
        
        if abs(m1.unghi - m2.unghi) < 10 and m1.banda == m2.banda and m1.activa and m2.activa:
            return -2000
            
        fitness -= 1 
        pasi += 1

    if not m1.activa: fitness += 1000
    if not m2.activa: fitness += 1000
    return fitness

def antreneaza_ai(date_intrare, generatii=20, marime_pop=40):
    populatie = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(marime_pop)]
    for gen in range(generatii):
        scoruri = [(genom, evalueaza_fitness(genom, date_intrare)) for genom in populatie]
        scoruri.sort(key=lambda x: x[1], reverse=True)
        elite = [s[0] for s in scoruri[:10]]
        urmatoarea_gen = elite[:]
        while len(urmatoarea_gen) < marime_pop:
            parinte = random.choice(elite)
            copil = [gena + random.uniform(-2, 2) for gena in parinte]
            urmatoarea_gen.append(copil)
        populatie = urmatoarea_gen
    return scoruri[0][0]

# ==========================================
# 4. INTERFATA PRINCIPALA PYGAME
# ==========================================
def ruleaza_aplicatia():
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    ecran = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulator V2X - Sens Giratoriu AI")
    
    font = pygame.font.SysFont("Arial", 16)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    font_mare = pygame.font.SysFont("Arial", 26, bold=True)
    
    CENTRUL = (300, 300)
    RAZA_EXT = 150
    RAZA_INT = 100

    # Definim butoanele clickable pe ecran
    butoane = {
        "NORD": pygame.Rect(260, 20, 80, 50),
        "SUD": pygame.Rect(260, 530, 80, 50),
        "VEST": pygame.Rect(20, 275, 80, 50),
        "EST": pygame.Rect(500, 275, 80, 50)
    }

    # Starea aplicatiei: 0=Alege M1 Start, 1=Alege M1 Iesire, 2=Alege M2 Start, 3=Alege M2 Iesire, 4=Antrenament, 5=Simulare
    stare_app = 0 
    date_intrare = {}
    
    m1_culoare = (50, 150, 255)
    m2_culoare = (255, 50, 50)
    
    mesaje_stare = [
        ("CLICK pe START pt Masina ALBASTRĂ", m1_culoare),
        ("CLICK pe IEȘIRE pt Masina ALBASTRĂ", m1_culoare),
        ("CLICK pe START pt Masina ROȘIE", m2_culoare),
        ("CLICK pe IEȘIRE pt Masina ROȘIE", m2_culoare)
    ]

    canal = CanalV2X()
    masini = []

    rulare = True
    while rulare:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # Logica de CLICK
            if event.type == pygame.MOUSEBUTTONDOWN and stare_app < 4:
                pos = pygame.mouse.get_pos()
                for nume_buton, rect in butoane.items():
                    if rect.collidepoint(pos):
                        if stare_app == 0: date_intrare['m1_in'] = nume_buton
                        elif stare_app == 1: date_intrare['m1_out'] = nume_buton
                        elif stare_app == 2: date_intrare['m2_in'] = nume_buton
                        elif stare_app == 3: date_intrare['m2_out'] = nume_buton
                        stare_app += 1
                        break

        # DESENARE FUNDAL SI SENS GIRATORIU
        ecran.fill((30, 30, 30))
        pygame.draw.circle(ecran, (80, 80, 80), CENTRUL, RAZA_EXT + 25) 
        pygame.draw.circle(ecran, (255, 255, 255), CENTRUL, RAZA_EXT, 2) 
        pygame.draw.circle(ecran, (200, 200, 200), CENTRUL, RAZA_INT + 25, 2) 
        pygame.draw.circle(ecran, (40, 100, 40), CENTRUL, RAZA_INT - 25) 
        
        pygame.draw.line(ecran, (255,255,255), (300, 0), (300, 125), 2) 
        pygame.draw.line(ecran, (255,255,255), (300, 475), (300, 600), 2) 
        pygame.draw.line(ecran, (255,255,255), (0, 300), (125, 300), 2) 
        pygame.draw.line(ecran, (255,255,255), (475, 300), (600, 300), 2) 

        # FAZA DE SELECTIE (CLICK-URI)
        if stare_app < 4:
            # Desenam Butoanele
            for nume_buton, rect in butoane.items():
                pygame.draw.rect(ecran, (200, 200, 0), rect, border_radius=5)
                text_btn = font_bold.render(nume_buton, True, (0,0,0))
                ecran.blit(text_btn, (rect.x + 15, rect.y + 12))

            # Desenam Panoul de Instructiuni
            pygame.draw.rect(ecran, (20, 20, 20), (600, 0, 300, 600))
            text_instructiune = font_mare.render(mesaje_stare[stare_app][0], True, mesaje_stare[stare_app][1])
            
            # Randare text pe mai multe linii pt panou
            cuvinte = mesaje_stare[stare_app][0].split(" pt ")
            ecran.blit(font_bold.render(cuvinte[0], True, (255,255,255)), (620, 100))
            ecran.blit(font_bold.render("pt " + cuvinte[1], True, mesaje_stare[stare_app][1]), (620, 130))

        # FAZA DE ANTRENAMENT AI
        elif stare_app == 4:
            pygame.draw.rect(ecran, (20, 20, 20), (600, 0, 300, 600))
            ecran.blit(font_mare.render("AI ÎNVAȚĂ RUTA...", True, (0, 255, 100)), (620, 100))
            ecran.blit(font.render("Asteapta 2-3 secunde", True, (200, 200, 200)), (620, 140))
            pygame.display.flip() # Fortam afisarea mesajului inainte de a bloca thread-ul cu antrenamentul
            
            # Rulam algoritmul genetic
            creier_antrenat = antreneaza_ai(date_intrare)
            
            # Initializam masinile cu creierul antrenat
            m1 = AgentGiratoriu("M1", date_intrare['m1_in'], date_intrare['m1_out'], m1_culoare)
            m2 = AgentGiratoriu("M2", date_intrare['m2_in'], date_intrare['m2_out'], m2_culoare)
            m1.incarca_creier(creier_antrenat)
            m2.incarca_creier(creier_antrenat)
            masini = [m1, m2]
            
            stare_app = 5 # Trecem la simulare

        # FAZA DE SIMULARE LIVE
        elif stare_app == 5:
            activa = False
            for m in masini:
                m.partajeaza_date(canal)
                if m.activa: activa = True
            
            stare = canal.citeste_stari()
            for m in masini:
                m.ia_decizie(stare)

            for m in masini:
                if m.activa:
                    raza_curenta = RAZA_EXT if m.banda == 0 else RAZA_INT
                    radiani = math.radians(m.unghi)
                    x = CENTRUL[0] + raza_curenta * math.cos(radiani)
                    y = CENTRUL[1] + raza_curenta * math.sin(radiani)
                    
                    pygame.draw.circle(ecran, m.culoare, (int(x), int(y)), 12)
                    ecran.blit(font.render(m.id, True, (255,255,255)), (x - 10, y - 25))

            # Dashboard V2X LIVE
            pygame.draw.rect(ecran, (20, 20, 20), (600, 0, 300, 600))
            ecran.blit(font_bold.render("DASHBOARD V2X (2 BENZI)", True, (0, 255, 100)), (620, 20))
            
            y_pos = 70
            for m in masini:
                randuri = [
                    f"Agent: {m.id}",
                    f"Banda: {'Exterioara' if m.banda == 0 else 'Interioara'}",
                    f"Pozitie (Grade): {int(m.unghi)}",
                    f"Actiune: {m.mesaj}",
                    "-"*30
                ]
                for r in randuri:
                    culoare_txt = m.culoare if "Agent" in r else (220, 220, 220)
                    ecran.blit(font.render(r, True, culoare_txt), (620, y_pos))
                    y_pos += 25

            if not activa:
                ecran.blit(font_bold.render("SIMULARE FINALIZATA!", True, (255, 200, 0)), (620, y_pos + 20))
                pygame.display.flip()
                time.sleep(3)
                # Reseteaza jocul pentru a putea alege alte rute (sau poti pune rulare = False ca sa inchizi)
                stare_app = 0 
                date_intrare = {}
                
            pygame.time.delay(300) # Controlul vitezei (300ms pe frame)

        pygame.display.flip()

if __name__ == "__main__":
    ruleaza_aplicatia()