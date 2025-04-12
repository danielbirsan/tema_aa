import numpy as np
import matplotlib.pyplot as plt


class AlgoritmGenetic:
    def __init__(self, dim_populatie, limita_inferioara, limita_superioara, coeficienti, precizie,
                 probabilitate_incrucisare, probabilitate_mutatie, numar_generatii,
                 procent_elitism=0.05, marime_turneu=3):
        self.dim_populatie = dim_populatie
        self.limita_inferioara = limita_inferioara
        self.limita_superioara = limita_superioara
        self.coef_a, self.coef_b, self.coef_c = coeficienti
        self.precizie = precizie
        self.probabilitate_incrucisare = probabilitate_incrucisare
        self.probabilitate_mutatie = probabilitate_mutatie
        self.numar_generatii = numar_generatii
        self.procent_elitism = procent_elitism
        self.marime_turneu = marime_turneu

        num_pasi = int((limita_superioara - limita_inferioara)
                       * (10 ** precizie))
        self.lungime_cromozom = int(np.ceil(np.log2(num_pasi + 1)))
        self.pondere = 2 ** np.arange(self.lungime_cromozom - 1, -1, -1)

        self.populatie = np.random.randint(
            0, 2, (dim_populatie, self.lungime_cromozom))
        self.log_detalii = []

    def cromozom_la_str(self, cromozom):
        return ''.join(cromozom.astype(str))

    def decodifica_populatie(self, populatie):
        valori_decodificate = populatie.dot(self.pondere)
        x = self.limita_inferioara + valori_decodificate * \
            (self.limita_superioara - self.limita_inferioara) / \
            (2**self.lungime_cromozom - 1)
        return x, valori_decodificate

    def functie_fitness(self, x):
        return self.coef_a * x**2 + self.coef_b * x + self.coef_c

    def evalueaza(self, populatie):
        x, _ = self.decodifica_populatie(populatie)
        fitness = self.functie_fitness(x)
        return x, fitness

    def selecteaza_turneu(self, populatie, fitness):

        num_elit = max(
            1, int(np.ceil(self.dim_populatie * self.procent_elitism)))
        indici_ordonati = np.argsort(fitness)[::-1]
        elitism = populatie[indici_ordonati[:num_elit]]
        self.log_detalii.append(
            f"Elitism extins: se păstrează {num_elit} indivizi.")
        selectie_turneu = []

        num_selectii = self.dim_populatie - num_elit
        for i in range(num_selectii):
            participanti = np.random.choice(
                self.dim_populatie, self.marime_turneu, replace=True)

            best_index = participanti[np.argmax(fitness[participanti])]
            selectie_turneu.append(populatie[best_index])
            self.log_detalii.append(
                f"Turneu {i+1}: participanți {participanti.tolist()} => câștigător: {self.cromozom_la_str(populatie[best_index])}")
        noua_populatie = np.vstack([elitism, np.array(selectie_turneu)])
        return noua_populatie

    def incrucisare_doua_puncte(self, parinte1, parinte2):
        punctele = np.sort(np.random.choice(
            range(1, self.lungime_cromozom), size=2, replace=False))
        p1, p2 = punctele
        copil1 = parinte1.copy()
        copil2 = parinte2.copy()
        copil1[p1:p2], copil2[p1:p2] = parinte2[p1:p2], parinte1[p1:p2]
        return copil1, copil2, (p1, p2)

    def incrucisare_trei(self, parinte1, parinte2, parinte3):

        punctele = np.sort(np.random.choice(
            range(1, self.lungime_cromozom), size=2, replace=False))
        p1, p2 = punctele
        copil1 = np.concatenate(
            [parinte1[:p1], parinte2[p1:p2], parinte3[p2:]])
        copil2 = np.concatenate(
            [parinte2[:p1], parinte3[p1:p2], parinte1[p2:]])
        copil3 = np.concatenate(
            [parinte3[:p1], parinte1[p1:p2], parinte2[p2:]])
        log_text = (f"Incrucișare în 3 între {self.cromozom_la_str(parinte1)}, "
                    f"{self.cromozom_la_str(parinte2)}, {self.cromozom_la_str(parinte3)} la punctele {p1} și {p2} => "
                    f"{self.cromozom_la_str(copil1)}, {self.cromozom_la_str(copil2)}, {self.cromozom_la_str(copil3)}")
        return [copil1, copil2, copil3], log_text

    def incrucisare(self, grupa):

        log_incrucisare = ""
        if grupa.shape[0] == 2:
            copil1, copil2, puncte = self.incrucisare_doua_puncte(
                grupa[0], grupa[1])
            log_incrucisare = (f"Incrucișare între {self.cromozom_la_str(grupa[0])} și {self.cromozom_la_str(grupa[1])} "
                               f"la punctele {puncte[0]}, {puncte[1]} => {self.cromozom_la_str(copil1)}, {self.cromozom_la_str(copil2)}")
            return np.array([copil1, copil2]), log_incrucisare
        elif grupa.shape[0] == 3:
            copii, log_incrucisare = self.incrucisare_trei(
                grupa[0], grupa[1], grupa[2])
            return np.array(copii), log_incrucisare
        else:
            return grupa, "Grup invalid pentru încrucișare"

    def mutatie(self, populatie):
        log_mutatie = []
        masca_mutatie = np.random.rand(
            *populatie.shape) < self.probabilitate_mutatie
        for i in range(populatie.shape[0]):
            stare_inainte = self.cromozom_la_str(populatie[i])
            indici = np.where(masca_mutatie[i])[0]
            if indici.size:
                populatie[i, indici] = 1 - populatie[i, indici]
                stare_dupa = self.cromozom_la_str(populatie[i])
                log_mutatie.append(
                    f"{stare_inainte} -> mutație la indici {indici.tolist()} -> {stare_dupa}")
            else:
                log_mutatie.append(f"{stare_inainte} -> fără mutație")
        return populatie, log_mutatie

    def log_populatie_initiala(self, populatie, x, fitness):
        self.log_detalii.append("Generația 0 - Populația inițială:")
        self.log_detalii.append("Index\tCromozom\t\tX\t\tFitness")
        for i, (crom, val, fit) in enumerate(zip(populatie, x, fitness)):
            self.log_detalii.append(
                f"{i}\t{self.cromozom_la_str(crom)}\t{val:.6f}\t{fit:.4f}")
        self.log_detalii.append("")

    def ruleaza(self):
        istoric_max, istoric_medie = [], []

        x, fitness = self.evalueaza(self.populatie)
        self.log_populatie_initiala(self.populatie, x, fitness)
        self.log_detalii.append(
            "Aplicare selecție turneu cu elitism extins (Generația 0):")
        self.log_detalii.append("")
        grup_selectie = self.selecteaza_turneu(self.populatie, fitness)

        for gen in range(1, self.numar_generatii + 1):
            x, fitness = self.evalueaza(grup_selectie)
            max_fit = np.max(fitness)
            medie_fit = np.mean(fitness)
            istoric_max.append(max_fit)
            istoric_medie.append(medie_fit)
            self.log_detalii.append(
                f"Generația {gen}: Max Fitness = {max_fit:.4f}, Media Fitness = {medie_fit:.4f}")

            grup_nou = self.selecteaza_turneu(grup_selectie, fitness)

            log_incrucisare = []
            noua_populatie = [grup_nou[0]]
            i = 1

            if (grup_nou.shape[0] - 1) % 2 == 1:
                limita = grup_nou.shape[0] - 3
            else:
                limita = grup_nou.shape[0]
            while i < limita:
                copii, log_text = self.incrucisare(grup_nou[i:i+2])
                log_incrucisare.append(log_text)
                noua_populatie.extend(copii)
                i += 2
            if i < grup_nou.shape[0]:
                copii_trei, log_text = self.incrucisare(
                    grup_nou[i:grup_nou.shape[0]])
                log_incrucisare.append(log_text)
                noua_populatie.extend(copii_trei)
            self.log_detalii.append("Detalii încrucișare:")
            self.log_detalii.extend(log_incrucisare)

            noua_populatie = np.array(noua_populatie)[:self.dim_populatie]
            # Mutație
            noua_populatie, log_mut = self.mutatie(noua_populatie)
            self.log_detalii.append("Detalii mutație:")
            self.log_detalii.extend(log_mut)

            grup_selectie = noua_populatie.copy()

        with open("Evolutie.txt", "w") as f:
            f.write("\n".join(self.log_detalii))
            print("Log-ul a fost salvat în Evolutie.txt")

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.numar_generatii + 1),
                 istoric_max, label="Max Fitness", marker='o')
        plt.plot(range(1, self.numar_generatii + 1),
                 istoric_medie, label="Media Fitness", marker='x')
        plt.xlabel("Generația")
        plt.ylabel("Fitness")
        plt.title("Evoluția Fitness-ului Populației")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    dim_pop = int(input("Introduceți dimensiunea populației: "))
    limita_inf = float(input("Introduceți limita inferioară (a): "))
    limita_sup = float(input("Introduceți limita superioară (b): "))
    coef_a = float(input("Introduceți coeficientul a pentru f(x)=ax^2+bx+c: "))
    coef_b = float(input("Introduceți coeficientul b pentru f(x)=ax^2+bx+c: "))
    coef_c = float(input("Introduceți coeficientul c pentru f(x)=ax^2+bx+c: "))
    precizie = int(input("Introduceți precizia (numărul de zecimale): "))
    prob_incr = float(input(
        "Introduceți probabilitatea de încrucișare (în procente, ex: 25): ")) / 100.0
    prob_mut = float(
        input("Introduceți probabilitatea de mutație (în procente, ex: 1): ")) / 100.0
    nr_generatii = int(input("Introduceți numărul de generații: "))

    algoritm = AlgoritmGenetic(dim_pop, limita_inf, limita_sup, (coef_a, coef_b, coef_c),
                               precizie, prob_incr, prob_mut, nr_generatii,
                               procent_elitism=0.05, marime_turneu=3)
    algoritm.ruleaza()
