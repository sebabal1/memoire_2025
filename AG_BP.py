import random
import time
import copy
from typing import List, Tuple, Dict
from instances import get_instance_par_categorie

class BinPackingGeneticAlgorithm:
    def __init__(self, items: List[int], capacite_max_bac: int, 
                 taille_population: int = 50, max_generations: int = 100,
                 probabilite_croisement_pc: float = 0.8, probabilite_mutation_pm: float = 0.1, nb_elites: int = 1):
        self.items = items
        self.capacite_max_bac = capacite_max_bac
        self.n_items = len(items)
        self.taille_population = taille_population
        self.max_generations = max_generations
        self.probabilite_croisement_pc = probabilite_croisement_pc
        self.probabilite_mutation_pm = probabilite_mutation_pm
        self.nb_elites = nb_elites
        
        
        # Statistiques
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generations_computed = 0
        
    def solve(self) -> Tuple[List[List[int]], int, Dict]:
        """
        Résout le problème de bin packing avec l'algorithme génétique
        Retourne: (solution, nombre de bins, statistiques)
        """
        start_time = time.time()
        
        print(f"Initialisation de l'algorithme genetique...")
        print(f"Objets: {self.items}")
        print(f"Capacite des bacs: {self.capacite_max_bac}")
        print(f"Taille de population: {self.taille_population}")
        print(f"Generations max: {self.max_generations}")
        
        # 1. Initialisation de la population
        population = self._initialize_population()
        
        best_individual = None
        best_fitness = float('inf')
        
        # 2. Boucle évolutionnaire
        diversity_history = []
        for generation in range(self.max_generations):
            self.generations_computed = generation + 1

            unique_solutions = len({tuple(ind) for ind in population})
            diversity_history.append(unique_solutions)
            
            # 3. Évaluation de la population
            fitness_scores = [self._calcul_fitness(individual) for individual in population]

            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:self.nb_elites]
            elites = [copy.deepcopy(population[i]) for i in elite_indices]
            
            # Mise à jour du meilleur individu
            current_best_idx = fitness_scores.index(min(fitness_scores))
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = copy.deepcopy(population[current_best_idx])
                print(f"Génération {generation + 1}: Nouvelle meilleure solution = {best_fitness} bins")
            
            # Statistiques
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            if generation % 10 == 0:
                print(f"Génération {generation + 1}: Meilleur = {best_fitness}, Moyenne = {avg_fitness:.2f}")
            
            # 4. Sélection
            selected_population = self._selection(population, fitness_scores)
            
            # 5. Croisement
            offspring = self._crossover(selected_population)
            
            # 6. Mutation
            mutated_offspring = self._mutation(offspring)
            
            # 7. Remplacement (nouvelle génération)
            population = elites + mutated_offspring[:self.taille_population - self.nb_elites]
        
        end_time = time.time()
        
        # Conversion de la solution
        final_solution = self._chromosome_to_bins(best_individual)
        
        stats = {
            'cpu_time': end_time - start_time,
            'generations': self.generations_computed,
            'final_bins': best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'taille_population': self.taille_population,
            'probabilite_croisement_pc': self.probabilite_croisement_pc,
            'probabilite_mutation_pm': self.probabilite_mutation_pm,
            'diversity_history': diversity_history,
            'items': self.items,
        }
        
        return final_solution, int(best_fitness), stats
    
    def _initialize_population(self) -> List[List[int]]:
        """
        Initialise la population aléatoirement
        Chaque individu est représenté comme une permutation des items
        """
        population = []
        
        for _ in range(self.taille_population):
            # Créer une permutation aléatoire des indices des items
            individual = list(range(self.n_items))
            random.shuffle(individual)
            population.append(individual)
        
        return population
    
    def _calcul_fitness(self, individual: List[int]) -> float:
        """
        Fitness = (MaxBinPossibles + 1) - Nombre de bacs utilisés
        Plus grand = meilleur
        """
        bins = self._chromosome_to_bins(individual)
        max_bins = len(self.items)  # Max possible: 1 item par bin
        return (max_bins + 1) - len(bins)
    
    def _chromosome_to_bins(self, chromosome: List[int]) -> List[List[int]]:
        """
        Convertit un chromosome (permutation) en solution de bins
        Utilise l'algorithme First Fit
        """
        bins = []
        
        for item_idx in chromosome:
            item_size = self.items[item_idx]
            placed = False
            
            # Essayer de placer dans un bin existant
            for bin_items in bins:
                current_load = sum(self.items[idx] for idx in bin_items)
                if current_load + item_size <= self.capacite_max_bac:
                    bin_items.append(item_idx)
                    placed = True
                    break
            
            # Créer un nouveau bin si nécessaire
            if not placed:
                bins.append([item_idx])
        
        return bins
    
    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """
        Sélection par tournoi (sélection basique)
        """
        selected = []
        tournament_size = 3
        
        for _ in range(self.taille_population):
            # Sélection par tournoi
            tournament_indices = random.sample(range(len(population)), 
                                             min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            selected.append(copy.deepcopy(population[winner_idx]))
        
        return selected
    
    def _crossover(self, population: List[List[int]]) -> List[List[int]]:
        """
        Croisement en un point (Order Crossover - OX)
        Croisement basique adapté aux permutations
        """
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]
            
            if random.random() < self.probabilite_croisement_pc:
                child1, child2 = self._order_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
        
        return offspring[:self.taille_population]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) - croisement basique pour permutations
        """
        size = len(parent1)
        
        # Choisir deux points de croisement
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)
        
        # Créer les enfants
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copier la section entre les points de croisement
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Remplir le reste avec l'ordre de l'autre parent
        self._fill_remaining(child1, parent2, start, end)
        self._fill_remaining(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_remaining(self, child: List[int], parent: List[int], start: int, end: int):
        """Remplit les positions restantes du child avec l'ordre du parent"""
        child_set = set(child[start:end])
        parent_filtered = [item for item in parent if item not in child_set]
        
        # Remplir après la section copiée
        pos = end
        for item in parent_filtered:
            if pos >= len(child):
                pos = 0
            while child[pos] != -1:
                pos = (pos + 1) % len(child)
            child[pos] = item
            pos = (pos + 1) % len(child)
    
    def _mutation(self, population: List[List[int]]) -> List[List[int]]:
        """
        Mutation par échange (swap mutation) - mutation basique
        """
        mutated_population = []
        
        for individual in population:
            mutated_individual = copy.deepcopy(individual)
            
            if random.random() < self.probabilite_mutation_pm:
                # Échanger deux positions aléatoires
                pos1 = random.randint(0, len(mutated_individual) - 1)
                pos2 = random.randint(0, len(mutated_individual) - 1)
                mutated_individual[pos1], mutated_individual[pos2] = \
                    mutated_individual[pos2], mutated_individual[pos1]
            
            mutated_population.append(mutated_individual)
        
        return mutated_population

def print_solution(solution: List[List[int]], items: List[int], capacite_max_bac: int):
    """Affiche la solution de manière lisible"""
    print(f"\n=== SOLUTION FINALE ===")
    print(f"Nombre d'objets utilises: {len(solution)}")
    print(f"Capacite par bac: {capacite_max_bac}")
    
    total_items = 0
    for i, bin_items in enumerate(solution):
        bin_values = [items[idx] for idx in bin_items]
        total_size = sum(bin_values)
        utilization = (total_size / capacite_max_bac) * 100
        print(f"Bin {i+1}: {bin_values} (total: {total_size}/{capacite_max_bac}, utilisation: {utilization:.1f}%)")
        total_items += len(bin_items)
    
    print(f"Total items placés: {total_items}/{len(items)}")
    quality = calculate_solution_quality(solution, items, capacite_max_bac)
    print(f"Taux d'utilisation moyen : {quality:.2%}")

def calculate_solution_quality(solution: List[List[int]], items: List[int], capacite_max_bac: int) -> float:
    """Calcule le taux d'utilisation moyen des bacs"""
    if not solution:
        return 0.0
    total_utilization = 0
    for bin_items in solution:
        bin_weight = sum(items[idx] for idx in bin_items)
        total_utilization += bin_weight / capacite_max_bac
    return total_utilization / len(solution)

def print_statistics(stats: Dict):
    """Affiche les statistiques de performance"""
    print(f"\n=== STATISTIQUES ===")
    print(f"Temps CPU: {stats['cpu_time']:.3f} secondes")
    print(f"Generations calculees: {stats['generations']}")
    print(f"Bins dans la solution finale: {stats['final_bins']}")
    print(f"Taille de population: {stats['taille_population']}")
    print(f"Taux de croisement: {stats['probabilite_croisement_pc']}")
    print(f"Taux de mutation: {stats['probabilite_mutation_pm']}")
    
    if len(stats['best_fitness_history']) > 1:
        improvement = stats['best_fitness_history'][0] - stats['best_fitness_history'][-1]
        print(f"Amélioration totale: {improvement} bins")
    
    print(f"Meilleur fitness final: {stats['best_fitness_history'][-1]}")
    print(f"Fitness moyen final: {stats['avg_fitness_history'][-1]:.2f}")
    print(f"Diversite finale de la population: {stats['diversity_history'][-1]}")

def plot_evolution(stats: Dict):
    """Affiche l'évolution des fitness (optionnel - nécessite matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        generations = range(1, len(stats['best_fitness_history']) + 1)
        
       # Graphe principal : fitness
        plt.figure(figsize=(10, 5))
        plt.plot(generations, stats['best_fitness_history'], 'b-', label='Meilleur fitness', linewidth=2)
        plt.plot(generations, stats['avg_fitness_history'], 'r--', label='Fitness moyen', alpha=0.7)
        plt.xlabel('Génération')
        plt.ylabel('Nombre de bins')
        plt.title('Évolution du nombre de bins (fitness)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Graphe secondaire : diversité et/ou écart-type
        plt.figure(figsize=(10, 4))
        plt.plot(generations, stats['diversity_history'], 'g-.', label='Diversité population')
        plt.xlabel('Génération')
        plt.ylabel('Diversité de la population')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib non disponible pour l'affichage graphique")

# Exemple d'utilisation
if __name__ == "__main__":
    
    instances = get_instance_par_categorie('basic')

    for i, instance in enumerate(instances, 1):
        algo_genetique = BinPackingGeneticAlgorithm(
            items= instance['items'], 
            capacite_max_bac= instance['capacite_max_bac'],
            taille_population=4,
            max_generations=20,
            probabilite_croisement_pc=0.7,
            probabilite_mutation_pm=0.1,
            nb_elites=2
        )
        solution, n_bins, stats = algo_genetique.solve()
        print_solution(solution, instance['items'], instance['capacite_max_bac'])
        print_statistics(stats)
        #plot_evolution(stats)

    
