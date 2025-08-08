import random
import time
import copy
from typing import List, Tuple, Dict

class BinPackingGeneticAlgorithm:
    def __init__(self, items: List[int], bin_capacity: int, 
                 population_size: int = 50, max_generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        self.items = items
        self.bin_capacity = bin_capacity
        self.n_items = len(items)
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
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
        
        print(f"Initialisation de l'algorithme génétique...")
        print(f"Items: {self.items}")
        print(f"Capacité des bins: {self.bin_capacity}")
        print(f"Taille de population: {self.population_size}")
        print(f"Générations max: {self.max_generations}")
        
        # 1. Initialisation de la population
        population = self._initialize_population()
        
        best_individual = None
        best_fitness = float('inf')
        
        # 2. Boucle évolutionnaire
        for generation in range(self.max_generations):
            self.generations_computed = generation + 1
            
            # 3. Évaluation de la population
            fitness_scores = [self._fitness(individual) for individual in population]
            
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
            population = mutated_offspring
        
        end_time = time.time()
        
        # Conversion de la solution
        final_solution = self._chromosome_to_bins(best_individual)
        
        stats = {
            'cpu_time': end_time - start_time,
            'generations': self.generations_computed,
            'final_bins': best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'population_size': self.population_size,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate
        }
        
        return final_solution, int(best_fitness), stats
    
    def _initialize_population(self) -> List[List[int]]:
        """
        Initialise la population aléatoirement
        Chaque individu est représenté comme une permutation des items
        """
        population = []
        
        for _ in range(self.population_size):
            # Créer une permutation aléatoire des indices des items
            individual = list(range(self.n_items))
            random.shuffle(individual)
            population.append(individual)
        
        return population
    
    def _fitness(self, individual: List[int]) -> float:
        """
        Fonction de fitness : nombre de bins utilisés
        Plus petit = meilleur
        """
        bins = self._chromosome_to_bins(individual)
        return len(bins)
    
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
                if current_load + item_size <= self.bin_capacity:
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
        
        for _ in range(self.population_size):
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
            
            if random.random() < self.crossover_rate:
                child1, child2 = self._order_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
        
        return offspring[:self.population_size]
    
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
            
            if random.random() < self.mutation_rate:
                # Échanger deux positions aléatoires
                pos1 = random.randint(0, len(mutated_individual) - 1)
                pos2 = random.randint(0, len(mutated_individual) - 1)
                mutated_individual[pos1], mutated_individual[pos2] = \
                    mutated_individual[pos2], mutated_individual[pos1]
            
            mutated_population.append(mutated_individual)
        
        return mutated_population

def print_solution(solution: List[List[int]], items: List[int], bin_capacity: int):
    """Affiche la solution de manière lisible"""
    print(f"\n=== SOLUTION FINALE ===")
    print(f"Nombre de bins utilisés: {len(solution)}")
    print(f"Capacité par bin: {bin_capacity}")
    
    total_items = 0
    for i, bin_items in enumerate(solution):
        bin_values = [items[idx] for idx in bin_items]
        total_size = sum(bin_values)
        utilization = (total_size / bin_capacity) * 100
        print(f"Bin {i+1}: {bin_values} (total: {total_size}/{bin_capacity}, utilisation: {utilization:.1f}%)")
        total_items += len(bin_items)
    
    print(f"Total items placés: {total_items}/{len(items)}")

def print_statistics(stats: Dict):
    """Affiche les statistiques de performance"""
    print(f"\n=== STATISTIQUES ===")
    print(f"Temps CPU: {stats['cpu_time']:.3f} secondes")
    print(f"Générations calculées: {stats['generations']}")
    print(f"Bins dans la solution finale: {stats['final_bins']}")
    print(f"Taille de population: {stats['population_size']}")
    print(f"Taux de croisement: {stats['crossover_rate']}")
    print(f"Taux de mutation: {stats['mutation_rate']}")
    
    if len(stats['best_fitness_history']) > 1:
        improvement = stats['best_fitness_history'][0] - stats['best_fitness_history'][-1]
        print(f"Amélioration totale: {improvement} bins")
    
    print(f"Meilleur fitness final: {stats['best_fitness_history'][-1]}")
    print(f"Fitness moyen final: {stats['avg_fitness_history'][-1]:.2f}")

def plot_evolution(stats: Dict):
    """Affiche l'évolution des fitness (optionnel - nécessite matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        generations = range(1, len(stats['best_fitness_history']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, stats['best_fitness_history'], 'b-', label='Meilleur fitness', linewidth=2)
        plt.plot(generations, stats['avg_fitness_history'], 'r--', label='Fitness moyen', alpha=0.7)
        
        plt.xlabel('Génération')
        plt.ylabel('Nombre de bins')
        plt.title('Évolution de la fitness au cours des générations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ImportError:
        print("Matplotlib non disponible pour l'affichage graphique")

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple 1: Petit problème
    print("=== EXEMPLE 1 ===")
    items1 = [7, 5, 3, 3, 2, 2, 1]
    capacity1 = 10
    
    ga1 = BinPackingGeneticAlgorithm(
        items=items1, 
        bin_capacity=capacity1,
        population_size=30,
        max_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    solution1, n_bins1, stats1 = ga1.solve()
    print_solution(solution1, items1, capacity1)
    print_statistics(stats1)
    
    print("\n" + "="*60 + "\n")
    
    # Exemple 2: Problème plus complexe
    print("=== EXEMPLE 2 ===")
    items2 = [8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1]
    capacity2 = 12
    
    ga2 = BinPackingGeneticAlgorithm(
        items=items2, 
        bin_capacity=capacity2,
        population_size=50,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.15
    )
    
    solution2, n_bins2, stats2 = ga2.solve()
    print_solution(solution2, items2, capacity2)
    print_statistics(stats2)
    
    # Affichage optionnel de l'évolution
    plot_evolution(stats2)
    
    print("\n" + "="*60 + "\n")
    
    # Exemple 3: Test de performance
    print("=== EXEMPLE 3 - TEST DE PERFORMANCE ===")
    items3 = [9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1]
    capacity3 = 15
    
    ga3 = BinPackingGeneticAlgorithm(
        items=items3, 
        bin_capacity=capacity3,
        population_size=100,
        max_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    solution3, n_bins3, stats3 = ga3.solve()
    print_solution(solution3, items3, capacity3)
    print_statistics(stats3)
