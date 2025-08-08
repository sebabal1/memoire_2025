import time
import copy
import random
from typing import List, Tuple, Dict, Set

class BinPackingTabuSearch:
    def __init__(self, items: List[int], bin_capacity: int,
                 max_iterations: int = 100, tabu_tenure: int = 7,
                 max_no_improvement: int = 20):
        self.items = items
        self.bin_capacity = bin_capacity
        self.n_items = len(items)
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.max_no_improvement = max_no_improvement
        
        # Structures de données pour la recherche tabou
        self.tabu_list = []  # Liste des mouvements interdits
        self.best_solution = None
        self.best_cost = float('inf')
        self.current_solution = None
        self.current_cost = float('inf')
        
        # Statistiques
        self.iterations_performed = 0
        self.cost_history = []
        self.best_cost_history = []
        self.tabu_hits = 0
        self.moves_evaluated = 0
        
    def solve(self) -> Tuple[List[List[int]], int, Dict]:
        """
        Résout le problème de bin packing avec la recherche tabou
        Retourne: (solution, nombre de bins, statistiques)
        """
        start_time = time.time()
        
        print(f"Initialisation de la recherche tabou...")
        print(f"Items: {self.items}")
        print(f"Capacité des bins: {self.bin_capacity}")
        print(f"Itérations max: {self.max_iterations}")
        print(f"Tenure tabou: {self.tabu_tenure}")
        
        # 1. Solution initiale
        self.current_solution = self._generate_initial_solution()
        self.current_cost = self._evaluate_solution(self.current_solution)
        
        # Initialiser la meilleure solution
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.current_cost
        
        print(f"Solution initiale: {self.best_cost} bins")
        
        no_improvement_count = 0
        
        # 2. Boucle principale de recherche tabou
        for iteration in range(self.max_iterations):
            self.iterations_performed = iteration + 1
            
            # 3. Générer le voisinage
            neighborhood = self._generate_neighborhood(self.current_solution)
            
            # 4. Évaluer les solutions du voisinage
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None
            
            for neighbor, move in neighborhood:
                self.moves_evaluated += 1
                neighbor_cost = self._evaluate_solution(neighbor)
                
                # Vérifier si le mouvement est tabou
                is_tabu = self._is_tabu(move)
                
                # Critère d'aspiration : accepter si meilleur que la meilleure solution globale
                aspiration_criterion = neighbor_cost < self.best_cost
                
                # Accepter le mouvement si non-tabou ou si critère d'aspiration satisfait
                if (not is_tabu or aspiration_criterion) and neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_move = move
                
                if is_tabu and not aspiration_criterion:
                    self.tabu_hits += 1
            
            # 5. Mise à jour de la solution courante
            if best_neighbor is not None:
                self.current_solution = best_neighbor
                self.current_cost = best_neighbor_cost
                
                # Ajouter le mouvement à la liste tabou
                self._add_to_tabu_list(best_move)
                
                # 6. Mise à jour de la meilleure solution
                if self.current_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(self.current_solution)
                    self.best_cost = self.current_cost
                    no_improvement_count = 0
                    print(f"Itération {iteration + 1}: Nouvelle meilleure solution = {self.best_cost} bins")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Enregistrer l'historique
            self.cost_history.append(self.current_cost)
            self.best_cost_history.append(self.best_cost)
            
            if iteration % 10 == 0:
                print(f"Itération {iteration + 1}: Actuel = {self.current_cost}, Meilleur = {self.best_cost}")
            
            # 7. Critère d'arrêt par stagnation
            if no_improvement_count >= self.max_no_improvement:
                print(f"Arrêt par stagnation après {no_improvement_count} itérations sans amélioration")
                break
        
        end_time = time.time()
        
        # Conversion de la solution
        final_solution = self._convert_solution_format(self.best_solution)
        
        stats = {
            'cpu_time': end_time - start_time,
            'iterations': self.iterations_performed,
            'final_bins': self.best_cost,
            'cost_history': self.cost_history,
            'best_cost_history': self.best_cost_history,
            'tabu_tenure': self.tabu_tenure,
            'tabu_hits': self.tabu_hits,
            'moves_evaluated': self.moves_evaluated,
            'max_no_improvement': self.max_no_improvement
        }
        
        return final_solution, int(self.best_cost), stats
    
    def _generate_initial_solution(self) -> List[int]:
        """
        Génère une solution initiale avec First Fit Decreasing
        Représentation: liste où solution[i] = numéro du bin pour l'item i
        """
        # Trier les items par taille décroissante avec leurs indices
        sorted_items = sorted(enumerate(self.items), key=lambda x: x[1], reverse=True)
        
        solution = [0] * self.n_items
        bin_loads = []
        current_bin = 0
        
        for item_idx, item_size in sorted_items:
            placed = False
            
            # Essayer de placer dans un bin existant
            for bin_num in range(current_bin + 1):
                if bin_num >= len(bin_loads):
                    bin_loads.append(0)
                
                if bin_loads[bin_num] + item_size <= self.bin_capacity:
                    solution[item_idx] = bin_num
                    bin_loads[bin_num] += item_size
                    placed = True
                    break
            
            # Créer un nouveau bin si nécessaire
            if not placed:
                current_bin += 1
                bin_loads.append(item_size)
                solution[item_idx] = current_bin
        
        return solution
    
    def _evaluate_solution(self, solution: List[int]) -> int:
        """
        Évalue une solution : retourne le nombre de bins utilisés
        """
        if not solution:
            return float('inf')
        
        return max(solution) + 1 if solution else 0
    
    def _generate_neighborhood(self, solution: List[int]) -> List[Tuple[List[int], Tuple]]:
        """
        Génère le voisinage par déplacement d'items entre bins
        Retourne: liste de (solution_voisine, mouvement)
        """
        neighborhood = []
        current_bins = max(solution) + 1
        
        # Pour chaque item
        for item_idx in range(self.n_items):
            current_bin = solution[item_idx]
            item_size = self.items[item_idx]
            
            # Essayer de déplacer vers chaque autre bin existant
            for target_bin in range(current_bins):
                if target_bin != current_bin:
                    # Vérifier si le déplacement est faisable
                    if self._is_move_feasible(solution, item_idx, target_bin):
                        new_solution = copy.deepcopy(solution)
                        new_solution[item_idx] = target_bin
                        
                        # Nettoyer la solution (supprimer les bins vides)
                        cleaned_solution = self._clean_solution(new_solution)
                        
                        move = (item_idx, current_bin, target_bin)
                        neighborhood.append((cleaned_solution, move))
            
            # Essayer de créer un nouveau bin (si cela peut améliorer)
            new_bin = current_bins
            new_solution = copy.deepcopy(solution)
            new_solution[item_idx] = new_bin
            
            move = (item_idx, current_bin, new_bin)
            neighborhood.append((new_solution, move))
        
        return neighborhood
    
    def _is_move_feasible(self, solution: List[int], item_idx: int, target_bin: int) -> bool:
        """
        Vérifie si un déplacement d'item est faisable (capacité respectée)
        """
        item_size = self.items[item_idx]
        
        # Calculer la charge actuelle du bin cible
        target_load = sum(self.items[i] for i in range(self.n_items) 
                         if solution[i] == target_bin)
        
        return target_load + item_size <= self.bin_capacity
    
    def _clean_solution(self, solution: List[int]) -> List[int]:
        """
        Nettoie une solution en supprimant les bins vides et en renumérotant
        """
        # Identifier les bins utilisés
        used_bins = sorted(set(solution))
        
        # Créer un mapping des anciens numéros vers les nouveaux
        bin_mapping = {old_bin: new_bin for new_bin, old_bin in enumerate(used_bins)}
        
        # Appliquer le mapping
        cleaned_solution = [bin_mapping[solution[i]] for i in range(self.n_items)]
        
        return cleaned_solution
    
    def _is_tabu(self, move: Tuple) -> bool:
        """
        Vérifie si un mouvement est dans la liste tabou
        """
        item_idx, from_bin, to_bin = move
        
        # Un mouvement est tabou si le mouvement inverse est dans la liste tabou
        reverse_move = (item_idx, to_bin, from_bin)
        
        for tabu_move, _ in self.tabu_list:
            if tabu_move == reverse_move:
                return True
        
        return False
    
    def _add_to_tabu_list(self, move: Tuple):
        """
        Ajoute un mouvement à la liste tabou avec sa durée
        """
        # Ajouter le mouvement avec l'itération d'expiration
        expiration_iteration = self.iterations_performed + self.tabu_tenure
        self.tabu_list.append((move, expiration_iteration))
        
        # Nettoyer la liste tabou (supprimer les mouvements expirés)
        self.tabu_list = [(m, exp) for m, exp in self.tabu_list 
                         if exp > self.iterations_performed]
    
    def _convert_solution_format(self, solution: List[int]) -> List[List[int]]:
        """
        Convertit la solution du format interne vers le format de sortie
        """
        if not solution:
            return []
        
        n_bins = max(solution) + 1
        bins = [[] for _ in range(n_bins)]
        
        for item_idx, bin_num in enumerate(solution):
            bins[bin_num].append(self.items[item_idx])
        
        return bins

def print_solution(solution: List[List[int]], bin_capacity: int):
    """Affiche la solution de manière lisible"""
    print(f"\n=== SOLUTION FINALE ===")
    print(f"Nombre de bins utilisés: {len(solution)}")
    print(f"Capacité par bin: {bin_capacity}")
    
    total_items = 0
    total_utilization = 0
    
    for i, bin_items in enumerate(solution):
        total_size = sum(bin_items)
        utilization = (total_size / bin_capacity) * 100
        total_utilization += utilization
        print(f"Bin {i+1}: {bin_items} (total: {total_size}/{bin_capacity}, utilisation: {utilization:.1f}%)")
        total_items += len(bin_items)
    
    avg_utilization = total_utilization / len(solution) if solution else 0
    print(f"Total items placés: {total_items}")
    print(f"Utilisation moyenne: {avg_utilization:.1f}%")

def print_statistics(stats: Dict):
    """Affiche les statistiques de performance"""
    print(f"\n=== STATISTIQUES ===")
    print(f"Temps CPU: {stats['cpu_time']:.3f} secondes")
    print(f"Itérations effectuées: {stats['iterations']}")
    print(f"Bins dans la solution finale: {stats['final_bins']}")
    print(f"Tenure tabou: {stats['tabu_tenure']}")
    print(f"Mouvements évalués: {stats['moves_evaluated']}")
    print(f"Mouvements tabou rejetés: {stats['tabu_hits']}")
    print(f"Critère d'arrêt: {stats['max_no_improvement']} itérations sans amélioration")
    
    if len(stats['best_cost_history']) > 1:
        initial_cost = stats['best_cost_history'][0]
        final_cost = stats['best_cost_history'][-1]
        improvement = initial_cost - final_cost
        print(f"Amélioration totale: {improvement} bins")
        
        if initial_cost > 0:
            improvement_percent = (improvement / initial_cost) * 100
            print(f"Amélioration relative: {improvement_percent:.1f}%")

def plot_evolution(stats: Dict):
    """Affiche l'évolution des coûts (optionnel - nécessite matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        iterations = range(1, len(stats['cost_history']) + 1)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, stats['cost_history'], 'b-', label='Solution courante', alpha=0.7)
        plt.plot(iterations, stats['best_cost_history'], 'r-', label='Meilleure solution', linewidth=2)
        plt.xlabel('Itération')
        plt.ylabel('Nombre de bins')
        plt.title('Évolution des solutions - Recherche Tabou')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        improvements = []
        best_so_far = float('inf')
        for cost in stats['best_cost_history']:
            if cost < best_so_far:
                improvements.append(len(improvements))
                best_so_far = cost
        
        if improvements:
            plt.scatter(improvements, [stats['best_cost_history'][i] for i in improvements], 
                       color='red', s=50, zorder=5)
            plt.plot(iterations, stats['best_cost_history'], 'r-', alpha=0.5)
            plt.xlabel('Itération')
            plt.ylabel('Nombre de bins')
            plt.title('Points d\'amélioration')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib non disponible pour l'affichage graphique")

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple 1: Petit problème
    print("=== EXEMPLE 1 ===")
    items1 = [7, 5, 3, 3, 2, 2, 1]
    capacity1 = 10
    
    ts1 = BinPackingTabuSearch(
        items=items1,
        bin_capacity=capacity1,
        max_iterations=50,
        tabu_tenure=5,
        max_no_improvement=15
    )
    
    solution1, n_bins1, stats1 = ts1.solve()
    print_solution(solution1, capacity1)
    print_statistics(stats1)
    
    print("\n" + "="*60 + "\n")
    
    # Exemple 2: Problème plus complexe
    print("=== EXEMPLE 2 ===")
    items2 = [8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1]
    capacity2 = 12
    
    ts2 = BinPackingTabuSearch(
        items=items2,
        bin_capacity=capacity2,
        max_iterations=100,
        tabu_tenure=7,
        max_no_improvement=25
    )
    
    solution2, n_bins2, stats2 = ts2.solve()
    print_solution(solution2, capacity2)
    print_statistics(stats2)
    
    # Affichage optionnel de l'évolution
    plot_evolution(stats2)
    
    print("\n" + "="*60 + "\n")
    
    # Exemple 3: Test avec différents paramètres
    print("=== EXEMPLE 3 - PARAMÈTRES DIFFÉRENTS ===")
    items3 = [9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1]
    capacity3 = 15
    
    ts3 = BinPackingTabuSearch(
        items=items3,
        bin_capacity=capacity3,
        max_iterations=150,
        tabu_tenure=10,
        max_no_improvement=30
    )
    
    solution3, n_bins3, stats3 = ts3.solve()
    print_solution(solution3, capacity3)
    print_statistics(stats3)
