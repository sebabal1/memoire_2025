import time
import copy
import random
from typing import List, Tuple, Dict, Set
from instances import get_instance_par_categorie

class BinPackingTabuSearch:
    def __init__(self, items: List[int], capacite_max_bac: int,
                 max_iterations: int = 100, tps_tabou: int = 7,
                 nb_iterations_max: int = 20):
        # Vérification de la taille des objets
        for item in items:
            if item > capacite_max_bac:
                raise ValueError(f"Un item ({item}) dépasse la capacité maximale du bac ({capacite_max_bac}).") 
        
        self.items = items
        self.capacite_max_bac = capacite_max_bac
        self.n_items = len(items)
        self.max_iterations = max_iterations
        self.tps_tabou = tps_tabou # Durée pendant laquelle un mouvement reste tabou
        self.nb_iterations_max = nb_iterations_max # Nombre d'itérations sans amélioration avant d'arrêter

        # Structures de données pour la recherche tabou
        self.liste_tabou = []  # Liste des mouvements interdits
        self.meilleure_solution = None
        self.meilleure_cout = float('inf')
        self.solution_en_court = None
        self.cout_solution_current = float('inf') # Coût de la solution courante
        
        # Statistiques
        self.nb_iteration = 0 # Nombre d'itérations effectuées
        self.cost_history = []
        self.meilleure_cout_history = []
        self.nb_tabou_exclus = 0 # Nombre de mouvements tabou rejetés
        self.moves_evaluated = 0 # Nombre de mouvements évalués
        
    def solve(self) -> Tuple[List[List[int]], int, Dict]:
        """
        Résout le problème de bin packing avec la recherche tabou
        Retourne: (solution, nombre de bins, statistiques)
        """
        start_time = time.time()
        
        print(f"Initialisation de la recherche tabou...")
        print(f"Objets: {self.items}")
        print(f"Capacite des bins: {self.capacite_max_bac}")
        print(f"Iterations max: {self.max_iterations}")
        print(f"Tenure tabou: {self.tps_tabou}")
        
        # 1. Solution initiale
        self.solution_en_court = self._generate_initial_solution()
        self.cout_solution_current = self._evaluate_solution(self.solution_en_court)
        
        # Initialiser la meilleure solution
        self.meilleure_solution = copy.deepcopy(self.solution_en_court)
        self.meilleure_cout = self.cout_solution_current
        
        print(f"Solution initiale: {self.meilleure_cout} bins")
        
        no_improvement_count = 0
        
        # 2. Boucle principale de recherche tabou
        for iteration in range(self.max_iterations):
            self.nb_iteration = iteration + 1
            
            # 3. Générer le voisinage
            neighborhood = self._generate_neighborhood(self.solution_en_court)
            
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
                aspiration_criterion = neighbor_cost < self.meilleure_cout
                
                # Accepter le mouvement si non-tabou ou si critère d'aspiration satisfait
                if (not is_tabu or aspiration_criterion) and neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_move = move
                
                if is_tabu and not aspiration_criterion:
                    self.nb_tabou_exclus += 1
            
            # 5. Mise à jour de la solution courante
            if best_neighbor is not None:
                self.solution_en_court = best_neighbor
                self.cout_solution_current = best_neighbor_cost
                
                # Ajouter le mouvement à la liste tabou
                self._add_to_liste_tabou(best_move)
                
                # 6. Mise à jour de la meilleure solution
                if self.cout_solution_current < self.meilleure_cout:
                    self.meilleure_solution = copy.deepcopy(self.solution_en_court)
                    self.meilleure_cout = self.cout_solution_current
                    no_improvement_count = 0
                    print(f"Itération {iteration + 1}: Nouvelle meilleure solution = {self.meilleure_cout} bins")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Enregistrer l'historique
            self.cost_history.append(self.cout_solution_current)
            self.meilleure_cout_history.append(self.meilleure_cout)
            
            if iteration % 10 == 0:
                print(f"Itération {iteration + 1}: Actuel = {self.cout_solution_current}, Meilleur = {self.meilleure_cout}")
            
            # 7. Critère d'arrêt par stagnation
            if no_improvement_count >= self.nb_iterations_max:
                print(f"Arrêt par stagnation après {no_improvement_count} itérations sans amélioration")
                break
        
        end_time = time.time()
        
        # Conversion de la solution
        final_solution = self._convert_solution_format(self.meilleure_solution)
        
        stats = {
            'cpu_time': end_time - start_time,
            'iterations': self.nb_iteration,
            'final_bins': self.meilleure_cout,
            'cost_history': self.cost_history,
            'meilleure_cout_history': self.meilleure_cout_history,
            'tps_tabou': self.tps_tabou,
            'nb_tabou_exclus': self.nb_tabou_exclus,
            'moves_evaluated': self.moves_evaluated,
            'nb_iterations_max': self.nb_iterations_max
        }
        
        return final_solution, int(self.meilleure_cout), stats
    
    def _generate_initial_solution(self) -> List[int]:
        """
        Génère une solution initiale avec First Fit Decreasing
        Représentation: liste où solution[i] = numéro du bin pour l'item i
        """
        # Trier les items par taille décroissante avec leurs indices
        #sorted_items = sorted(enumerate(self.items), key=lambda x: x[1], reverse=True)
        
        solution = [0] * self.n_items
        bin_loads = []
        current_bin = 0
        
        for item_idx, item_size in enumerate(self.items):
            placed = False
            
            # Essayer de placer dans un bin existant
            for bin_num in range(current_bin + 1):
                if bin_num >= len(bin_loads):
                    bin_loads.append(0)
                
                if bin_loads[bin_num] + item_size <= self.capacite_max_bac:
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
        
        return target_load + item_size <= self.capacite_max_bac
    
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
        
        for tabu_move, _ in self.liste_tabou:
            if tabu_move == reverse_move:
                return True
        
        return False
    
    def _add_to_liste_tabou(self, move: Tuple):
        """
        Ajoute un mouvement à la liste tabou avec sa durée
        """
        # Ajouter le mouvement avec l'itération d'expiration
        expiration_iteration = self.nb_iteration + self.tps_tabou
        self.liste_tabou.append((move, expiration_iteration))
        
        # Nettoyer la liste tabou (supprimer les mouvements expirés)
        self.liste_tabou = [(m, exp) for m, exp in self.liste_tabou 
                         if exp > self.nb_iteration]
    
    def _convert_solution_format(self, solution: List[int]) -> List[List[int]]:
        """
        Convertit la solution du format interne vers le format de sortie
        """
        if not solution:
            return []
        
        n_bins = max(solution) + 1 # Nombre de bins utilisés
        # Créer une liste de bins
        # Chaque bin est une liste d'items
        bins = [[] for _ in range(n_bins)]
        
        # Remplir les bins avec les items
        # Chaque item est placé dans le bin correspondant
        # selon la solution
        for item_idx, bin_num in enumerate(solution):
            bins[bin_num].append(self.items[item_idx])
        
        return bins

def print_solution(solution: List[List[int]], capacite_max_bac: int, items: List[int]):
    """Affiche la solution de manière lisible"""
    print(f"\n=== SOLUTION FINALE ===")
    print(f"Nombre d'objets utilises: {len(solution)}")
    print(f"Capacite par bac: {capacite_max_bac}")
    
    total_items = 0
    total_utilization = 0
    
    for i, bin_items in enumerate(solution):
        total_size = sum(bin_items)
        utilization = (total_size / capacite_max_bac) * 100
        total_utilization += utilization
        print(f"Bin {i+1}: {bin_items} (total: {total_size}/{capacite_max_bac}, utilisation: {utilization:.1f}%)")
        total_items += len(bin_items)
    
    quality = calculate_solution_quality(solution, capacite_max_bac)
    print(f"Total items placés: {total_items}/{len(items)}")
    print(f"Utilisation moyenne: {quality:.2f}%")

def calculate_solution_quality(solution: List[List[int]], capacite_max_bac: int) -> float:
    """Calcule le taux d'utilisation moyen des bacs"""
    if not solution:
        return 0.0
    total_utilization = 0
    for bin_items in solution:
        bin_weight = sum(bin_items)
        total_utilization += bin_weight / capacite_max_bac
    return total_utilization / len(solution) * 100

def print_statistics(stats: Dict):
    """Affiche les statistiques de performance"""
    print(f"\n=== STATISTIQUES ===")
    print(f"Temps CPU: {stats['cpu_time']:.3f} secondes")
    print(f"Itérations effectuées: {stats['iterations']}")
    print(f"Bins dans la solution finale: {stats['final_bins']}")
    print(f"Temps Tabou iterations: {stats['tps_tabou']}")
    print(f"Mouvements évalués: {stats['moves_evaluated']}")
    print(f"Mouvements tabou rejetés: {stats['nb_tabou_exclus']}")
    print(f"Critère d'arrêt: {stats['nb_iterations_max']} itérations sans amélioration")
    
    if len(stats['meilleure_cout_history']) > 1:
        initial_cost = stats['meilleure_cout_history'][0]
        final_cost = stats['meilleure_cout_history'][-1]
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
        plt.plot(iterations, stats['meilleure_cout_history'], 'r-', label='Meilleure solution', linewidth=2)
        plt.xlabel('Itération')
        plt.ylabel('Nombre de bins')
        plt.title('Évolution des solutions - Recherche Tabou')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        improvements = []
        best_so_far = float('inf')
        for cost in stats['meilleure_cout_history']:
            if cost < best_so_far:
                improvements.append(len(improvements))
                best_so_far = cost
        
        if improvements:
            plt.scatter(improvements, [stats['meilleure_cout_history'][i] for i in improvements], 
                       color='red', s=50, zorder=5)
            plt.plot(iterations, stats['meilleure_cout_history'], 'r-', alpha=0.5)
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
   
    instances = get_instance_par_categorie('basic')

    for i, instance in enumerate(instances, 1):
        tabou_search = BinPackingTabuSearch(
            items= instance['items'],
            capacite_max_bac=instance['capacite_max_bac'],
            max_iterations=50,
            tps_tabou=5,
            nb_iterations_max=15)
       
        solution, n_bins, stats = tabou_search.solve()
        print_solution(solution, instance['capacite_max_bac'], instance['items'])
        print_statistics(stats)
        
        print("\n" + "="*60 + "\n")
    
    # ts1 = BinPackingTabuSearch(
    #     items=items1,
    #     capacite_max_bac=capacity1,
    #     max_iterations=50,
    #     tps_tabou=5,
    #     nb_iterations_max=15
    # )
    
    # solution1, n_bins1, stats1 = ts1.solve()
    # print_solution(solution1, capacity1)
    # print_statistics(stats1)
    
    # print("\n" + "="*60 + "\n")
    
    # Exemple 2: Problème plus complexe
    # print("=== EXEMPLE 2 ===")
    # items2 = [8, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1]
    # capacity2 = 12
    
    # ts2 = BinPackingTabuSearch(
    #     items=items2,
    #     capacite_max_bac=capacity2,
    #     max_iterations=100,
    #     tps_tabou=7,
    #     nb_iterations_max=25
    # )
    
    # solution2, n_bins2, stats2 = ts2.solve()
    # print_solution(solution2, capacity2)
    # print_statistics(stats2)
    
    # Affichage optionnel de l'évolution
    # plot_evolution(stats2)
    
    # print("\n" + "="*60 + "\n")
    
    # Exemple 3: Test avec différents paramètres
    # print("=== EXEMPLE 3 - PARAMÈTRES DIFFÉRENTS ===")
    # items3 = [9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1]
    # capacity3 = 15
    
    # ts3 = BinPackingTabuSearch(
    #     items=items3,
    #     capacite_max_bac=capacity3,
    #     max_iterations=150,
    #     tps_tabou=10,
    #     nb_iterations_max=30
    # )
    
    # solution3, n_bins3, stats3 = ts3.solve()
    # print_solution(solution3, capacity3)
    # print_statistics(stats3)
