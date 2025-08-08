import time
import copy
from typing import List, Tuple, Optional
import math

from instances import get_instance_par_categorie

class BinPackingBnB:
    def __init__(self, items: List[int], capacite_max_bac: int):
        self.items = items
        self.capacite_max_bac = capacite_max_bac
        self.n_items = len(items)
        self.best_solution = None
        self.best_num_bins = float('inf')
        self.nodes_explored = 0
        
    def lower_bound(self, articles_restants: int):
        """Calcule une borne inférieure simple : somme des items / capacité"""
        if not articles_restants: return 0
    
        poids_total_des_items = sum(articles_restants)
        
        return max(1, (poids_total_des_items + self.capacite_max_bac - 1) // self.capacite_max_bac)
    
    def can_fit(self, item, bin_contents):
        """Vérifie si un item peut être placé dans un bin"""
        return sum(bin_contents) + item <= self.capacite_max_bac
    
    def branch_and_bound(self, item_index=0, current_bins=None):
        """Algorithme Branch and Bound basique"""
        if current_bins is None:
            current_bins = []
        
        self.nodes_explored += 1
        
        # Cas de base : tous les items ont été placés
        if item_index == len(self.items):
            if len(current_bins) < self.best_num_bins:
                self.best_num_bins = len(current_bins)
                self.best_solution = copy.deepcopy(current_bins)
            return
        
        
        # Élagage : si le nombre actuel de bins + borne inférieure >= meilleure solution
        articles_restants = self.items[item_index:]
        lower_bound = self.lower_bound(articles_restants)
        
        if len(current_bins) + lower_bound >= self.best_num_bins:
            return
        
        current_item = self.items[item_index]
        
        # Branche 1 : Essayer de placer l'item dans chaque bin existant
        for i, bin_contents in enumerate(current_bins):
            if self.can_fit(current_item, bin_contents):
                # Placer l'item dans le bin i
                current_bins[i].append(current_item)
                self.branch_and_bound(item_index + 1, current_bins)
                # Backtrack
                current_bins[i].pop()
        
        # Branche 2 : Créer un nouveau bin pour cet item
        if len(current_bins) + 1 < self.best_num_bins:  # Élagage
            new_bin = [current_item]
            current_bins.append(new_bin)
            self.branch_and_bound(item_index + 1, current_bins)
            # Backtrack
            current_bins.pop()
    
    def solve(self):
        """Résout le problème et retourne les statistiques"""
        start_time = time.time()
        
        self.branch_and_bound()
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        return {
            'solution': self.best_solution,
            'num_bins': self.best_num_bins,
            'cpu_time': cpu_time,
            'nodes_explored': self.nodes_explored
        }

def calculate_solution_quality(num_bins, capacite_max_bac, items):
    """Calcule la qualité de la solution (taux d'utilisation moyen)"""
    if not num_bins: return 0
    
    # Permet de parcourir tous les bacs et de faire le poids total par bac
    total_capacity = num_bins * capacite_max_bac 
    utilization_rate = sum(items) / total_capacity
    
    return utilization_rate

def print_solution(items, solution, stats, capacite_max_bac, nom_instance):
    """Affiche la solution et les statistiques"""
    print(nom_instance)
    print("=" * 50)
    
    print(f"Items a emballer : {items}")
    print(f"Capacite des bins : {capacite_max_bac}")
    print(f"Nombre d'items : {len(items)}")
    print()
    
    if solution:
        print("Solution trouvee :")
        for i, bin_contents in enumerate(solution):
            utilization = sum(bin_contents) / capacite_max_bac * 100
            print(f"  Bin {i+1}: {bin_contents} (utilisation: {utilization:.1f}%)")
        print()
        
        quality = calculate_solution_quality(stats['num_bins'], capacite_max_bac, items)
        print(f"Nombre de bins utilises : {stats['num_bins']}")
        print(f"Taux d'utilisation moyen : {quality:.2%}")
    else:
        print("Aucune solution trouvee")
    
    print()
    print("PERFORMANCES :")
    print(f"  Temps CPU : {stats['cpu_time']:.7f} secondes")
    print(f"  Nombre d'items : {len(items):,}".replace(',', '.'))
    print(f"  Noeuds explores : {stats['nodes_explored']:,}".replace(',', '.'))
    print(f"  Noeuds par seconde : {stats['nodes_explored']/max(stats['cpu_time'], 0.0001):,.0f}".replace(',', '.'))

# Exemple d'utilisation
if __name__ == "__main__":
    print("=" * 50)
    print("RESULTATS DU BRANCH AND BOUND - BIN PACKING")
    print("=" * 50)

    instances = get_instance_par_categorie('basic')
    print("BRANCH AND BOUND BASIC:")
    for i, instance in enumerate(instances, 1):
        try:
            solver = BinPackingBnB(instance['items'], instance['capacite_max_bac'])
            stats = solver.solve()
            print_solution(instance['items'], stats['solution'], stats, instance['capacite_max_bac'], instance['name'])
            print("\n" + "="*70 + "\n")
        except Exception as e:
            print(f" Erreur: {e}")

    print("BRANCH AND BOUND HARD:")
    instances = get_instance_par_categorie('hard')

    for i, instance in enumerate(instances, 1):
        try:
            solver = BinPackingBnB(instance['items'], instance['capacite_max_bac'])
            stats = solver.solve()
            print_solution(instance['items'], stats['solution'], stats, instance['capacite_max_bac'], instance['name'])
            print("\n" + "="*70 + "\n")
        except Exception as e:
            print(f" Erreur: {e}")