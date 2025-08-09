import time
import copy
from typing import List, Tuple, Optional
import math
from instances import get_instance_par_categorie

class BinPackingBnC:
    def __init__(self, items: List[int], capacite_max_bac: int):
        self.items = items  # Pas de tri, comme demandé
        self.capacite_max_bac = capacite_max_bac
        self.n_items = len(items)
        self.best_solution = None
        self.best_num_bins = float('inf')
        self.nodes_explored = 0
        self.cuts_added = 0  # Compteur de coupes ajoutées
        self.cuts_applied = 0  # Compteur de coupes appliquées
        
    def lower_bound(self, articles_restants: List[int]):
        """Calcule une borne inférieure améliorée avec les coupes"""
        if not articles_restants: 
            return 0
    
        poids_total_des_items = sum(articles_restants)
        # Borne inférieure basique
        lb_basic = max(1, (poids_total_des_items + self.capacite_max_bac - 1) // self.capacite_max_bac)
    
        # Amélioration avec coupe : items trop gros
        #lb_big_items = sum(1 for item in articles_restants if item > self.capacite_max_bac // 2)
        return lb_basic
        #return max(lb_basic, lb_big_items)
    
    def generate_cuts(self, articles_restants: List[int], current_bins: List[List[int]]):
        """Génère des coupes (inégalités valides) pour améliorer la borne inférieure"""
        cuts_improvement = 0
        
        if not articles_restants:
            return cuts_improvement
        
        # Coupe 1: Contrainte de capacité renforcée
        # Si on a des items qui ne peuvent pas être ensemble
        incompatible_pairs = 0
        for i, item1 in enumerate(articles_restants):
            for item2 in articles_restants[i+1:]:
                if item1 + item2 > self.capacite_max_bac:
                    incompatible_pairs += 1
        
        if incompatible_pairs > 0:
            self.cuts_added += 1
            cuts_improvement += 1
        
        # Coupe 2: Contrainte de cardinalité
        # Maximum d'items par bin basé sur la taille minimale
        if articles_restants:
            min_item = min(articles_restants)
            max_items_per_bin = self.capacite_max_bac // min_item
            if len(articles_restants) > max_items_per_bin:
                min_bins_needed = math.ceil(len(articles_restants) / max_items_per_bin)
                if min_bins_needed > 1:
                    self.cuts_added += 1
                    cuts_improvement += min_bins_needed - 1
        
        # Coupe 3: Contrainte de fractionnement
        # Certains groupes d'items nécessitent plusieurs bins
        large_items = [item for item in articles_restants if item > self.capacite_max_bac * 0.6]
        if len(large_items) > 1:
            self.cuts_added += 1
            cuts_improvement += len(large_items) - 1
        
        return cuts_improvement
    
    def can_fit(self, item, bin_contents):
        """Vérifie si un item peut être placé dans un bin"""
        return sum(bin_contents) + item <= self.capacite_max_bac
    
    def apply_cutting_planes(self, articles_restants: List[int], current_bins: List[List[int]]):
        """Applique les plans de coupe pour améliorer la borne inférieure"""
        base_lb = self.lower_bound(articles_restants)
        cuts_improvement = self.generate_cuts(articles_restants, current_bins)
        
        if cuts_improvement > 0:
            self.cuts_applied += 1
        
        return base_lb + cuts_improvement
    
    def branch_and_cut(self, item_index=0, current_bins=None):
        """Algorithme Branch and Cut basique"""
        if current_bins is None:
            current_bins = []
        
        self.nodes_explored += 1
        
        # Cas de base : tous les items ont été placés
        if item_index == len(self.items):
            if len(current_bins) < self.best_num_bins:
                self.best_num_bins = len(current_bins)
                self.best_solution = copy.deepcopy(current_bins)
            return
        
        # Élagage avec plans de coupe (différence principale avec B&B)
        articles_restants = self.items[item_index:]
        
        # Application des coupes pour améliorer la borne inférieure
        improved_lower_bound = self.apply_cutting_planes(articles_restants, current_bins)
        
        # Élagage amélioré
        if len(current_bins) + improved_lower_bound >= self.best_num_bins:
            return
        
        current_item = self.items[item_index]
        
        # Branche 1 : Essayer de placer l'item dans chaque bin existant
        for i, bin_contents in enumerate(current_bins):
            if self.can_fit(current_item, bin_contents):
                # Placer l'item dans le bin i
                current_bins[i].append(current_item)
                self.branch_and_cut(item_index + 1, current_bins)
                # Backtrack
                current_bins[i].pop()
        
        # Branche 2 : Créer un nouveau bin pour cet item
        if len(current_bins) + 1 < self.best_num_bins:  # Élagage
            new_bin = [current_item]
            current_bins.append(new_bin)
            self.branch_and_cut(item_index + 1, current_bins)
            # Backtrack
            current_bins.pop()
        
    
    def solve(self):
        """Résout le problème et retourne les statistiques"""
        start_time = time.time()
        
        self.branch_and_cut()
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        return {
            'solution': self.best_solution,
            'num_bins': self.best_num_bins,
            'cpu_time': cpu_time,
            'nodes_explored': self.nodes_explored,
            'cuts_added': self.cuts_added,
            'cuts_applied': self.cuts_applied
        }

def calculate_solution_quality(num_bins, capacite_max_bac, items):
    """Calcule la qualité de la solution (taux d'utilisation moyen)"""
    if not num_bins: return 0
    
    total_capacity = num_bins * capacite_max_bac 
    utilization_rate = sum(items) / total_capacity
    
    return utilization_rate

def print_solution_bnc(items, solution, stats, capacite_max_bac, nom_instance):
    """Affiche la solution et les statistiques pour Branch and Cut"""
    print(nom_instance)
    print("=" * 50)
    
    print(f"Objets a emballer : {items}")
    print(f"Capacite des bins : {capacite_max_bac}")
    print(f"Nombre d'objets : {len(items)}")
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
    print("PERFORMANCES BRANCH AND CUT :")
    print(f"  Temps CPU : {stats['cpu_time']:.7f} secondes")
    print(f"  Nombre d'objets : {len(items):,}".replace(',', '.'))
    print(f"  Noeuds explores : {stats['nodes_explored']:,}".replace(',', '.'))
    print(f"  Noeuds par seconde : {stats['nodes_explored']/max(stats['cpu_time'], 0.0001):,.0f}".replace(',', '.'))
    print(f"  Coupes generees : {stats['cuts_added']}")
    print(f"  Coupes appliquees : {stats['cuts_applied']}")

# Exemple d'utilisation
if __name__ == "__main__":
    print("=" * 50)
    print("RESULTATS DU BRANCH AND CUT - BIN PACKING")
    print("=" * 50)

    instance = get_instance_par_categorie('basic')
    
    # Test Branch and Cut seul
    print("BRANCH AND CUT BASIC:")
    for i, instance in enumerate(instance, 1):
        solver_bnc = BinPackingBnC(instance['items'], instance['capacite_max_bac'])
        stats_bnc = solver_bnc.solve()
        print_solution_bnc(instance['items'], stats_bnc['solution'], stats_bnc, 
                        instance['capacite_max_bac'], instance['name'])
        print("\n" + "="*80 + "\n")

    print("BRANCH AND CUT HARD:")
    instance = get_instance_par_categorie('hard')
    for i, instance in enumerate(instance, 1):
        solver_bnc = BinPackingBnC(instance['items'], instance['capacite_max_bac'])
        stats_bnc = solver_bnc.solve()
        print_solution_bnc(instance['items'], stats_bnc['solution'], stats_bnc, 
                        instance['capacite_max_bac'], instance['name'])
        print("\n" + "="*80 + "\n")
    
    
    
    
    
 
