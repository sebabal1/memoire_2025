INSTANCES_BASIC = [
   {
        'name': 'Petit - 6 items',
        'items': [9, 8, 7, 6, 5, 4],
        'capacite_max_bac': 10
    },
    {
        'name': 'Petit - items similaires',
        'items': [6, 6, 5, 5, 5, 4, 4, 3],
        'capacite_max_bac': 10,
    },
    {
        'name': 'Distribution variée',
        'items': [8, 7, 9, 8, 7, 6, 5, 4, 3, 2],
        'capacite_max_bac': 10,
    },
    {
        'name': 'Nombreux petits items',
        'items': [8, 7, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1],
        'capacite_max_bac': 10,
    },
    {
        'name': 'Items identiques',
        'items': [4] * 10,
        'capacite_max_bac': 10,
    }
 ]

INSTANCES_HARD = [
   {
        'name': 'Grandes tailles',
        'items': [18, 17, 16, 15, 14, 8, 7, 6, 5] * 2,
        'capacite_max_bac': 25,
    },
    {
        'name': 'Plusieurs items identiques',
        'items': [5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2],
        'capacite_max_bac': 10,
    },
    {
        'name': '2 chiffres identiques plusieurs fois',
        'items': [3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2],
        'capacite_max_bac': 8,
    },
    {
        'name': 'Augmentation du nombre d\'items',
        'items': [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2] * 10,
        'capacite_max_bac': 25,
    },
    {
        'name': 'Augmentation du nombre d\'items',
        'items': [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1] * 10,
        'capacite_max_bac': 10,
    },
    {
        'name': 'Augmentation du nombre d\'items',
        'items': [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1] * 20,
        'capacite_max_bac': 10,
    }
]  
ALL_INSTANCES = {
    'basic': INSTANCES_BASIC,
    'hard': INSTANCES_HARD
}

# Permet de retourner une instance suivant sa catégorie
def get_instance_par_categorie(categorie='basic'):
    if categorie == 'all':
        instances = []
        for categorie_instance in ALL_INSTANCES.values():
            instances.extend(categorie_instance)
        return instances
    return ALL_INSTANCES.get(categorie, [])
