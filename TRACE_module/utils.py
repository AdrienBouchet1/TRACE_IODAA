"""Fonctions utiles
"""

from functools import wraps
import time


### _________________ IMPORTATION DES LIBRAIRIES _________________
from itertools import chain, combinations


###  _________________ FONCTIONS CONSECUTIVE POWERSET _________________

def are_consecutive(liste_num : list[int]) -> bool :
    """True si c'est une liste d'entier consécutifs

    Args:
        liste_num (list[int]): liste d'entiers

    Returns:
        bool: True -> Liste d'entier consécutifs ; False sinon
    """
    assert len(liste_num) >= 2; "Une liste vide ou de longueur 1 ne contient pas d'entier consécutifs"
    num = liste_num[0]
    ind = 1
    while ind < len(liste_num) and (num + 1 == liste_num[ind]): 
        num = liste_num[ind]
        ind += 1
    return (ind == len(liste_num))


def powerset(liste_num : list[int]) -> list:
    """Powerset d'un iterable

    Args:
        iterable : Iterable

    Returns:
        list: liste de tous les sous ensembles possibles
    """
    s = list(liste_num)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def subset_int_consecutif(liste_num : list[int]) -> list:
    """Fonction qui renvoie tous les subsets d'entiers consécutifs à partir d'une liste d'entier 

    Args:
        liste_num (list[int]): liste d'entier ordonnée

    Returns:
        list: liste de tous les subets d'entier consécutifs de la liste 
    """
    liste_consecutif = [] #output final avec tous les subsets d'entier consécutifs
    subsets = powerset(liste_num)[1:] # Premier élément toujours vide

    for list_num in subsets : 
        list_num = list(list_num)
        if len(list_num) == 1 :  # On ajoute toutes les listes de longueur 1
            liste_consecutif.append(list_num)
        else : # Si c'est un liste d'entier consécutifs on les ajoutent 
            if are_consecutive(list_num) : liste_consecutif.append(list_num)
    return liste_consecutif

def subset_int(liste_num : list[int]) -> list : 
    """Fonction qui renvoie le powerset des nombres

    Args:
        liste_num (list[int]): Liste des numéros à partir desquels il faut faire le powerset

    Returns:
        list: List de toutes les combinaisons de subsets
    """
    return powerset(liste_num)[1:]
    

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper