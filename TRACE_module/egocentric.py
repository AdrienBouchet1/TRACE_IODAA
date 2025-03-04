import pandas as pd
from TRACE_module.motif import *

def stack_to_df_motif(stack : np.ndarray, list_timestep: list[datetime], list_id: list[str]) -> pd.DataFrame: 
    """Transform the stack of adjacency matrixes to a dataframe containing the motif at each timestep

    Args:
        stack (np.ndarray): Stack of adjacency matrix
        list_timestep (list[datetime]): List of all the timesteps
        list_id (list[str]): List of all the ids of cows

    Returns:
        pd.DataFrame: Dataframe containing 2 cols : glob_sensor_DateTime -> timesteps | motif : The correponsing motif
    """
    list_motif = []
    for matrix in stack :
        list_motif.append(interaction_matrix_to_motif(matrix,list_id))
    return pd.DataFrame({"glob_sensor_DateTime" : list_timestep, "motif" : list_motif})


def count_fuzzy_sequences_of_ones(arr : np.array, max_gap=1):
    """
    Compte le nombre de séquences de 1 en considérant qu'une interruption 
    de `max_gap` zéros ne casse pas la continuité.

    Args:
        arr (np.array): Tableau 1D contenant des 0 et des 1.
        max_gap (int): Nombre maximum de 0 autorisés à l'intérieur d'une séquence de 1.

    Returns:
        int: Nombre de séquences considérées comme continues.
    """
    # Récupérer les indices où il y a des 1
    ones_indices = np.where(arr == 1)[0]

    if len(ones_indices) == 0:
        return 0  # Aucun 1 dans le tableau

    # Compter les groupes en vérifiant les écarts entre indices
    count = 1  # Au moins une séquence détectée
    for i in range(1, len(ones_indices)):
        if (ones_indices[i] - ones_indices[i - 1]) > (max_gap + 1):
            count += 1  # Nouvelle séquence détectée
    
    return count


def mat_stack_to_list_connected_components(stack : np.ndarray, list_id : list[str]) : 
    """Fonction qui à partir du stack de matrice d'adjacence permet de sortir la liste des composantes connexes à chaque time step

    Args:
        stack (np.nd_array): Stack des matrices d'adjacence
        list_id (list[str]): list des id des vaches

    Returns:
        list[list[Motif]]: Liste des composantes connexes à chaque timestep
    """
    list_connected = []
    for matrix in stack : 
        list_comp = interaction_matrix_to_motif(matrix,list_id).connected_components()
        list_connected.append(list_comp)
    return list_connected



def mask_connected_comp(motif : Motif, list_connected_per_ts : list[Motif]):
    """Fonction qui permet de sortir le mask d'apparition d'un motif"""
    mask = []
    for list_comp in list_connected_per_ts : 
        for motif_comp in list_comp : 
            if motif.is_subgraph(motif_comp) : 
                submotif = True
            else :
                submotif = False
        if motif in list_comp or submotif :
            mask.append(1)
        else :
            mask.append(0)
    return np.array(mask)



def masks_all_components(list_connected_components : list[Motif], list_connected_per_ts :list[Motif]) -> dict[Motif, np.array]:
    """Extract all the mask for a given list of components

    Args:
        list_connected_components (list[Motif]): List of all the connected components
        list_connected_per_ts (list[Motif]): All the connected components that appears for each timestep. Be careful it needs to be ordered

    Returns:
        dict[Motif, np.array]: Dictionnary containing the motifs as keys and their corresponding masks
    """
    masks = dict()
    for comp in tqdm(list_connected_components) : 
        masks[comp] = mask_connected_comp(comp, list_connected_per_ts)
    return masks


def fuzzy_count_all_masks(masks_all_comp : dict[Motif, np.array], max_gap : int) -> dict[Motif, int]:
    """Counts all the interaction sequences of the different motifs

    Args:
        masks_all_comp (dict[Motif, np.array]): Mask of all the motifs
        max_gap (int): Maximum error gap for 2 sequences to be counted as one 

    Returns:
        dict[Motif, int]: Counts of the interaction sequences of the motifs
    """
    fuzzy_counts = dict()
    for comp,mask in masks_all_comp.items(): 
        fuzzy_counts[comp] = count_fuzzy_sequences_of_ones(mask, max_gap)
    return fuzzy_counts