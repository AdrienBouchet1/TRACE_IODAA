# Module d'implémentation du papier https://doi.org/10.48550/arXiv.1612.09259

from datetime import datetime,timedelta
from tqdm import tqdm
from TRACE_module.utils import *
import numpy as np
from typing import List
from collections import Counter
#############################################################################
                        # Types definition
#############################################################################

Individu = str  # Accelero_id de la vache
TimeStep = datetime      # Timestep en seconde ?
Sequence = List # Séquence d'intéraction
ListeMotif = List # Liste de motifs

#############################################################################
                        # Classes definition
#############################################################################

class Arc :
    def __init__(self, ind1 : Individu, ind2 : Individu, oriented : bool = True):
        """Definition of an Arc/ Edge in the sense of a Graph

        Args:
            ind1 (Individu): First individual
            ind2 (Individu): Second individual
            oriented (bool, optional): True : This is an Arc, False : This is an edge. Defaults to True.
        """

        self._ind1 = ind1 
        self._ind2 = ind2 
        self._oriented = oriented
    
    def __repr__(self):
        if self._oriented :
            sep = "->"
        else :
            sep = "--"
        return(f"{self._ind1} {sep} {self._ind2}")
    
    def __eq__(self, obj) : 
        if self._oriented :
            result = ((self._ind1 == obj._ind1) & (self._ind2 == obj._ind2))
        else :
            result = ({self._ind1, self._ind2} == {obj._ind1, obj._ind2}) 
        return result

    def __hash__(self):
        if self._oriented :
            hashed = hash((self.ind1, self.ind2))
        else :
            hashed = hash(frozenset((self.ind1, self.ind2)))   
        return hashed

    @property
    def inds(self):
        return (self._ind1, self._ind2)

    @property
    def ind1(self): 
        return self._ind1

    @property
    def ind2(self): 
        return self._ind2
    
    @property
    def oriented(self):
        return self._oriented

    @oriented.setter
    def oriented(self, val):
        if not isinstance(val, bool) : 
            raise ValueError("La valeur d'orientation doit être un booléen : True -> Orienté, False -> Non-orienté")
        self._oriented = val
    

class Interaction :
    def __init__(self, ind1 : Individu, ind2 : Individu, timestep : datetime) : 
        """Interaction entre l'individu 1 et 2 à un timestep donné
        L'individu 1 capte l'individu 2 à un instant donné

        Args:
            ind1 (Individu): Identifiant de l'individu 1
            ind2 (Individu): Identifiant de l'individu 2
            timestep (datetime): Date de l'interaction
        """
        self._ind1 = ind1
        self._ind2 = ind2
        self._ts = timestep
    
    def __repr__(self): 
        return f" {self.ind1} -> {self.ind2} ({self.ts})"

    def __lt__(self,obj):
        return (self._ts < obj._ts)
    
    def __gt__(self,obj):
        return (self._ts > obj._ts)
    
    def __eq__(self,obj):
        return ((self._ind1 == obj._ind1) & (self._ind2 == obj._ind2) & (self._ts == obj._ts))
    
    @property
    def inds(self):
        return (self._ind1, self._ind2)
    
    @property
    def ts(self): 
        return self._ts

    @property
    def ind1(self): 
        return self._ind1

    @property
    def ind2(self): 
        return self._ind2


    
class Motif:
    def __init__(self,*args : Arc, oriented : bool = True):
        """Motif au sens du papier, c'est un graph orienté ou non

        Args:
            args (Arc) : some Arc (cf. Above) that need to have the same orientation for all of them
            oriented (bool, optional): True -> Graph i oriented / False : the graph is not oriented. Defaults to True.

        Example : 
        
        M_oriented = Motif(
            Arc("a","b"),
            Arc("a","c"),
            Arc("c","a"))

        M_not_oriented = Motif(
            Arc("a","b"),
            Arc("a","c"),
            Arc("c","a"),
            oriented= False)

        """
        self._oriented = oriented
        if self._oriented :
            self._list_arc = tuple(args)
        else : 
            args_not_oriented = []
            for arc in args : 
                if arc.oriented : 
                    arc.oriented = False
                args_not_oriented.append(arc)
            self._list_arc = tuple(sorted(args_not_oriented, key= lambda arc : sorted((arc.ind1, arc.ind2)))) 

    def __repr__(self):
        arcs = ", ".join(map(str, self._list_arc))
        return f"Motif( {arcs}, oriented : {self._oriented})"
    
    def __getitem__(self,key):
        return self._list_arc[key]
    
    def __eq__(self,motif):
        return (motif.oriented == self.oriented) and (motif._list_arc == self._list_arc)
    
    def __hash__(self):
        return hash((self._oriented, self._list_arc))

    def __len__(self): 
        return len(self._list_arc)
    
    @property
    def oriented(self): 
        return self._oriented
    
    def __add__(self, arc : Arc):
        """Ajout d'un arc en fin de séquence de motif

        Args:
            arc (Arc): arc à ajouter

        Returns:
            Motif : Motif avec l'arc ajouté au début 
        """
        if not self._oriented : 
            arc._oriented = False
        return Motif(*(self._list_arc + (arc,)), oriented= self._oriented)
    
    def add_suffix(self, arc : Arc):
        """Ajout d'un arc en début de séquence de motif

        Args:
            arc (Arc): arc à ajouter

        Returns:
            Motif : Motif avec l'arc ajouté au début 
        """
        if not self._oriented : 
            arc._oriented = False
        return Motif(*((arc,) + self._list_arc), oriented= self._oriented)

    
    def graph(self, r_max : int):
        """
        Fonction qui génère le code Latex d'un motif donné. à utiliser avec le template suivant : 
        
        \documentclass{standalone}
        \usepackage{tikz}
        \begin{document}
        \begin{tikzpicture}
            .... 
        \end{tikzpicture}
        \end{document}

        
        Args : r_max (int) : rayon max du cercle de coordonnées
        """

        # Retrieving all the different nodes

        r_max = r_max/2

        set_nodes = set()
        for arc in self._list_arc : 
            set_nodes.add(arc.ind1)
            set_nodes.add(arc.ind2)

        set_nodes = list(set_nodes)
        nb_nodes = len(set_nodes)

        # Coords for the different nodes
        t = np.linspace(0, 2*np.pi, nb_nodes, endpoint=False)
        x = r_max * np.cos(t) + r_max
        y = r_max * np.sin(t) + r_max
        list_command = []
        for node, coords in zip(set_nodes, zip(x,y)):
            coords = (float(round(coords[0],3)), float(round(coords[1],3)))
            command = f"\\node[draw, circle] ({node}) at {coords} {{{node}}};"
            list_command.append(command)
        # LaTex Command for nodes creation
        nodes_command = "\n".join(list_command)

        ## Implementation des différents arcs
        list_arc = []
        for arc in self._list_arc : 
            command = f"\\draw[-, thick] ({arc.ind1}) to ({arc.ind2});"
            list_arc.append(command)
        arc_command = "\n".join(list_arc) 

        print("\n".join((nodes_command,arc_command)))

    def gen_submotif(self):
        """Fonction du génére l'ensemble des sous motifs du motif

        Returns:
            list[Motif] : liste des sous motifs (ne contient pas le motif nul, et contient le motif lui même)
        """
        list_sub_motif = [] #Liste qui va contenir les sous motifs 
        if self._oriented : 
            sub_seq_keys = subset_int_consecutif([i for i in range(len(self._list_arc))]) # On récupère les clées de tous les sous motifs
        else :
            sub_seq_keys = subset_int([i for i in range(len(self._list_arc))])
        for list_keys in sub_seq_keys : 
            list_sub_motif.append(Motif(*[self._list_arc[key] for key in list_keys], oriented= self._oriented)) # Instanciation des motifs à partir des arc
        return list_sub_motif
        
        

def get_list_interactions(
        stack : np.ndarray, 
        list_id : list[str],
        list_time_steps : list[datetime]
        ) -> list[Interaction]: 
    """Generate the interaction list from the stack matrix

    Args:
        stack (np.ndarray): 3D adjacency matrix (cow_id x cow_id x time_step)
        list_id (list[str]): List of the Ids of the cows
        list_time_steps (list[datetime]): List of the timesteps

    Returns:
        list[Interaction]: List of the Interaction

        Example :     

        date_init = datetime.fromisoformat("2024-12-11T12:00:00")

        sequence = [
        Interaction("a","b", date_init + timedelta(seconds=25)),
        Interaction("a","c", date_init + timedelta(seconds=17)),
        Interaction("a","c", date_init + timedelta(seconds=28)),
        Interaction("a","c", date_init + timedelta(seconds=30)),
        Interaction("a","c", date_init + timedelta(seconds=35)),
        Interaction("c","a", date_init + timedelta(seconds=15)),
        Interaction("c","a", date_init + timedelta(seconds=32)),
        ]
    """
    list_sequence=[]
        
    for i in tqdm(list_id): 

        for j in list_id[list_id.index(i):]: 
            index_i,index_j=list_id.index(i),list_id.index(j)

            for t in range(len(list_time_steps)) : 
                if stack[t,index_i,index_j]==1 : 
                    list_sequence.append(Interaction(i,j,list_time_steps[t]))
        
    return list_sequence
   

#############################################################################
     # Algorithm 1 of the paper (cf. doi at the beginning of the code)
#############################################################################

def interaction_in_motif(interaction : Interaction, motif : Motif):
    """Returns True if an arc (from the interaction) is in a motif

    Args:
        interaction (Interaction): Interaction
        motif (Motif): Motif instance

    Returns:
        bool: True if the Interaction is in the Motif. False if not.
    """
    return Arc(*interaction.inds, oriented=motif.oriented) in motif._list_arc

def count_instance_motif(sequence_raw: Sequence, motif: Motif, delta: int, oriented_graph : bool = False) -> tuple[int, dict[Motif,int], list[Interaction]]:
    """
    Fonction qui permet de calculer le nombre d'occurences du motif `motif` dans la séquence d'intéraction `sequence` dans une fenetre de temps `delta`

    Parameters:
    ________________
    - sequence (Sequence) : Séquence d'intéraction ordonnée par pas de temps
    - motif (Motif) : Motif d'intérêt à détecter
    - delta (int) : Durée (en s) de la fenêtre de temps dans laquelle on veut considérer le motif 

    Returns:
    ________________
    - int : Nombre d'occurence du motif `motif` dans la séquence d'intéraction

    """
    # On tri la séquence d'intéraction 
    sequence_raw.sort()
    delta = np.timedelta64(delta,"s")

    # Filtre de la séquence d'intéraction 
    sequence = [inter for inter in sequence_raw if interaction_in_motif(inter,motif)]
    print(f"Sequence filtered (len : {len(sequence)})")

    # On récupère la longueur du motif :
    l_motif = len(motif)
    # Génération de tous les sous motifs 
    submotifs = motif.gen_submotif()
    l_submotifs = len(submotifs) #Nombre de sous-motifs 
    print(f"Submotifs generated (# : {l_submotifs})")

    # Dictionnaire de comptage qui a pour clé un motif (cf figure 2 du papier)
    counts = dict(zip(submotifs, [0 for i in range(l_submotifs)]))
    dict_counts = dict(zip(submotifs, [[0] for i in range(l_submotifs)]))
    start = 0 

    print("Dictionary created !")
    print("Entering the loop")

    # Début de la boucle de counts 
    for end in tqdm(range(len(sequence))):
        while sequence[start].ts + delta < sequence[end].ts :
            # Decrement Counts 
            counts[Motif(Arc(*sequence[start].inds), oriented= motif._oriented)] -= 1
            for suffix in counts.keys() :
                # Si le motif est trop grand
                if len(suffix) > l_motif - 1 : 
                    continue
                else :
                    concat = suffix.add_suffix(Arc(*sequence[start].inds, oriented=oriented_graph))
                    if concat not in counts.keys() : continue #Motif n'est pas dans les sous motifs que l'on recherche
                    else :
                        counts[concat] -= counts[suffix] #Motif issu de la concatenation connu
            start +=1

        # Increment counts
        for prefix in reversed(list(counts.keys())) : 
            if len(prefix) >= l_motif : continue
            else : 
                concat = prefix + Arc(*sequence[end].inds, oriented= motif.oriented)
                if concat not in counts.keys() : continue #Motif n'est pas dans les sous motifs que l'on recherche
                else : counts[concat] += counts[prefix]
        counts[Motif(Arc(*sequence[end].inds), oriented= motif.oriented)] += 1
        
        # Ajout des données de counts dans le dictionnaire
        for motif in submotifs : 
            dict_counts[motif].append(counts[motif])

    return counts, dict_counts, sequence



def counts_interaction_sequence(sequence : list[Interaction]) -> dict[Arc, int] : 
    """Counts the number of occurence of each arc of a given interaction sequence

    Args:
        sequence (list[Interaction]): interaction sequence

    Returns:
        dict[Arc, int]: Counts of each the number of each arc of the sequence 
    """
    counts_nb_interaction = Counter()
    for interaction in sequence : 
        arc = Arc(*interaction.inds, oriented=False)
        counts_nb_interaction[arc] += 1

    print(f" Total du comptage : {counts_nb_interaction.total()}")
    print(f" Nombre d'interaction dans la séquence d'intéraction : {len(sequence)}")

    if counts_nb_interaction.total() != len(sequence) : 
        raise ValueError("Problème lors du comptage")

    print(f" Nombre d'interactions différentes : {len(counts_nb_interaction)}")

    return counts_nb_interaction



def counts_instance_k_motif(size_motif : int, sequence_interactions : list[Interaction], delta : int) -> tuple[int, dict[Motif,int], list[Interaction]] :
    """Generate the counts and the dictionnary of the counts for the motif generated by the k-most frequent arcs of the sequence

    Args:
        size_motif (int): size of the motif (the k most frequent arcs of the interaction sequence )
        sequence_interactions (list[Interaction]): interaction sequence 
        delta (int): delta for the counts

    Returns:
        tuple[int, dict[Motif,int], list[Interaction]]: Counts, dict_counts and filtered interaction sequence
    """
    counts_arcs = counts_interaction_sequence(sequence_interactions)  
    k_most_common = counts_arcs.most_common(size_motif)
    most_common_arcs = [arc for arc,count in k_most_common]
    motif_batch = Motif(*most_common_arcs, oriented=False)
    counts, dict_counts, sequence = count_instance_motif(sequence_raw=sequence_interactions, motif=motif_batch, delta=delta)

    return counts, dict_counts, sequence

#############################################################################
                        # Tests for the module
#############################################################################

if __name__ == "__main__" : 
    date_init = datetime.fromisoformat("2024-12-11T12:00:00")

    M_oriented = Motif(
        Arc("a","b"),
        Arc("a","c"),
        Arc("c","a"))

    M_not_oriented = Motif(
        Arc("a","b"),
        Arc("a","c"),
        Arc("c","a"),
        oriented= False)

    sequence = [
        Interaction("c","a", date_init + timedelta(seconds=25)),
        Interaction("a","c", date_init + timedelta(seconds=17)),
        Interaction("a","b", date_init + timedelta(seconds=28)),
        Interaction("a","c", date_init + timedelta(seconds=30)),
        Interaction("a","c", date_init + timedelta(seconds=35)),
        Interaction("c","a", date_init + timedelta(seconds=15)),
        Interaction("a","c", date_init + timedelta(seconds=32)),
    ]


    # Dictionnaire de comptage qui a pour clé un motif (cf figure 2 du papier)

    M2 = Motif(
        Arc("a","b"),
        Arc("a","c"),
        Arc("b","c"),
        oriented= False)

    M2 = Motif(
        Arc("a","c"),
        Arc("b","a"),
        Arc("c","a"),
        oriented= False)
