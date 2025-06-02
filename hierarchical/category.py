import torch
from sklearn.covariance import ledoit_wolf
import numpy as np
import inflect
import json
import networkx as nx
# (Keep other imports like torch, inflect, Ledoit-Wolf at the top of the file)
# Add this function to hierarchical/category.py
# Ensure 'p' (inflect.engine()) is initialized at the module level if noun_to_gemma_vocab_elements uses it
# from transformers import AutoTokenizer # Not needed here if tokenizer is passed as arg

def phrase_to_model_tokens(phrase_str, tokenizer, vocab_set):
    """
    Converts a phrase into a list of unique, valid tokens from the model's vocabulary.
    """
    if not phrase_str or not isinstance(phrase_str, str):
        return []

    tokenized_subwords = tokenizer.tokenize(phrase_str.lower()) 
    valid_model_tokens = [token for token in tokenized_subwords if token in vocab_set]

    # Fallback: If no direct tokens found for the whole phrase, 
    # and if the phrase is multi-word, you could try noun_to_gemma_vocab_elements on individual words.
    # This part is optional and depends on how good the direct tokenization is.
    if not valid_model_tokens and len(phrase_str.split()) > 1:
        # print(f"  Note: No direct tokens for phrase '{phrase_str}'. Trying word-by-word with noun_to_gemma_vocab_elements.")
        fallback_tokens = set()
        for word_in_phrase in phrase_str.split():
            # noun_to_gemma_vocab_elements is already in your category.py
            processed_word_tokens = noun_to_gemma_vocab_elements(word_in_phrase, vocab_set)
            fallback_tokens.update(processed_word_tokens)
        valid_model_tokens = list(fallback_tokens)

    return list(set(valid_model_tokens))

def load_finance_categories_and_graph(json_file_path):
    """
    Loads categories and constructs a graph from the finance.json file.
    'all_cats_raw_terms' will map L1 areas AND L2 specific categories 
                         to lists of their raw string terms/phrases.
                         For L1 areas, terms are aggregated from their L2 children.
    'G_finance' represents the hierarchy (e.g., "Personal Finance" -> "Budgeting and Saving").
    'all_sorted_category_keys' will be a sorted list of all L1 and L2 category keys that have terms.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cats_raw_terms_L2 = {} # For L2 specific categories
    G_finance = nx.DiGraph()
    finance_category_keys_L2 = [] # Stores L2 keys
    
    l1_to_l2_children_map = {} # Helper to map L1 keys to their L2 children keys

    print(f"Loading and processing finance data from: {json_file_path}")
    for l1_key, l1_value in data.items(): 
        if isinstance(l1_value, dict): # L1 key with L2 sub-categories
            if not G_finance.has_node(l1_key): G_finance.add_node(l1_key)
            l1_to_l2_children_map.setdefault(l1_key, [])
            for l2_key, l2_value in l1_value.items(): 
                if isinstance(l2_value, list) and all(isinstance(term, str) for term in l2_value):
                    cats_raw_terms_L2[l2_key] = l2_value 
                    if not G_finance.has_node(l2_key): G_finance.add_node(l2_key)
                    G_finance.add_edge(l1_key, l2_key)
                    finance_category_keys_L2.append(l2_key)
                    l1_to_l2_children_map[l1_key].append(l2_key)
                # else: (optional warning for malformed L2 items)
        elif isinstance(l1_value, list) and all(isinstance(term, str) for term in l1_value):
            # If an L1 key directly contains a list of terms, treat it as a main category
            # (It won't have children in l1_to_l2_children_map for aggregation, but will be in cats_raw_terms_L2)
            cats_raw_terms_L2[l1_key] = l1_value
            if not G_finance.has_node(l1_key): G_finance.add_node(l1_key)
            finance_category_keys_L2.append(l1_key) # Add to L2 keys list for now, it's a "leaf" in terms of terms
        # else: (optional warning for malformed L1 items)

    # Create the dictionary that will hold terms for BOTH L1 and L2 categories
    all_cats_raw_terms = cats_raw_terms_L2.copy() # Start with all L2 categories

    # Now, aggregate terms for L1 categories
    all_l1_graph_nodes = [node for node, in_degree in G_finance.in_degree() if in_degree == 0 and G_finance.out_degree(node) > 0] # More robust way to find L1 root nodes
    
    for l1_key in all_l1_graph_nodes: # Iterate over actual L1 keys found in the graph
        if l1_key in l1_to_l2_children_map: # Check if this L1 key had L2 children
            aggregated_terms_for_l1 = set()
            for l2_child_key in l1_to_l2_children_map.get(l1_key, []):
                aggregated_terms_for_l1.update(cats_raw_terms_L2.get(l2_child_key, []))
            if aggregated_terms_for_l1:
                 all_cats_raw_terms[l1_key] = list(aggregated_terms_for_l1)
            elif l1_key not in all_cats_raw_terms: # If L1 had no children or children had no terms, ensure it's not orphaned if it was a direct list earlier
                print(f"Warning: L1 category '{l1_key}' ended up with no aggregated terms and was not a direct term list.")


    # The keys for iteration in notebooks should now be all keys present in all_cats_raw_terms
    all_sorted_category_keys = sorted(list(all_cats_raw_terms.keys()))
    
    # Prune G_finance to only nodes present in all_cats_raw_terms (should be all L1 and L2 now)
    final_G_finance = G_finance.subgraph(all_sorted_category_keys).copy() 
    # This might remove L1 nodes if they somehow ended up with no terms and weren't L2-equivalents.
    # A final check for sorted keys based on the final graph and cats:
    final_sorted_keys_for_iteration = [k for k in all_sorted_category_keys if k in final_G_finance and k in all_cats_raw_terms and all_cats_raw_terms[k]]

    # Attempt a topological sort for the final keys if the graph is a DAG
    # This helps if there are any inter-dependencies, though less likely in L1->L2.
    # If not a DAG, alphabetical sort is used.
    if final_G_finance.number_of_nodes() > 0:
        try:
            # Filter topologically sorted nodes to those that are actual categories with terms
            topo_sorted_keys = [n for n in nx.topological_sort(final_G_finance) if n in final_sorted_keys_for_iteration]
            # Add any remaining category keys that might be isolated nodes but have terms
            remaining_cat_keys = [k for k in final_sorted_keys_for_iteration if k not in topo_sorted_keys]
            final_sorted_keys_for_iteration = topo_sorted_keys + sorted(remaining_cat_keys)
        except nx.NetworkXUnfeasible: 
            print("Warning (load_finance): Graph has cycles or is not a DAG. Using alphabetical sort for final category keys.")
            # final_sorted_keys_for_iteration is already sorted alphabetically if we take 'all_sorted_category_keys'
            # and filter it by G and non-empty terms. This is implicitly handled by how it was built.
    
    print(f"Loaded and processed {len(all_cats_raw_terms)} finance categories (L1 & L2 with terms).")
    print(f"Final finance graph: {final_G_finance.number_of_nodes()} nodes, {final_G_finance.number_of_edges()} edges.")
    print(f"Returning {len(final_sorted_keys_for_iteration)} keys for iteration.")
    
    return all_cats_raw_terms, final_G_finance, final_sorted_keys_for_iteration

def get_categories(noun_or_verb = 'noun', model_name = 'gemma'):

    cats = {}
    if noun_or_verb == 'noun':
        with open(f'data/noun_synsets_wordnet_{model_name}.json', 'r') as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist(f"data/noun_synsets_wordnet_hypernym_graph_{model_name}.adjlist", create_using=nx.DiGraph())
    elif noun_or_verb == 'verb':
        with open(f'data/verb_synsets_wordnet_{model_name}.json', 'r') as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist(f"data/verb_synsets_wordnet_hypernym_graph_{model_name}.adjlist", create_using=nx.DiGraph())
    
    cats = {k: list(set(v)) for k, v in cats.items() if len(set(v)) > 50}
    G = nx.DiGraph(G.subgraph(cats.keys()))

    reversed_nodes = list(reversed(list(nx.topological_sort(G))))
    for node in reversed_nodes:
        children = list(G.successors(node))
        if len(children) == 1:
            child = children[0]
            parent_lemmas_not_in_child = set(cats[node]) - set(cats[child])
            if len(list(G.predecessors(child))) == 1 or len(parent_lemmas_not_in_child) <5:
                grandchildren = list(G.successors(child))
                for grandchild in grandchildren:
                    G.add_edge(node, grandchild)
                G.remove_node(child)

    G = nx.DiGraph(G.subgraph(cats.keys()))
    sorted_keys = list(nx.topological_sort(G))
    cats = {k: cats[k] for k in sorted_keys}

    return cats, G, sorted_keys



def category_to_indices(category, vocab_dict):
    return [vocab_dict[w] for w in category]

def get_words_sim_to_vec(query: torch.tensor, unembed, vocab_list, k=300):
    similar_indices = torch.topk(unembed @ query, k, largest=True).indices.cpu().numpy()
    return [vocab_list[idx] for idx in similar_indices]

def estimate_single_dir_from_embeddings(category_embeddings):
    if not isinstance(category_embeddings, torch.Tensor) or category_embeddings.nelement() == 0:
        # Attempt to get embedding dimension from elsewhere if possible, or default
        # This case should ideally be caught before calling this with empty/invalid embeddings
        print("Warning (estimate_single_dir): Received empty or invalid category_embeddings.")
        # Try to determine expected dimension, e.g., from a global config or a passed 'g' matrix
        # For now, returning zero vectors of a default/common dimension.
        # This should be refined based on how g.shape[1] can be accessed or passed.
        D_EMBED_FALLBACK = 2048 # Example, adjust if possible
        return torch.zeros(D_EMBED_FALLBACK), torch.zeros(D_EMBED_FALLBACK)

    if category_embeddings.shape[0] == 0: # No embeddings
        dim = category_embeddings.shape[1] if category_embeddings.ndim > 1 and category_embeddings.shape[1] > 0 else 2048 
        return torch.zeros(dim, device=category_embeddings.device), torch.zeros(dim, device=category_embeddings.device)

    category_embeddings = category_embeddings.float() # Ensure float32
    category_mean = category_embeddings.mean(dim=0)

    if category_embeddings.shape[0] < 2: # LDA needs at least 2 samples
        norm_mean = torch.norm(category_mean)
        if norm_mean < 1e-9 : return category_mean, category_mean 
        lda_dir_normed = category_mean / norm_mean
        lda_dir_final = (category_mean @ lda_dir_normed) * lda_dir_normed 
        return lda_dir_final, category_mean

    embeddings_np = np.ascontiguousarray(category_embeddings.cpu().numpy()) 
    try:
        cov = ledoit_wolf(embeddings_np)[0]
    except Exception as e: 
        var_np = np.var(embeddings_np, axis=0)
        if np.all(np.abs(var_np) < 1e-9): # Check absolute value for near-zero
             cov = np.eye(embeddings_np.shape[1]) * 1e-6 
        else:
             cov = np.diag(var_np + 1e-6) 
    cov_torch = torch.tensor(cov, device=category_embeddings.device, dtype=torch.float32)

    try:
        pseudo_inv = torch.linalg.pinv(cov_torch, rcond=1e-5) # Add rcond for stability
    except Exception as e_pinv:
        mean_diag_cov = torch.mean(torch.diag(cov_torch)) + 1e-9
        pseudo_inv = torch.eye(cov_torch.shape[0], device=category_embeddings.device, dtype=torch.float32) / mean_diag_cov

    lda_dir = pseudo_inv @ category_mean
    lda_dir_norm_val = torch.norm(lda_dir)

    if lda_dir_norm_val < 1e-9: 
        norm_mean = torch.norm(category_mean)
        if norm_mean < 1e-9: return category_mean, category_mean
        lda_dir_final = category_mean / norm_mean 
    else:
        lda_dir_normalized = lda_dir / lda_dir_norm_val
        lda_dir_final = (category_mean @ lda_dir_normalized) * lda_dir_normalized 

    return lda_dir_final, category_mean

def estimate_cat_dir(category_tokens, unembed_matrix, vocab_dict):
    """Estimates LDA and mean directions for a category given its model tokens."""
    category_indices = category_to_indices(category_tokens, vocab_dict) # category_to_indices should be defined

    dim_embed = unembed_matrix.shape[1]
    if not category_indices:
        return {'lda': torch.zeros(dim_embed, device=unembed_matrix.device), 
                'mean': torch.zeros(dim_embed, device=unembed_matrix.device)}

    category_embeddings = unembed_matrix[category_indices]
    lda_dir, category_mean = estimate_single_dir_from_embeddings(category_embeddings)

    return {'lda': lda_dir, 'mean': category_mean}



import inflect
p = inflect.engine()

def noun_to_gemma_vocab_elements(word, vocab_set):
    word = word.lower()
    plural = p.plural(word)
    add_cap_and_plural = [word, word.capitalize(), plural, plural.capitalize()]
    add_space = ["▁" + w for w in add_cap_and_plural]
    return vocab_set.intersection(add_space)


def get_animal_category(data, categories, vocab_dict, g):
    vocab_set = set(vocab_dict.keys())

    animals = {}
    animals_ind = {}
    animals_g = {}
    animals_token = {}

    for category in categories:
        animals[category] = []
        animals_ind[category] = []
        animals_g[category] = []
        animals_token[category] = []

    for category in categories:
        lemmas = data[category]
        for w in lemmas:
            animals[category].extend(noun_to_gemma_vocab_elements(w, vocab_set))
        
        for word in animals[category]:
            animals_ind[category].append(vocab_dict[word])
            animals_token[category].append(word)
            animals_g[category] = g[animals_ind[category]]
    return animals_token, animals_ind, animals_g
