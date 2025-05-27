import json
import networkx as nx

import torch
from sklearn.covariance import ledoit_wolf

# Add this function to your existing hierarchical/category.py
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
    'cats_raw_terms' maps chosen category names (e.g., "Budgeting and Saving") 
                     to lists of their raw string terms/phrases.
    'G_finance' represents the hierarchy (e.g., "Personal Finance" -> "Budgeting and Saving").
    'finance_category_keys' will be the list of chosen category names, sorted.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cats_raw_terms = {}
    G_finance = nx.DiGraph()
    finance_category_keys = [] 

    print(f"Loading finance data from: {json_file_path}")
    for l1_key, l1_value in data.items(): 
        if isinstance(l1_value, dict):
            if not G_finance.has_node(l1_key): G_finance.add_node(l1_key)
            for l2_key, l2_value in l1_value.items(): 
                if isinstance(l2_value, list) and all(isinstance(term, str) for term in l2_value):
                    cats_raw_terms[l2_key] = l2_value 
                    if not G_finance.has_node(l2_key): G_finance.add_node(l2_key)
                    G_finance.add_edge(l1_key, l2_key)
                    finance_category_keys.append(l2_key)
                # else: (optional warning for malformed L2 items)
        elif isinstance(l1_value, list) and all(isinstance(term, str) for term in l1_value):
            cats_raw_terms[l1_key] = l1_value
            if not G_finance.has_node(l1_key): G_finance.add_node(l1_key)
            finance_category_keys.append(l1_key)
        # else: (optional warning for malformed L1 items)

    sorted_finance_category_keys = sorted(list(set(finance_category_keys))) 
    nodes_to_keep_in_G = set(sorted_finance_category_keys)
    for l2_node in sorted_finance_category_keys:
        for parent in list(G_finance.predecessors(l2_node)): nodes_to_keep_in_G.add(parent)
    final_G_finance = G_finance.subgraph(list(nodes_to_keep_in_G)).copy()
    print(f"Loaded {len(cats_raw_terms)} finance categories. Graph: {final_G_finance.number_of_nodes()} nodes, {final_G_finance.number_of_edges()} edges.")
    return cats_raw_terms, final_G_finance, sorted_finance_category_keys

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
    category_mean = category_embeddings.mean(dim=0)

    cov = ledoit_wolf(category_embeddings.cpu().numpy())
    cov = torch.tensor(cov[0], device = category_embeddings.device)
    pseudo_inv = torch.linalg.pinv(cov)
    lda_dir = pseudo_inv @ category_mean
    lda_dir = lda_dir / torch.norm(lda_dir)
    lda_dir = (category_mean @ lda_dir) * lda_dir

    return lda_dir, category_mean

def estimate_cat_dir(category_lemmas, unembed, vocab_dict):
    category_embeddings = unembed[category_to_indices(category_lemmas, vocab_dict)]
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
