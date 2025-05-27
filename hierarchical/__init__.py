"""
package initialization to support easy imports
(e.g., `import hierarchical as hrc; hrc.category_to_indices`)
"""

# From category.py
from .category import (
    get_categories,                      # Original WordNet loader
    load_finance_categories_and_graph, # New finance loader
    phrase_to_model_tokens,            # New phrase tokenizer
    category_to_indices, 
    estimate_single_dir_from_embeddings,
    estimate_cat_dir,
    noun_to_gemma_vocab_elements, 
    get_animal_category
    # If you add get_input_embeddings_g here, import it too
    # from .category import get_input_embeddings_g # Example
)

# From plotting.py
from .plotting import (
    cos_heatmap, 
    proj_2d, 
    proj_2d_single_diff, 
    proj_2d_double_diff, 
    show_evaluation
)

# From from_models.py
from .from_models import (
    get_gamma as get_output_embeddings_gamma,    # For notebook 07 & transformed_g
    get_g as get_transformed_g_and_covs,          # For notebook 04 (returns transformed g)
    get_vocab as get_tokenizer_vocab,            # For all notebooks (flexible version)
    compute_lambdas
)
