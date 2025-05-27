"""
package initialization to support easy imports
(e.g., `import hierarchical as hrc; hrc._category_to_indices`)
"""

from hierarchical.category import *
from hierarchical.plotting import *
from hierarchical.from_models import *

from .category import (
    get_categories, # Original WordNet loader
    load_finance_categories_and_graph, # New finance loader
    phrase_to_model_tokens, # New phrase tokenizer
    category_to_indices, 
    estimate_single_dir_from_embeddings,
    estimate_cat_dir,
    noun_to_gemma_vocab_elements, 
    get_animal_category,
    # Add get_input_embeddings_g if you define it in category.py for notebooks 01-03
    # Or if notebooks 01-03 will get 'g' directly from the model object.
)
from .plotting import (
    cos_heatmap, 
    proj_2d, 
    proj_2d_single_diff, 
    proj_2d_double_diff, 
    show_evaluation
)
from .from_models import (
    get_gamma as get_output_embeddings_gamma, # Alias for clarity
    get_g as get_transformed_g_and_covs,       # Alias for clarity and to avoid name clash
    get_vocab as get_tokenizer_vocab,        # Alias for clarity
    compute_lambdas
)

# It's good practice to define __all__ if using 'from .module import *' elsewhere,
# or rely on these specific imports when using 'import hierarchical as hrc'.
# For hrc.function_name access, the above imports are what matter.
