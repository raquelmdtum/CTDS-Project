### File containing all the functions for easier import

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def custom_tokenizer(text):
   return text.split(", ")

def similarity_search_ingredients(df, query):
   """
   Perform a similarity search based on cosine similarity of TF-IDF vectors.

   Parameters:
   - query (str): The input query string.
   - df (pd.DataFrame): A DataFrame containing 'id' and 'ingredients' columns.

   Returns:
   - results_df (pd.DataFrame): Rows from the original DataFrame sorted by similarity.
   """
   # Initialize TfidfVectorizer with a custom tokenizer (adjust lowercase as needed)
   vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)

   # Extract the 'ingredients' column
   ingredients = df['ingredients']

   # Fit and transform the ingredients
   tfidf_matrix = vectorizer.fit_transform(ingredients)

   # Transform the query into the TF-IDF space
   query_tfidf = vectorizer.transform([query])

   # Compute cosine similarity between the query and all documents
   similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

   # Add similarity scores to the DataFrame
   df['similarity_score_ingredients'] = similarities

   # Sort the DataFrame by similarity scores in descending order
   results_df = df.sort_values(by='similarity_score_ingredients', ascending=False).reset_index(drop=True)
   
   results_df['rank_ingredients'] = range(1, len(results_df) + 1)

   return results_df

def jaccard_similarity(set_a, set_b):
   """
   Calculate the Jaccard Similarity between two sets.
   """
   intersection = len(set_a.intersection(set_b))
   union = len(set_a.union(set_b))
   return intersection / union if union != 0 else 0.0

def similarity_search_highlights(df, token_list):
   """
   Perform similarity search based on Jaccard similarity between df and a token list.
   
   :param df: A pandas DataFrame with columns ['product_id','product_name', 'highlights'].
   :param token_list: A list of tokens to compare against (highlights).
   :return: A DataFrame with IDs and their Jaccard similarity scores, sorted by similarity score.
   """
   # Convert the token list to a set
   token_set = set(token_list)
   
   # List to store the similarity scores
   similarity_scores = []
   
   # Iterate over the rows of the DataFrame
   for index, row in df.iterrows():
      product_id = row['product_id']
      title = row['product_name']
      # Convert the tokens for this ID to a set 
      id_token_set = set(row['highlights'].split(", "))
      
      # Calculate Jaccard similarity
      similarity_score = jaccard_similarity(id_token_set, token_set)
      
      # Append the result as a tuple (id, score)
      similarity_scores.append((product_id, title, similarity_score, (row['highlights'])))
   
   # Convert the list of similarity scores to a DataFrame
   similarity_df = pd.DataFrame(similarity_scores, columns=['product_id', 'product_name', 'similarity_score_highlights', 'highlights'])
   
   # Sort the DataFrame by the 'similarity_score' column in descending order
   similarity_df_sorted = similarity_df.sort_values(by='similarity_score_highlights', ascending=False).reset_index(drop=True)
   
   similarity_df_sorted['rank_highlights'] = range(1, len(similarity_df_sorted) + 1)
   
   return similarity_df_sorted

def reciprocal_rank_fusion(df_highlights, df_ingredients, k=60):
   """
   Compute Reciprocal Rank Fusion (RRF) scores based on rank_highlights and rank_ingredients.

   Parameters:
   - df_highlights (pd.DataFrame): DataFrame containing 'product_id', 'rank_highlights', and other relevant columns.
   - df_ingredients (pd.DataFrame): DataFrame containing 'product_id', 'rank_ingredients', and other relevant columns.
   - k (int): A constant for RRF computation (default=60).

   Returns:
   - combined_df (pd.DataFrame): A new DataFrame with overall RRF scores and combined ranking.
   """
   # Merge the two DataFrames on 'product_id'
   merged_df = pd.merge(
      df_highlights,  # Include all columns from df_highlights
      df_ingredients[['product_id', 'rank_ingredients']],  # Include only product_id and rank_ingredients
      on='product_id',
      how='inner'
   )

   # Fill missing ranks with a large value (e.g., very low relevance)
   merged_df['rank_highlights'] = merged_df['rank_highlights'].fillna(float('inf'))
   merged_df['rank_ingredients'] = merged_df['rank_ingredients'].fillna(float('inf'))

   # Compute the RRF score
   merged_df['rrf_score'] = (
      1 / (k + merged_df['rank_highlights']) +
      1 / (k + merged_df['rank_ingredients'])
   )

   # Sort by the RRF score in descending order
   merged_df = merged_df.sort_values(by='rrf_score', ascending=False).reset_index(drop=True)

   # Add a new rank based on the RRF score
   merged_df['overall_rank'] = range(1, len(merged_df) + 1)
   
   return merged_df

def get_similar_items(product_id, df, n = 5):
   """
   Retrieve the top N products most similar to a given product based on highlights and ingredients.

   This function takes a product ID, performs similarity searches on the product's highlights and ingredients, 
   and combines the results using a reciprocal rank fusion algorithm. It returns the top N most similar products.

   Parameters:
   ----------
   product_id : int or str
      The ID of the product for which similar items are being searched.
   n : int, optional
      The number of similar products to return. Default is 5.

   Returns:
    - merged_results (pd.DataFrame): A new DataFrame containing top n similar items
   """

   # get the selected product
   product = df[df['product_id'] == product_id]

   # get the product highlights and ingredients
   product_highlights =  list(product['highlights'])[0].split(", ")
   product_ingredients = str(product['ingredients'])

   # remove the product I am searching for
   df = df[(df['product_id'] != product_id)]

   # perform similarity searches
   highlights_similarity_results = similarity_search_highlights(df, product_highlights)
   ingredients_similarity_results = similarity_search_ingredients(df, product_ingredients)

   # combine similarity searches with reciprocal rank fusion algorithm
   merged_results = reciprocal_rank_fusion(highlights_similarity_results, ingredients_similarity_results)
   
   # Return only the top-n products
   return merged_results[:n]

def preprocess_rules(rules):
   """
   Preprocess the rules DataFrame to ensure 'antecedents' and 'consequents' are sets.
   """
    # Create a copy of the DataFrame to avoid modifying the original
   rules = rules.copy()
   rules['antecedents'] = rules['antecedents'].apply(lambda x: x.split(", ") if isinstance(x, str) else x)
   rules['consequents'] = rules['consequents'].apply(lambda x: x.split(", ") if isinstance(x, str) else x)
   return rules   

# Function to find items associated with an input item
def find_associated_items(input_item, rules):
   """
   Find all items associated with an input item based on association rules.

   Parameters:
      input_item (str): The item to find associations for.
      rules (pd.DataFrame): A DataFrame of association rules with columns 'antecedents' and 'consequents'.

   Returns:
      list: A list of items associated with the input item, preserving the order of rules.
   """
   # Sort rules by confidence in descending order
   rules = rules.sort_values(by='confidence', ascending=False).reset_index(drop=True)

   input_item_antecedents = [input_item]

   associated_items = []
   
   # Iterate through the rules
   for _, rule in rules.iterrows():
      antecedents = rule['antecedents']
      consequents = rule['consequents']
            
      # Check if the input item is in the antecedents
      if input_item_antecedents == antecedents:
         # Add consequents to the associated items if not already present
         for item in consequents:
               if item not in associated_items:
                  associated_items.append(item)
                  
   return associated_items