def _k_init(X, n_clusters, random_state, n_local_trials=None):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features))
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))
        # Pick first center randomly
    center_id = int(n_samples * (np.random.random_sample()))
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    # Initialize list of closest distances and calculate current potential
    """
    closest_dist_sq = cosine_similarity(
        centers[0], X, Y_norm_squared=x_squared_norms, squared=True)
    """
    closest_dist_sq = (1 - cosine_distances(centers[0], X))
    current_pot = closest_dist_sq.sum()
    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = int(n_local_trials * (np.random.random_sample())) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)
        if candidate_ids == n_samples:
            candidate_ids = (n_samples - 1)
        # Compute distances to center candidates
        distance_to_candidates = (1 - cosine_distances(X[candidate_ids], X))
        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,distance_to_candidates)
                                     #distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()
            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids#candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq
        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq
    return centers
_k_init(X = features[0:100], n_clusters  = 10,random_state  = 20)