import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def get_stratified_splits(df: pd.DataFrame, y: pd.Series, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Try to stratify, if a class has < n_splits members, it will fail
    # So we group rare classes or fallback to KFold
    value_counts = y.value_counts()
    rare_classes = value_counts[value_counts < n_splits].index
    
    if len(rare_classes) > 0:
        # Fallback to KFold if stratification is impossible
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(df))
    else:
        return list(skf.split(df, y))

def get_covariate_splits(df: pd.DataFrame, n_splits: int, seed: int):
    """
    Simulate covariate shift by projecting numeric features onto the first principal component,
    sorting by this projection, and binning into n_splits contiguous blocks.
    Each fold uses n_splits-1 blocks for training and 1 block for testing.
    """
    numeric_df = df.select_dtypes(include=["number"]).fillna(0)
    if numeric_df.shape[1] == 0:
        # Fallback if no numeric features
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(df))
        
    pca = PCA(n_components=1, random_state=seed)
    # Scale briefly for PCA
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(numeric_df)
    pc1 = pca.fit_transform(scaled).flatten()
    
    # Sort indices by PC1
    sorted_idx = np.argsort(pc1)
    
    # Split into n_splits bins
    bins = np.array_split(sorted_idx, n_splits)
    
    splits = []
    for i in range(n_splits):
        test_idx = bins[i]
        train_idx = np.concatenate([bins[j] for j in range(n_splits) if j != i])
        splits.append((train_idx, test_idx))
        
    return splits

def get_population_splits(df: pd.DataFrame, n_splits: int, seed: int):
    """
    Simulate population/domain shift using KMeans clustering.
    We cluster the data into n_splits clusters.
    Each fold uses n_splits-1 clusters for training and 1 cluster for testing.
    """
    numeric_df = df.select_dtypes(include=["number"]).fillna(0)
    if numeric_df.shape[1] == 0:
        # Fallback if no numeric features
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(df))
        
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(numeric_df)
    
    kmeans = KMeans(n_clusters=n_splits, random_state=seed, n_init="auto")
    clusters = kmeans.fit_predict(scaled)
    
    splits = []
    for i in range(n_splits):
        test_idx = np.where(clusters == i)[0]
        train_idx = np.where(clusters != i)[0]
        
        # If a cluster is entirely empty (rare but possible), fallback to random for this fold
        if len(test_idx) == 0 or len(train_idx) == 0:
            np.random.seed(seed + i)
            shuffled = np.random.permutation(len(df))
            split_point = int(len(df) * 0.8)
            train_idx = shuffled[:split_point]
            test_idx = shuffled[split_point:]
            
        splits.append((train_idx, test_idx))
        
    return splits
