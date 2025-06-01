#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate Recommendation System ML Model Training & Tuning
This script performs training and tuning for the KMeans clustering model
used in the candidate recommendation system.
"""

import os
import sys
import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import dagshub
import warnings

# Suppress specific pandas warnings that we're handling properly
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("candidate_recommendation.training")

# Set MLflow tracking URI and experiment
# Tentukan target logging: dagshub atau local
TARGET = os.getenv("MLFLOW_TARGET", "dagshub")  # default: dagshub

if TARGET == "dagshub":
    dagshub.init(repo_owner='lexynotfound', repo_name='mlops', mlflow=True)
    MLFLOW_URI = "https://dagshub.com/lexynotfound/mlops.mlflow"
    print("[MLflow] Logging to DagsHub")
elif TARGET == "local":
    mlflow.set_tracking_uri("http://localhost:8000")
    MLFLOW_URI = "http://localhost:8000"
    print("[MLflow] Logging to LOCALHOST")
else:
    raise ValueError("Unknown MLFLOW_TARGET. Use 'dagshub' or 'local'.")

# Tetap set experiment setelah tracking URI
EXPERIMENT_NAME = "candidate_recommendation_system"
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data(data_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load preprocessed data for model training.

    Args:
        data_dir: Directory containing preprocessed data files

    Returns:
        Tuple of features array (X) and target dataframe
    """
    try:
        data_path = os.path.join(data_dir, 'processed_data.npy')
        target_path = os.path.join(data_dir, 'target_data.csv')

        if not os.path.exists(data_path) or not os.path.exists(target_path):
            raise FileNotFoundError(f"Data files not found in {data_dir}")

        X = np.load(data_path)
        target_df = pd.read_csv(target_path)

        logger.info(f"Data loaded successfully: X shape={X.shape}, target rows={len(target_df)}")
        return X, target_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_text_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Clean text data by removing NaN values and ensuring string type.

    Args:
        df: DataFrame containing text data
        column: Column name containing text to clean

    Returns:
        Cleaned DataFrame
    """
    # Check for NaN values
    nan_count = df[column].isna().sum()
    logger.info(f"Total NaN values in {column}: {nan_count}")

    # Drop NaN values
    df_clean = df.dropna(subset=[column]).copy()
    logger.info(f"Rows after cleaning NaN: {len(df_clean)} (removed {len(df) - len(df_clean)} rows)")

    # Ensure string type
    df_clean.loc[:, column] = df_clean[column].astype(str)

    return df_clean


def find_optimal_k(X_train: np.ndarray, k_range: range) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        X_train: Training data
        k_range: Range of k values to try

    Returns:
        Tuple of optimal k value and list of silhouette scores
    """
    silhouette_scores = []

    for k in k_range:
        logger.info(f"Training KMeans with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_train)
        score = silhouette_score(X_train, cluster_labels)
        silhouette_scores.append(score)
        logger.info(f"K={k}, Silhouette Score={score:.4f}")

    optimal_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"Optimal k value: {optimal_k}")

    return optimal_k, silhouette_scores


def analyze_clusters(model: KMeans, X_test: np.ndarray) -> Dict[str, float]:
    """
    Analyze cluster quality using multiple metrics.

    Args:
        model: Trained KMeans model
        X_test: Test data

    Returns:
        Dictionary of evaluation metrics
    """
    test_labels = model.predict(X_test)

    metrics = {
        "silhouette_score": silhouette_score(X_test, test_labels),
        "calinski_harabasz_score": calinski_harabasz_score(X_test, test_labels),
        "davies_bouldin_score": davies_bouldin_score(X_test, test_labels)
    }

    logger.info(f"Test metrics: {metrics}")
    return metrics


def visualize_clusters(cluster_centers: np.ndarray, save_path: str = "cluster_centers.png") -> str:
    """
    Create visualization of cluster centers if possible.

    Args:
        cluster_centers: Model cluster centers
        save_path: Path to save visualization

    Returns:
        Path to saved visualization
    """
    if cluster_centers.shape[1] < 2:
        logger.warning("Cannot visualize clusters: need at least 2 dimensions")
        return None

    plt.figure(figsize=(10, 8))

    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, c='red', marker='X')

    # Add labels
    for i, (x, y) in enumerate(cluster_centers[:, :2]):
        plt.annotate(f"Cluster {i}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    plt.title('Cluster Centers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Cluster visualization saved to {save_path}")
    return save_path


def analyze_position_clusters(position_matrix, cluster_centers: np.ndarray, top_n: int = 3) -> List[Tuple[int, float]]:
    """
    Analyze relationship between positions and clusters.

    Args:
        position_matrix: TF-IDF matrix of positions
        cluster_centers: Model cluster centers
        top_n: Number of closest clusters to return

    Returns:
        List of tuples with (cluster_idx, distance)
    """
    if position_matrix.shape[0] == 0:
        logger.warning("No position data available for analysis")
        return []

    # Use first position as example
    sample_position = position_matrix[0]

    # Calculate distances between position and cluster centers
    distances = []
    for i, center in enumerate(cluster_centers):
        # Handle different dimensions
        if sample_position.shape[1] < center.shape[0]:
            center_subset = center[:sample_position.shape[1]]
            dist = np.linalg.norm(sample_position.toarray().flatten() - center_subset)
        else:
            sample_subset = sample_position.toarray().flatten()[:center.shape[0]]
            dist = np.linalg.norm(sample_subset - center)
        distances.append((i, dist))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])

    # Return top_n closest clusters
    return distances[:top_n]


def train_model(data_dir: str = '../preprocessing/dataset/career_form_preprocessed') -> Tuple[Any, Any]:
    """
    Train the candidate recommendation model and log results to MLflow.

    Args:
        data_dir: Directory containing preprocessed data

    Returns:
        Tuple of (trained_model, vectorizer)
    """
    logger.info(f"Starting model training with data from {data_dir}")

    try:
        # Load data
        X, target_df = load_data(data_dir)

        # Clean text data
        target_df_clean = clean_text_data(target_df, 'desired_positions')

        # Start MLflow run
        with mlflow.start_run(run_name="candidate_clustering_model") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started with ID: {run_id}")

            # Log dataset info
            mlflow.log_param("data_path", data_dir)
            mlflow.log_param("data_shape", X.shape)
            mlflow.log_param("original_records", len(target_df))
            mlflow.log_param("clean_records", len(target_df_clean))

            # Split data for validation
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            logger.info(f"Data split: train={X_train.shape}, test={X_test.shape}")

            # Find optimal number of clusters
            k_range = range(2, 10)
            optimal_k, silhouette_scores = find_optimal_k(X_train, k_range)

            # Log silhouette scores for each k
            for k, score in zip(k_range, silhouette_scores):
                mlflow.log_metric(f"silhouette_score_k{k}", score)

            # Create visualization of silhouette scores
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, silhouette_scores, 'o-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score by Number of Clusters')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('silhouette_scores.png')
            plt.close()
            mlflow.log_artifact('silhouette_scores.png')

            # Log optimal k
            mlflow.log_param("optimal_k", optimal_k)

            # Train final model with optimal k
            logger.info(f"Training final model with k={optimal_k}")
            final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_model.fit(X_train)

            # Evaluate model
            metrics = analyze_clusters(final_model, X_test)

            # Log evaluation metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Save cluster centers visualization
            cluster_centers = final_model.cluster_centers_
            viz_path = visualize_clusters(cluster_centers)
            if viz_path:
                mlflow.log_artifact(viz_path)

            # Save cluster centers as CSV
            centers_df = pd.DataFrame(cluster_centers)
            centers_df.to_csv("cluster_centers.csv", index=False)
            mlflow.log_artifact("cluster_centers.csv")

            # Process text data for job matching
            logger.info("Creating TF-IDF vectorizer for job positions")
            tfidf = TfidfVectorizer(stop_words='english')
            position_matrix = tfidf.fit_transform(target_df_clean['desired_positions'])

            # Analyze position-cluster relationships
            position_clusters = analyze_position_clusters(position_matrix, cluster_centers)

            # Log position-cluster relationships
            for i, (cluster_idx, dist) in enumerate(position_clusters):
                mlflow.log_metric(f"closest_cluster_{i + 1}", cluster_idx)
                mlflow.log_metric(f"closest_cluster_{i + 1}_distance", dist)

            # Log vectorizer and model
            logger.info("Logging model artifacts to MLflow")
            mlflow.sklearn.log_model(final_model, "kmeans_model")
            mlflow.sklearn.log_model(tfidf, "tfidf_vectorizer")

            # Log position matrix
            joblib.dump(position_matrix, "position_matrix.pkl")
            mlflow.log_artifact("position_matrix.pkl")

            # Log additional parameters
            mlflow.log_param("model_type", "KMeans")
            mlflow.log_param("vectorizer", "TfidfVectorizer")
            mlflow.log_param("n_clusters", optimal_k)
            mlflow.log_param("cluster_centers_shape", str(cluster_centers.shape))

            # Print summary
            logger.info(f"Model training completed successfully")
            logger.info(f"Model trained with {X.shape[0]} samples and {X.shape[1]} features")
            logger.info(f"Optimal number of clusters: {optimal_k}")
            logger.info(f"Test silhouette score: {metrics['silhouette_score']:.4f}")

            return final_model, tfidf

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        logger.info("Starting candidate recommendation model training")
        model, vectorizer = train_model()
        logger.info("Model training completed and logged to MLflow")

        # Print MLflow UI URL for convenience
        print(f"\nView MLflow UI at: {MLFLOW_URI}")
        print(
            f"View experiment at: {MLFLOW_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)