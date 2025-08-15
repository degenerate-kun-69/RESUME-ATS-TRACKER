from langchain_core.tools import tool
import numpy as np


#add similarity from faiss query results
@tool("Confidence_score_calculation_tool")
def confidence_score_calculation_tool(similarity:float)->float:
    """
    Calculate the confidence score based on the similarity percentage with additional weighting factors.
    
    This function transforms the raw similarity into a confidence score that accounts for:
    - Similarity strength (exponential weighting)
    - Market competitiveness factor
    - Uncertainty penalty for low similarities
    
    Args:
        similarity (float): The similarity percentage between 0 and 100.
        
    Returns:
        float: The confidence score, adjusted from the raw similarity percentage.
    """
    if not (0 <= similarity <= 100):
        raise ValueError("Similarity must be between 0 and 100.")
    
    # Normalize similarity to 0-1 range
    norm_sim = similarity / 100
    
    # Apply confidence transformation factors:
    
    # 1. Exponential weighting - higher similarities get boosted more
    exponential_factor = np.power(norm_sim, 0.8)
    
    # 2. Market competitiveness penalty - reduce confidence for mid-range scores
    competitiveness_factor = 1 - (0.3 * np.exp(-2 * (norm_sim - 0.5)**2))
    
    # 3. Uncertainty penalty for very low similarities
    uncertainty_penalty = 1 if norm_sim > 0.3 else (norm_sim / 0.3) * 0.7 + 0.3
    
    # Combine factors
    confidence = exponential_factor * competitiveness_factor * uncertainty_penalty
    
    # Convert back to percentage and ensure bounds
    confidence_score = np.clip(confidence * 100, 0, 100)
    
    return np.round(confidence_score, 2)


@tool("Hiring_decision_tool")
def hiring_decision_tool(confidence: float, threshold: float=75) -> str:
    """
    Make a hiring decision based on the confidence score and a threshold.

    Args:
        confidence (float): The confidence score of the candidate.
        threshold (float, optional): The threshold for making a positive hiring decision. Defaults to 75.

    Returns:
        str: The hiring decision ("Hire" or "No Hire").
    """
    if not (0 <= confidence <= 100):
        raise ValueError("Confidence must be between 0 and 100.")

    if confidence >= threshold:
        return "Hire"
    else:
        return "No Hire"