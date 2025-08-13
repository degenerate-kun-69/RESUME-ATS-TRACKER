from langchain_core.tools import tool
import numpy as np


#add similarity from faiss query results
@tool("Confidence_score_calculation_tool")
def confidence_score_calculation_tool(similarity:float)->float:
    """
    Calculate the confidence score based on the similarity percentage.
    
    Args:
        similarity (float): The similarity percentage between 0 and 100.
        
    Returns:
        float: The confidence score, which is the same as the similarity percentage.
    """
    if not (0 <= similarity <= 100):
        raise ValueError("Similarity must be between 0 and 100.")
    
    # The confidence score is directly derived from the similarity percentage
    return np.round(similarity, 2)


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