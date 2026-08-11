import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini Pro model
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
def generate_retention_strategy(customer_id, risk_segment, churn_prob, clv, top_driver):
    """
    Acts as an AI Retention Agent. It reasons through the customer's 
    value and the technical reason for their churn risk.
    """
    
    # Building a detailed prompt for multi-step reasoning
    prompt = f"""
    You are an AI Retention Specialist for a SaaS company. 
    Analyze the following customer data point:
    - Customer ID: {customer_id}
    - Risk Segment: {risk_segment}
    - Churn Probability: {churn_prob:.2%}
    - Annual Revenue at Risk (CLV): Rs. {clv}
    - Primary Churn Driver (from SHAP analysis): {top_driver}

    Instructions:
    1. Evaluate if the customer is worth a high-value intervention based on CLV.
    2. Map the SHAP driver ({top_driver}) to a specific business solution.
    3. Generate a personalized, empathetic email draft.
    4. Provide a 1-sentence internal 'Reasoning' for the manager.

    Return the output in this format:
    Recommended Offer: [Insert Offer]
    Internal Reasoning: [Insert Reasoning]
    Email Draft: [Insert Email]
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating strategy: {str(e)}"
    
def process_retention_layer(at_risk_df, top_n=5):
    """
    Processes the top high-risk customers through the AI Agent.
    Filters by High Risk to save API tokens.
    """
    # Filter for high-risk customers as defined in Layer 6 (business_engine.py)
    high_risk = at_risk_df[at_risk_df['Risk_Segment'] == 'HIGH RISK'].head(top_n)
    
    retention_output = []
    
    for _, row in high_risk.iterrows():
        # Calling our Gemini Agent
        strategy_text = generate_retention_strategy(
            customer_id=row['CustomerID'],
            risk_segment=row['Risk_Segment'],
            churn_prob=row['Churn_Probability'],
            clv=row['Annual_CLV'],
            top_driver=row['Top_Driver']
        )
        
        retention_output.append({
            "CustomerID": row['CustomerID'],
            "Full_Strategy": strategy_text
        })
        
    return retention_output