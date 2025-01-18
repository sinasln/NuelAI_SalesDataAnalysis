

import openai
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

# Initialize FastAPI
app = FastAPI()

# Set your OpenAI API key
openai.api_key = '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'

class EdaResults(BaseModel):
    correlation_matrix: dict
    vif: dict
    ols_summary: str

@app.post("/analyze/")
def analyze_results(data: EdaResults):
    # Extract data
    correlation_matrix = pd.DataFrame(data.correlation_matrix)
    vif_data = pd.DataFrame(list(data.vif.items()), columns=['Variable', 'VIF'])
    ols_summary = data.ols_summary
    
    # Create a prompt for OpenAI's GPT model
    prompt = f"""
    Analyze the following EDA results:
    
    1. **Correlation Matrix:** {correlation_matrix.to_string()}
    2. **VIF Data:** {vif_data.to_string()}
    3. **OLS Summary:** {ols_summary}
    
    Provide an insightful analysis of the data. Point out high correlations, high VIFs, potential multicollinearity issues, and highlight any relevant OLS results.
    """

    # Send the prompt to OpenAI API for analysis
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Or a model you have access to
            prompt=prompt,
            max_tokens=10000  # Limit the response length as needed
        )

        # Get the text output from OpenAI
        analysis = response.choices[0].text.strip()

        return {"analysis": analysis}

    except Exception as e:
        return {"error": str(e)}

