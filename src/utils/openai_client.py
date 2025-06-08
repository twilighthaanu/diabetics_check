import openai
from typing import Dict, Any

class AzureOpenAIClient:
    def __init__(self):
        # Initialize with Azure OpenAI credentials
        openai.api_type = "azure"
        openai.api_base = "https://your-resource-name.openai.azure.com"
        openai.api_version = "2023-05-15"
        openai.api_key = "4n5ciztByPDA4FpcHhZNhE6THKRXAKGfcygIOXniZL37XHKgdH6zJQQJ99BEACYeBjFXJ3w3AAAAACOG4Bec"
        self.deployment_name = "gpt-4"

    def analyze_report(self, extracted_text: str, clinical_features: Dict[str, Any]) -> str:
        """Generate an AI analysis of the medical report."""
        try:
            prompt = f"""
            Analyze the following medical report and provide insights about potential diabetes risk factors.

            Extracted Text:
            {extracted_text}

            Clinical Features:
            {clinical_features}

            Please provide a concise analysis focusing on:
            1. Key risk factors for diabetes
            2. Any concerning values that stand out
            3. Recommended next steps
            """

            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant that analyzes medical reports for diabetes risk factors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return response.choices[0].message['content'].strip()

        except Exception as e:
            print(f"Error in Azure OpenAI API call: {str(e)}")
            return "Unable to generate AI analysis at this time."
