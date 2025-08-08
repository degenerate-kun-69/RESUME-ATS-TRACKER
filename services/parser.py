import json
import re

def parse_json_response(response_text):
    try:
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif response_text.strip().startswith("{"):
            json_text = response_text.strip()
        else:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_text = match.group()
            else:
                raise ValueError("No JSON found")

        return json.loads(json_text)
    except Exception as e:
        return {
            "Job Description Match": "Unable to parse",
            "Missing Keywords and Skills": ["Analysis failed"],
            "Confidence Score": 0,
            "Profile Summary": f"Error parsing response: {str(e)}",
            "Should I hire them or not?": "no"
        }
