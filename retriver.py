import requests

def query_api(query: str):
    BASE_URL = "https://mop.rekt.life/v1/query"
    PARAMS = {"query": query}
    
    try:
        response = requests.get(BASE_URL, params=PARAMS)
        
        if response.status_code == 200:
            response = response.json()
            top_items = sorted(response["data"], key=lambda x: x['distance'])[:3]  # Get top 2 items
            return [item['chunk'] for item in top_items]
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}

