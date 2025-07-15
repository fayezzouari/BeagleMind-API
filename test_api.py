import requests
import json

API_BASE_URL = "http://localhost:8000/api"

def test_api():
    print("Testing Information Retrieval API...")
    
    print("\n1. Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Health check: {response.json()}")
    
    print("\n2. Initializing retrieval system...")
    init_data = {"collection_name": "beaglemind_w_chonkie"}
    response = requests.post(f"{API_BASE_URL}/initialize", json=init_data)
    
    if response.status_code == 200:
        print(f"Initialization successful: {response.json()}")
        
        print("\n3. Testing retrieval...")
        query_data = {
            "query": "machine learning and artificial intelligence",
            "n_results": 5,
            "include_metadata": True,
            "rerank": True
        }
        
        response = requests.post(f"{API_BASE_URL}/retrieve", json=query_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Retrieval successful!")
            print(f"Total found: {result['total_found']}")
            print(f"Filtered results: {result['filtered_results']}")
            print(f"Documents: {len(result['documents'][0])}")
        else:
            print(f"Retrieval failed: {response.status_code} - {response.text}")
    else:
        print(f"Initialization failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_api()
