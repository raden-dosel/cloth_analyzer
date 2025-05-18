from locust import HttpUser, task, between
import random

SAMPLE_TEXTS = [
    "Formal summer dress in dark colors",
    "Casual cotton t-shirt with logo",
    "No polyester or slim fit please",
    "Office wear suitable for humid weather"
]

class ClothingAnalyzerUser(HttpUser):
    wait_time = between(0.5, 2.5)
    
    @task(5)
    def analyze_short_text(self):
        self.client.post("/analyze", json={
            "text": random.choice(SAMPLE_TEXTS)
        })
    
    @task(1)
    def analyze_long_text(self):
        self.client.post("/analyze", json={
            "text": "I need a formal outfit for an outdoor wedding in summer. " 
                    "Prefer breathable fabrics like linen but avoid light colors. " 
                    "No synthetic materials please."
        })
    
    @task(1)
    def health_check(self):
        self.client.get("/health")