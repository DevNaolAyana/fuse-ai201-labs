import random
from datetime import datetime, timedelta, timezone
from api.logging_db import init_db, log_prediction

random.seed(42)
init_db()

now = datetime.now(timezone.utc)

for days_ago in range(14, -1, -1):
    day = now - timedelta(days=days_ago)
    for _ in range(random.randint(40, 80)):
        ts = day.replace(hour=random.randint(6, 20))
        # Simulate a recent spike in fraud probability
        prob = min(0.99, random.betavariate(2, 5) + (0.15 if days_ago <= 6 else 0.0))
        decision = "Fraud" if prob >= 0.5 else "Legitimate"
        
        log_prediction(
            features={"Time": 0, "Amount": 100}, 
            prob=round(prob, 4), 
            decision=decision, 
            model_version="1.0.0", 
            latency_ms=random.uniform(20, 50), 
            timestamp=ts, 
            is_simulated=1  # Mark as simulated!
        )

print("Simulated 15 days of logs successfully!")