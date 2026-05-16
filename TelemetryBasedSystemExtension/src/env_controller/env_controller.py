from fastapi import FastAPI
import asyncio
import random
from datetime import datetime

app = FastAPI()

# Initial state: Assume all users start with a perfect LTE connection (CQI 15)
network_state = {
    "UE1": {"cqi": 15},
    "UE2": {"cqi": 15},
    "UE3": {"cqi": 15}
}

def log(msg: str):
    """Helper to print logs with a timestamp for K8s pod monitoring. 
    Uses flush=True to ensure Uvicorn doesn't buffer the output."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

async def simulate_mobility():
    """
    Background task: Simulates user mobility and signal attenuation.
    Updated to use a weighted Markov-chain approach for realistic, gradual drifting.
    """
    log("Mobility simulation started. Shifting network weather every 30 seconds.")
    while True:
        await asyncio.sleep(30)
        
        for ue in network_state:
            # 70% chance to stay stable, 15% chance to drift by 1
            shift = random.choices([-1, 0, 1], weights=[15, 70, 15], k=1)[0]
            
            # Keep CQI within realistic physical bounds (7 to 15)
            new_cqi = max(7, min(15, network_state[ue]["cqi"] + shift))
            network_state[ue]["cqi"] = new_cqi
            
        log(f"Network Weather Update: {network_state}")

@app.on_event("startup")
async def startup_event():
    log("Environment Controller initializing...")
    asyncio.create_task(simulate_mobility())

@app.get("/state/{ue_id}")
def get_state(ue_id: str):
    """
    Endpoint called by user pods to retrieve their current network limits.
    If an unknown UE requests state, it defaults to a perfect connection.
    """
    current_state = network_state.get(ue_id, {"cqi": 15})
    return current_state

