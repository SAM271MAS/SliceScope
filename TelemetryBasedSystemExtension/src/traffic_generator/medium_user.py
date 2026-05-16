import time
import random
import requests
from datetime import datetime
import math

# --- CONFIGURATION & SETUP ---
print("Starting UE2 (Medium User) - Continuous SIPp Traffic")

PROXY_HOST = "proxy-service.slicescope.svc.cluster.local"
PROXY_PORT = 3128
ENV_CONTROLLER_URL = "http://env-controller-service.slicescope.svc.cluster.local:9081/state/UE2"
PAYLOAD_SERVER_URL = "http://payload-server-service.slicescope.svc.cluster.local:9080"

# LTE CQI to Max Bandwidth Mapping (Mbps)
CQI_BANDWIDTH_CAP_MBPS = {
    15: 80.0, 14: 64.0, 13: 48.0, 12: 36.0, 11: 28.0, 
    10: 24.0, 9: 20.0, 8: 16.0, 7: 12.0, 6: 8.0, 
    5: 4.0, 4: 2.0, 3: 1.0, 2: 0.6, 1: 0.4
}

# --- HELPER FUNCTIONS ---
def log(msg):
    """Helper to print logs with a timestamp for K8s pod monitoring."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def get_current_cqi():
    """Fetches the current CQI state, bypassing the Squid proxy."""
    bypass_proxy = {"http": None, "https": None}
    try:
        response = requests.get(ENV_CONTROLLER_URL, timeout=2, proxies=bypass_proxy)
        response.raise_for_status()
        return response.json().get("cqi", 15)
    except Exception:
        return 15

# --- TRAFFIC GENERATION LOOP ---
# Initial state updated to Mbps and ms
time_step = 0.0
current_noise_tput = 0.0
current_noise_jit = 0.0

# Medium Profile Limits (Smoothed)
BASE_TPUT = 22.0
TREND_AMP_TPUT = 2.5     
NOISE_VOL_TPUT = 0.15    

BASE_JIT = 15.0
TREND_AMP_JIT = 1.0
NOISE_VOL_JIT = 0.05

MACRO_SPEED = 0.005      

app_type = "sipp"
log("Starting continuous smooth traffic generation loop...")

while True:
    loop_start = time.time()
    
    current_cqi = get_current_cqi()
    cqi_max_mbps = CQI_BANDWIDTH_CAP_MBPS.get(current_cqi, 80.0)
    
    scale_factor = cqi_max_mbps / 80.0 
    
    dynamic_base = BASE_TPUT * scale_factor
    dynamic_amp = TREND_AMP_TPUT * scale_factor
    
    # 1. Macro Trend 
    macro_tput = dynamic_base + (dynamic_amp * math.sin(time_step * MACRO_SPEED))
    macro_jit = BASE_JIT + (TREND_AMP_JIT * math.sin(time_step * MACRO_SPEED))

    # 2. Micro Jitter 
    current_noise_tput = (current_noise_tput * 0.95) + random.gauss(0, NOISE_VOL_TPUT)
    current_noise_jit = (current_noise_jit * 0.95) + random.gauss(0, NOISE_VOL_JIT)

    # 3. Combine & Apply Physical Bounds
    target_tput = macro_tput + current_noise_tput
    target_jit = macro_jit + current_noise_jit
    
    actual_mbps = min(cqi_max_mbps, max(0.1, target_tput))
    actual_jitter_ms = max(0.001, target_jit)
    
    # 4. Dispatch Telemetry
    telemetry_url = f"{PAYLOAD_SERVER_URL}/100KB.zip?ml_feature=UE2_{app_type}&cqi={current_cqi}&tput={round(actual_mbps, 4)}&jit={round(actual_jitter_ms, 4)}"
    proxies = {"http": f"http://{PROXY_HOST}:{PROXY_PORT}"}
    
    try:
        requests.get(telemetry_url, proxies=proxies, timeout=2)
        log(f"CQI: {current_cqi:>2} | {app_type:<10} -> Tput: {actual_mbps:>8.2f} Mbps | Jitter: {actual_jitter_ms:>5.2f} ms")
    except Exception as e:
        log(f"Proxy error: {e}")

    time_step += 1.0
    elapsed = time.time() - loop_start
    time.sleep(max(0.0, 1.0 - elapsed))
