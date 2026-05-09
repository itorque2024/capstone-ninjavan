"""
One-time script — seeds the blockchain with status logs for all 20 mock parcels.
Run AFTER deploying ParcelTracker.sol and setting .env variables.

Usage:
    conda run -n ninjavan python src/scripts/seed_blockchain.py
"""
import sys
import time
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.agents.chatbot.blockchain_logger import log_status, is_available

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mock_parcels.csv"


def main():
    if not is_available():
        print("ERROR: Blockchain not configured. Set INFURA_API_KEY, CONTRACT_ADDRESS, and WALLET_PRIVATE_KEY in .env")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Seeding {len(df)} parcels to blockchain...\n")

    for _, row in df.iterrows():
        parcel_id = row["parcel_id"]
        status = row["status"]
        print(f"  Logging {parcel_id} → {status}...", end=" ", flush=True)
        tx_hash = log_status(parcel_id, status)
        if tx_hash:
            print(f"✅ tx: {tx_hash[:20]}…")
        else:
            print("❌ failed (check .env and gas balance)")
        time.sleep(1)   # avoid nonce collision

    print("\nDone. View contract on Etherscan:")
    from src.agents.chatbot.blockchain_logger import get_etherscan_url
    print(f"  {get_etherscan_url('')}")


if __name__ == "__main__":
    main()
