"""
Blockchain Logger — reads/writes parcel status to Ethereum Sepolia testnet.
Uses web3.py + Infura endpoint (from lecture 6.5.1).

Graceful degradation: if INFURA_API_KEY or CONTRACT_ADDRESS is not set,
all functions return safe fallback values — the app works without Web3.

Setup (one-time, user must do):
  1. Sign up at infura.io → create project → copy Sepolia endpoint key
  2. Deploy contracts/ParcelTracker.sol via remix.ethereum.org to Sepolia
  3. Add to .env:
       INFURA_API_KEY=your_key_here
       CONTRACT_ADDRESS=0x...
       WALLET_PRIVATE_KEY=0x...   (needed only for logStatus writes)
"""
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

_INFURA_KEY      = os.environ.get("INFURA_API_KEY")
_CONTRACT_ADDR   = os.environ.get("CONTRACT_ADDRESS")
_PRIVATE_KEY     = os.environ.get("WALLET_PRIVATE_KEY")
_RPC_URL         = f"https://sepolia.infura.io/v3/{_INFURA_KEY}" if _INFURA_KEY else None
_ETHERSCAN_BASE  = "https://sepolia.etherscan.io"

# Minimal ABI — only the functions we call
_ABI = [
    {
        "inputs": [
            {"name": "parcelId", "type": "string"},
            {"name": "status",   "type": "string"},
        ],
        "name": "logStatus",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "parcelId", "type": "string"}],
        "name": "getHistory",
        "outputs": [
            {
                "components": [
                    {"name": "status",    "type": "string"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "updater",   "type": "address"},
                ],
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

_w3 = None
_contract = None


def _get_contract():
    global _w3, _contract
    if _contract is not None:
        return _contract
    if not _RPC_URL or not _CONTRACT_ADDR:
        return None
    try:
        from web3 import Web3
        _w3 = Web3(Web3.HTTPProvider(_RPC_URL))
        if not _w3.is_connected():
            return None
        _contract = _w3.eth.contract(
            address=Web3.to_checksum_address(_CONTRACT_ADDR),
            abi=_ABI,
        )
        return _contract
    except Exception:
        return None


def is_available() -> bool:
    """True if blockchain connection is configured and live."""
    return _get_contract() is not None


def get_history(parcel_id: str) -> list[dict]:
    """
    Read parcel status history from blockchain. Free (read-only, no gas).
    Returns list of {"status": str, "timestamp": str, "tx_url": str} dicts.
    """
    contract = _get_contract()
    if not contract:
        return []
    try:
        logs = contract.functions.getHistory(parcel_id).call()
        result = []
        for log in logs:
            status, ts, updater = log
            dt = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
            result.append({
                "status": status,
                "timestamp": dt,
                "updater": updater,
            })
        return result
    except Exception:
        return []


def log_status(parcel_id: str, status: str) -> str | None:
    """
    Write a status update to the blockchain. Costs gas (test ETH).
    Returns transaction hash string, or None on failure.
    """
    contract = _get_contract()
    if not contract or not _PRIVATE_KEY or not _w3:
        return None
    try:
        account = _w3.eth.account.from_key(_PRIVATE_KEY)
        nonce = _w3.eth.get_transaction_count(account.address)
        txn = contract.functions.logStatus(parcel_id, status).build_transaction({
            "chainId": 11155111,   # Sepolia chain ID
            "gas": 200000,
            "gasPrice": _w3.eth.gas_price,
            "nonce": nonce,
        })
        signed = _w3.eth.account.sign_transaction(txn, private_key=_PRIVATE_KEY)
        tx_hash = _w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()
    except Exception:
        return None


def get_etherscan_url(parcel_id: str) -> str:
    """Returns Etherscan URL to view parcel's contract interactions."""
    if _CONTRACT_ADDR:
        return f"{_ETHERSCAN_BASE}/address/{_CONTRACT_ADDR}"
    return f"{_ETHERSCAN_BASE}"
