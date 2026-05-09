// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * ParcelTracker — NinjaVan on-chain parcel status audit trail.
 * Deploy to Ethereum Sepolia testnet via Remix IDE (remix.ethereum.org).
 *
 * Instructions:
 *   1. Open remix.ethereum.org in browser
 *   2. Create new file, paste this code
 *   3. Compile (Solidity 0.8.x)
 *   4. Deploy to "Injected Provider - MetaMask" → select Sepolia network
 *   5. Copy the deployed contract address → add to .env as CONTRACT_ADDRESS
 */
contract ParcelTracker {

    struct StatusLog {
        string  status;
        uint256 timestamp;
        address updater;
    }

    // parcel_id (string) → list of status logs
    mapping(string => StatusLog[]) private _history;

    event StatusUpdated(
        string indexed parcelId,
        string  status,
        uint256 timestamp,
        address updater
    );

    /**
     * Log a new status for a parcel. Anyone can call this (demo).
     * In production: restrict with onlyOwner or role-based access.
     */
    function logStatus(string calldata parcelId, string calldata status) external {
        _history[parcelId].push(StatusLog(status, block.timestamp, msg.sender));
        emit StatusUpdated(parcelId, status, block.timestamp, msg.sender);
    }

    /**
     * Read the full status history for a parcel. Free (read-only, no gas).
     */
    function getHistory(string calldata parcelId)
        external
        view
        returns (StatusLog[] memory)
    {
        return _history[parcelId];
    }

    /**
     * Get the latest status only.
     */
    function getLatestStatus(string calldata parcelId)
        external
        view
        returns (string memory status, uint256 timestamp)
    {
        StatusLog[] storage logs = _history[parcelId];
        require(logs.length > 0, "No history for this parcel");
        StatusLog storage latest = logs[logs.length - 1];
        return (latest.status, latest.timestamp);
    }

    /**
     * Count how many status entries exist for a parcel.
     */
    function getHistoryCount(string calldata parcelId)
        external
        view
        returns (uint256)
    {
        return _history[parcelId].length;
    }
}
