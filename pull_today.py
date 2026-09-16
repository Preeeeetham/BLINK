from src.ingestion.mosdac_client import MOSDACClient
from datetime import datetime, timedelta
import os

def main():
    print("=" * 60)
    print("  BLINK: DATA PULLER (Optimized)")
    print("=" * 60)

    # Initialize client
    client = MOSDACClient()

    if not client.is_configured:
        print("[ERROR] MOSDAC credentials not configured in config.json")
        print("Please update config.json with your username and password.")
        return

    print(f"Storage Path: {client.cache_dir}")
    print(f"Processed Path: {client.processed_cache_dir}")

    # Try yesterday since today's data might not be live yet
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    t0_time = "00:00"

    print(f"\nPulling images for yesterday ({yesterday}) starting at {t0_time} UTC...")

    try:
        # Manually calling fetch_scan_pair for yesterday's date
        t0, t1 = client.fetch_scan_pair(
            date_str=yesterday,
            t0_time_str=t0_time,
            cadence_minutes=15
        )

        if t0 and t1:
            print("\n[SUCCESS] Successfully pulled and processed 2 images.")
            print(f"  - Frame T0: {t0['file_name']} -> {t0['local_path']}")
            print(f"  - Frame T1: {t1['file_name']} -> {t1['local_path']}")
            print(f"\n[INFO] Compressed .npz files are stored in: {client.processed_cache_dir}")
        else:
            print("\n[FAILED] Could not retrieve images from MOSDAC. They may not be available for this date.")

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
