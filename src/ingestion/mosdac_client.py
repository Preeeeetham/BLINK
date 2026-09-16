"""
Official MOSDAC API Client for INSAT-3DS / INSAT-3DR Satellite Feeds.
Implements the exact ISRO MOSDAC protocol (gettoken, datasets.json, download API)
matching the official MOSDAC data download tool (mdapi.py & config.json).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
import numpy as np


logger = logging.getLogger("blink.mosdac_client")


@dataclass
class MOSDACObservation:
    """Represents an available MOSDAC satellite observation granule."""
    satellite_id: str
    dataset_id: str
    record_id: str
    identifier: str
    prod_date: str
    timestamp_utc: str
    timestamp_ist: str
    local_path: Optional[str]
    is_cached: bool


class MOSDACClient:
    """
    Client for ISRO Meteorological & Oceanographic Satellite Data Archival Centre (MOSDAC).
    Implements authentication (gettoken), catalogue search (datasets.json), and binary
    HDF5 streaming (download API) according to official ISRO specifications.
    """

    TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
    SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
    CHECK_INTERNET_URL = "https://mosdac.gov.in/download_api/check-internet"
    DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
    REFRESH_URL = "https://mosdac.gov.in/download_api/refresh-token"
    LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"

    DEFAULT_DATASET_ID = "3SIMG_L1B_STD"      # INSAT-3DS Level-1B Standard Imager
    DEFAULT_INSAT3DR_DATASET_ID = "3RIMG_L1B_STD"  # INSAT-3DR Level-1B Standard Imager

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = "config.json",
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        # Prefer config.local.json if caller used default config.json
        resolved_config = Path(config_path) if config_path else Path("config.json")
        if resolved_config == Path("config.json") and Path("config.local.json").exists():
            resolved_config = Path("config.local.json")
        self.config_path = resolved_config

        # Initial default
        self.cache_dir = Path(cache_dir or "data/raw_netcdf/mosdac")

        self.username = username or ""
        self.password = password or ""
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

        # Load from config file first to get custom path and credentials
        self._load_from_config_file()

        # If explicit username/password were passed to constructor, ensure they take precedence
        if username:
            self.username = username
        if password:
            self.password = password

        # Check environment variables as overrides or fallbacks
        env_user = os.environ.get("MOSDAC_USERNAME")
        env_pwd = os.environ.get("MOSDAC_PASSWORD")
        if env_user:
            self.username = env_user
        if env_pwd:
            self.password = env_pwd

        # NOW ensure directories are created based on final resolved path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_cache_dir = self.cache_dir / "processed"
        self.processed_cache_dir.mkdir(parents=True, exist_ok=True)

        # Parser for processing HDF5 -> NumPy
        from src.ingestion.mosdac_parser import MOSDACParser
        self.parser = MOSDACParser()

    def _load_from_config_file(self) -> bool:
        """Loads credentials from config.json or config.local.json if available."""
        # Prefer config.local.json if available and config_path is default config.json
        cfg_path = self.config_path
        if cfg_path == Path("config.json") and Path("config.local.json").exists():
            cfg_path = Path("config.local.json")
            self.config_path = cfg_path

        if not cfg_path.exists():
            # If default config.json doesn't exist, check env vars
            env_user = os.environ.get("MOSDAC_USERNAME")
            env_pwd = os.environ.get("MOSDAC_PASSWORD")
            if env_user:
                self.username = env_user
            if env_pwd:
                self.password = env_pwd
            return False
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Windows path preprocessing
            fixed = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', raw)
            data = json.loads(fixed)

            creds = data.get("user_credentials", {})
            user = creds.get("username/email") or creds.get("username")
            pwd = creds.get("password")

            if user and user not in ("your_username", "your_email@example.com"):
                self.username = user
            if pwd and pwd not in ("your_password", "your_mosdac_password"):
                self.password = pwd

            # Environment variables always take precedence if defined
            env_user = os.environ.get("MOSDAC_USERNAME")
            env_pwd = os.environ.get("MOSDAC_PASSWORD")
            if env_user:
                self.username = env_user
            if env_pwd:
                self.password = env_pwd

            dl_settings = data.get("download_settings", {})
            custom_dl_path = dl_settings.get("download_path")
            env_dl_path = os.environ.get("MOSDAC_DOWNLOAD_PATH")
            target_dl_path = env_dl_path or custom_dl_path
            if target_dl_path and target_dl_path != "/home/sys_oper/MOSDAC_Downloads/":
                self.cache_dir = Path(target_dl_path)
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            return True
        except Exception as e:
            logger.warning(f"Could not load {cfg_path}: {e}")
            return False

    @property
    def is_configured(self) -> bool:
        """Returns True if user has configured valid non-default credentials."""
        return bool(
            self.username
            and self.password
            and self.username not in ("your_username", "your_email@example.com")
            and self.password not in ("your_password", "your_mosdac_password")
        )

    def set_credentials(self, username: str, password: str) -> None:
        """Sets and persists credentials. Saves to config.local.json if default config.json is target."""
        self.username = username.strip()
        self.password = password.strip()
        self.access_token = None
        self.refresh_token = None

        # Write to config.local.json if using default config.json to keep git tree clean
        target_path = Path("config.local.json") if self.config_path == Path("config.json") else self.config_path
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        config_data = {
            "user_credentials": {
                "username/email": self.username,
                "password": self.password,
            },
            "search_parameters": {
                "datasetId": self.DEFAULT_DATASET_ID,
                "startTime": today_str,
                "endTime": today_str,
                "count": "",
                "boundingBox": "",
                "gId": "",
            },
            "download_settings": {
                "download_path": str(self.cache_dir.resolve()).replace("\\", "/"),
                "organize_by_date": False,
                "skip_user_input": True,
                "generate_error_logs": True,
                "error_logs_dir": "error_logs",
            },
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        self.config_path = target_path

    def authenticate(self) -> bool:
        """
        Authenticates with MOSDAC token endpoint:
        POST https://mosdac.gov.in/download_api/gettoken
        """
        if not self.is_configured:
            return False

        payload = {"username": self.username, "password": self.password}
        try:
            resp = requests.post(self.TOKEN_URL, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                return bool(self.access_token)
            else:
                logger.warning(f"MOSDAC Auth failed with status {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"MOSDAC Auth network exception: {e}")
            return False

    def search_live_catalog(
        self,
        dataset_id: str = DEFAULT_DATASET_ID,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries official MOSDAC search endpoint:
        GET https://mosdac.gov.in/apios/datasets.json
        """
        params = {"datasetId": dataset_id}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if count:
            params["count"] = str(count)

        try:
            resp = requests.get(self.SEARCH_URL, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                entries = data.get("entries", [])
                results = []
                for item in entries:
                    results.append({
                        "id": item.get("id"),
                        "identifier": item.get("identifier"),
                        "updated": item.get("updated"),
                        "datasetId": dataset_id,
                    })
                return results
        except Exception as e:
            logger.warning(f"MOSDAC search catalog failed: {e}")
        return []

    def download_file(self, record_id: str, identifier: str) -> Optional[Path]:
        """
        Downloads a specific HDF5 granule using Bearer token:
        GET https://mosdac.gov.in/download_api/download?id=<record_id>
        """
        if not self.access_token:
            if not self.authenticate():
                return None

        dest_file = self.cache_dir / identifier
        if dest_file.exists():
            return dest_file

        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"id": record_id}

        try:
            resp = requests.get(self.DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=15)
            if resp.status_code == 401 and self.refresh_token:
                # Refresh token and retry
                ref_resp = requests.post(self.REFRESH_URL, json={"refresh_token": self.refresh_token}, timeout=8)
                if ref_resp.status_code == 200:
                    self.access_token = ref_resp.json().get("access_token")
                    headers = {"Authorization": f"Bearer {self.access_token}"}
                    resp = requests.get(self.DOWNLOAD_URL, headers=headers, params=params, stream=True, timeout=15)

            if resp.status_code == 200:
                tmp_file = self.cache_dir / (identifier + ".part")
                with open(tmp_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
                os.rename(tmp_file, dest_file)
                return dest_file
            else:
                logger.warning(f"MOSDAC download failed with status {resp.status_code}")
        except Exception as e:
            logger.warning(f"MOSDAC download exception: {e}")
        return None

    def get_processed_scan(
        self,
        record_id: str,
        identifier: str,
        target_size: Optional[Tuple[int, int]] = (512, 512)
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        High-efficiency data fetcher:
        1. Checks for processed compressed cache.
        2. If missing, downloads raw HDF5, parses it, saves compressed version, and PURGES raw file.
        """
        processed_path = self.processed_cache_dir / (identifier + ".npz")

        if processed_path.exists():
            try:
                return self.parser.load_processed_data(processed_path)
            except Exception as e:
                logger.warning(f"Failed to load processed cache {processed_path}: {e}")

        # Not in cache, download raw
        raw_path = self.download_file(record_id, identifier)
        if raw_path:
            try:
                # Process and resample
                channel_data = self.parser.read_hdf5(raw_path, target_size=target_size)
                # Save compressed version
                self.parser.save_processed_data(processed_path, channel_data)
                # PURGE raw file immediately to save space
                os.unlink(raw_path)
                logger.info(f"Processed and purged raw file: {identifier}")
                return channel_data
            except Exception as e:
                logger.error(f"Processing failed for {identifier}: {e}")

        return None

    def fetch_today_pair(self, t0_time_str: str = "00:00", cadence_minutes: int = 15, satellite_id: str = "INSAT-3DS") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Convenience method to pull exactly two images (T0 and T1) for the current date.
        """
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        return self.fetch_scan_pair(
            date_str=today_str,
            t0_time_str=t0_time_str,
            cadence_minutes=cadence_minutes,
            satellite_id=satellite_id
        )

    def query_available_scans(
        self,
        date_str: str,
        start_hour_utc: int = 0,
        end_hour_utc: int = 23,
        satellite_id: str = "INSAT-3DS",
    ) -> List[Dict[str, Any]]:
        """
        Queries or generates synoptic 15-minute scan list for a given date.
        If live catalog is reachable, complements with real MOSDAC granule IDs.
        """
        scans = []
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            target_date = datetime.utcnow().date()

        dataset_id = self.DEFAULT_DATASET_ID if satellite_id == "INSAT-3DS" else self.DEFAULT_INSAT3DR_DATASET_ID

        # Try live catalog search
        live_entries = self.search_live_catalog(
            dataset_id=dataset_id,
            start_time=date_str,
            end_time=date_str,
        )
        live_map = {e["identifier"]: e["id"] for e in live_entries}

        for hour in range(start_hour_utc, min(24, end_hour_utc + 1)):
            for minute in [0, 15, 30, 45]:
                dt_utc = datetime(target_date.year, target_date.month, target_date.day, hour, minute)
                dt_ist = dt_utc + timedelta(hours=5, minutes=30)

                prefix = "3SIMG" if satellite_id == "INSAT-3DS" else "3RIMG"
                file_name = f"{prefix}_{dt_utc.strftime('%d%b%Y_%H%M')}_L1B_STD_V01R00.h5".upper()

                local_path = self.cache_dir / file_name
                is_cached = local_path.exists()
                record_id = live_map.get(file_name)

                scans.append({
                    "satellite_id": satellite_id,
                    "dataset_id": dataset_id,
                    "record_id": record_id,
                    "timestamp_utc": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "timestamp_ist": dt_ist.strftime("%d-%m-%Y %H:%M IST"),
                    "file_name": file_name,
                    "local_path": str(local_path) if is_cached else None,
                    "is_cached": is_cached,
                    "channels": ["IMG_VIS", "IMG_SWIR", "IMG_MWIR", "IMG_WV", "IMG_TIR1", "IMG_TIR2"],
                    "spatial_resolution_km": {"VIS": 1.0, "SWIR": 1.0, "MIR": 4.0, "WV": 8.0, "TIR": 4.0},
                    "coverage": {"lat_north": 35.0, "lat_south": 5.0, "lon_west": 65.0, "lon_east": 100.0},
                })

        return scans

    def fetch_scan_pair(
        self,
        date_str: str,
        t0_time_str: str,
        cadence_minutes: int = 15,
        satellite_id: str = "INSAT-3DS",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Pairs two consecutive observations T0 and T1, downloading real HDF5 if authenticated.
        """
        try:
            hour, minute = map(int, t0_time_str.split(":"))
            t0_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=minute)
        except Exception:
            t0_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        t1_dt = t0_dt + timedelta(minutes=cadence_minutes)

        scans_t0 = self.query_available_scans(
            t0_dt.strftime("%Y-%m-%d"),
            start_hour_utc=t0_dt.hour,
            end_hour_utc=t0_dt.hour,
            satellite_id=satellite_id,
        )
        scans_t1 = self.query_available_scans(
            t1_dt.strftime("%Y-%m-%d"),
            start_hour_utc=t1_dt.hour,
            end_hour_utc=t1_dt.hour,
            satellite_id=satellite_id,
        )

        match_0 = next(
            (s for s in scans_t0 if s["timestamp_utc"].endswith(t0_dt.strftime("%H:%M:00Z"))),
            scans_t0[0] if scans_t0 else None,
        )
        match_1 = next(
            (s for s in scans_t1 if s["timestamp_utc"].endswith(t1_dt.strftime("%H:%M:00Z"))),
            scans_t1[0] if scans_t1 else None,
        )

        # If authenticated and record_ids are known, download and process files
        if self.is_configured and match_0 and match_0.get("record_id"):
            data0 = self.get_processed_scan(match_0["record_id"], match_0["file_name"])
            if data0:
                match_0["local_path"] = str(self.processed_cache_dir / (match_0["file_name"] + ".npz"))
                match_0["is_cached"] = True
                match_0["processed_data"] = data0

        if self.is_configured and match_1 and match_1.get("record_id"):
            data1 = self.get_processed_scan(match_1["record_id"], match_1["file_name"])
            if data1:
                match_1["local_path"] = str(self.processed_cache_dir / (match_1["file_name"] + ".npz"))
                match_1["is_cached"] = True
                match_1["processed_data"] = data1

        return match_0, match_1
