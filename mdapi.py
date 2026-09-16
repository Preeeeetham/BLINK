import requests
import os
from pathlib import Path
import json
import glob
import time
import logging
import threading
from datetime import datetime
import re
import sys

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("\n[INFO] 'tqdm' Library is Not Installed on your system. Hence, the Progress Bar while Downloading Files will appear a bit differently.\n")

token_url = "https://mosdac.gov.in/download_api/gettoken"
search_url = "https://mosdac.gov.in/apios/datasets.json"
check_internet_url = "https://mosdac.gov.in/download_api/check-internet"
download_url = "https://mosdac.gov.in/download_api/download"
refresh_url = "https://mosdac.gov.in/download_api/refresh-token"
logout_url = "https://mosdac.gov.in/download_api/logout"

def preprocess_json(raw_json):
    """
    Escapes Unescaped Backslashes for Windows-style Paths provided in 'config.json' 
    """
    fixed_json = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', raw_json)
    fixed_json = re.sub(r'(?<!\\)\\(?=\s*")', r'\\\\', fixed_json)

    return fixed_json

def load_config(): 
    """Loads and validates configuration from config.local.json or config.json."""
    config_filename = "config.local.json" if os.path.exists("config.local.json") else "config.json"
    try:
        with open(config_filename, "r", encoding="utf-8") as file:
            raw_config = file.read()
        
        try:
            config = json.loads(raw_config)
        except json.JSONDecodeError:
            # Preprocess the JSON to fix common Windows Path issues
            fixed_json = preprocess_json(raw_config)
            try:
                config = json.loads(fixed_json)
            except json.JSONDecodeError:
                print(f"[ERROR] Invalid JSON format in '{config_filename}'! Please correct it and Try Again.")
                sys.exit(1)

        # Validate required fields
        required_fields = ["user_credentials", "search_parameters"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing Required Config Section: {field} inside '{config_filename}'")
            
        # Sets Current Directory for Download if 'download_settings' not specified
        if "download_settings" not in config:
            print(f"\n[Warning]: 'download_settings' not set in '{config_filename}'. Downloading in the Current Directory..")
            config["download_settings"] = {
                "download_path": ""
            }
        return config
    
    except FileNotFoundError as e:
        print(f"[ERROR] Neither 'config.local.json' nor 'config.json' was Found!")
        exit(1)
        
    except ValueError as e:
        print(f"[ERROR] in configuration: {e}")
        exit(1)

config_file = load_config()

# Fetching Information from configuration with environment variable override
user_creds = config_file.get('user_credentials', {})
username = os.environ.get("MOSDAC_USERNAME") or user_creds.get("username/email", "")
password = os.environ.get("MOSDAC_PASSWORD") or user_creds.get("password", "")

download_settings = config_file.get('download_settings', {})

# Retrieves Download Path from env var or config
env_dl_path = os.environ.get("MOSDAC_DOWNLOAD_PATH")
download_path = (env_dl_path or download_settings.get("download_path", "")).replace("\\", "/") or os.path.join(os.getcwd(), "MOSDAC Data Download")

use_date_structure = download_settings.get("organize_by_date", False)
skip_user_input = download_settings.get("skip_user_input", False)
generate_logs = download_settings.get("generate_error_logs", False)

bool_fields = {
    "organize_by_date": use_date_structure,
    "skip_user_input": skip_user_input,
    "generate_error_logs": generate_logs
}

invalid_fields = []

# Validates Fields of 'download_settings' of 'config.json'
for field, value in bool_fields.items():
    if not isinstance(value, bool):
        invalid_fields.append((field, value))
    
if invalid_fields:
    print("\n[ERROR] Configuration Error: The following fields must be either: true or false (Boolean):")
    for field, value in invalid_fields:
        print(f" - '{field}' has Invalid Value: {value}")
    print("\nPlease Correct these in your 'config.json' and Try Again.\n")
    sys.exit(1)

search_params = config_file['search_parameters']
datasetId = search_params.get("datasetId", "")
startTime = search_params.get("startTime", "")
endTime = search_params.get("endTime", "")
startIndex = int(search_params.get("startIndex", 1) or 1)
count = search_params.get("count", "")
boundingBox = search_params.get("boundingBox", "")
gId = search_params.get("gId", "")



logger = logging.getLogger("client_error_logger")

try:
    if generate_logs:
        # Set up Error Logging if enabled
        error_logs_dir = download_settings.get("error_logs_dir") or os.path.join(os.getcwd(), "error_logs")
        os.makedirs(error_logs_dir, exist_ok=True)

        date_str = datetime.now().strftime("%d-%m-%Y")
        log_file_path = os.path.join(error_logs_dir, f"{date_str}_error.log")

        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S"
        )

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.ERROR)
        logger.propogate = False
except PermissionError:
        print(f"\n[ERROR]: No Permission to Write on '{error_logs_dir}'. Please Check and Update Directory Permissions or use Another Directory for Storing Logs.\n")
        sys.exit(1)
except Exception as e:
        print(f"\nException encountered in Generating Logs: {e}\n")

def supports_color():
    if sys.platform != "win32":
        return True
    return "ANSICON" in os.environ or "WT_SESSION" in os.environ or os.environ.get("TERM_PROGRAM") == "vscode"

if supports_color():
    GREEN = "\033[92m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
else:
    GREEN = RED = RESET = BOLD = UNDERLINE = ""

def get_token():
    """Fetch access token from the token endpoint."""

    data = {
        "username": username, 
        "password": password
    }
    try:
        response = requests.post(token_url,json=data)

        # Catches and Displays - 'Server Maintainance' messages from Server Side
        if response.status_code == 503:
            print("Service Unavailable: ", response.json().get("message"))
            
        # Checks for Validation Errors before Raising Exceptions for Other Errors
        if response.status_code == 400:
            try:
                resp = response.json()
                err_msg = resp['error']
                print(f"\n[ERROR] Validation Error: {err_msg}.\n")
                if generate_logs:
                    logger.error(f"Validation Error was encountered while Fetching Token.\nHere are the Error Details: {err_msg}")
            except ValueError:
                print("\n[ERROR] Received status 400 but could not parse response.\n")
                if generate_logs:
                    logger.error("Validation Error: Status 400 [Validation Error] received but response was not JSON.", exc_info=True)
            sys.exit(1)

        if response.status_code == 401:
            try:
                resp = response.json()
                err_msg = resp['error']
                print(f"{err_msg}\n")
                if generate_logs:
                    logger.error(f"{err_msg}\n")
            except ValueError:
                print("\n[ERROR] Received status 401 but could not parse response.\n")
                if generate_logs:
                    logger.error("Validation Error: Status 401 [Invalid Username/Password] received but response was not JSON.", exc_info=True)
            sys.exit(1)    

        response.raise_for_status()
        token_response = response.json()
        return {
            "access_token": token_response.get("access_token"),
            "refresh_token": token_response.get("refresh_token")
        }, username
    
    except requests.exceptions.RequestException as e:
        
        error_msg = str(e)
        if '503 Server Error' in error_msg:
            print("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            if generate_logs:
                logger.error("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            sys.exit(1)
        elif 'Service Unavailable for url' in error_msg:
            print("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            if generate_logs:
                logger.error("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            sys.exit(1)
        elif 'Not Found for url' in error_msg:
            print("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            if generate_logs:
                logger.error("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            sys.exit(1)
        elif 'Max retries exceeded with url: /download_api/gettoken':
            print("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            if generate_logs:
                logger.error("\nServer Unavailable: The server is currently unreachable or not responding.\nPlease Try Again later or Contact Support if the issue persists. Thank you for your patience!\n")
            sys.exit(1)
        else:
            print("[ERROR] Error Occured in 'get_token()': ", error_msg)
            if generate_logs:
                logger.error("Error fetching Access Token: ", exc_info=True) 

        sys.exit(1) 

def format_size(size_mb):
    if size_mb < 1024:
        return f"{size_mb:,.2f} MB"
    elif size_mb < 1024 ** 2:
        size_gb = size_mb / 1024
        return f"{size_gb:,.2f} GB"
    else:
        size_tb = size_mb / (1024 ** 2)
        return f"{size_tb:,.2f} TB"

# Fetches Total Results found for User's Search
def search_results():
    """Fetches all data from the search endpoint using pagination.""" 
    print()
    print("Searching Data for Provided Parameters...")
    data = {"datasetId": datasetId}

    optional_parameters = {
    "startTime": startTime,
    "endTime": endTime,
    "count": count,
    "boundingBox": boundingBox,
    "gId": gId
    }

    # Filters out Empty Values
    data.update({k: v for k, v in optional_parameters.items() if v})

    try:
        res = requests.get(search_url, params=data)
        if res.status_code == 200:
            list = res.json()
            totalResults = list["totalResults"]
            totalSize = list["totalSizeMB"]
            itemsPerPage = list["itemsPerPage"]

            formatted_size = format_size(totalSize)

            if count != "":
                if skip_user_input:
                    print(f"\n{UNDERLINE}{itemsPerPage}{RESET} Files Found for {datasetId}{RESET}")
                else:
                    print(f"\n{UNDERLINE}{itemsPerPage}{RESET} Files Found for {datasetId}{RESET}.\nDo you want to Download them? [Y/N]: ")
                return list['itemsPerPage']

            if skip_user_input:
                print(f"\n{UNDERLINE}{totalResults:,}{RESET} Files Found with Total Size of {UNDERLINE}{formatted_size}{RESET}")
            else:
                print(f"\n{UNDERLINE}{totalResults:,}{RESET} Files Found with Total Size of {UNDERLINE}{formatted_size}{RESET}.\nDo you want to Download them? [Y/N]: ")
            return list["totalResults"]
        
        elif res.status_code // 100 in [4, 5]: 
            list = res.json()
            error_message = list['message'][0] 
            print(f"\n[ERROR] Error Fetching Data from Search Endpoint. Please enter correct 'search_parameters' in your 'config.json' and try again.\n\nStatus Code: {res.status_code}\nError Message: {error_message}\n")
            if generate_logs:
                logger.error(f"\n\nError Fetching Data from Search Endpoint.\n\nError Message: {error_message}\nError Details: {list}\n")
            sys.exit(1)
        
    except (requests.ConnectionError, requests.Timeout):
        print("\n[ERROR] Network Error: No Internet Connection Detected.\nPlease check your Network Connection and try running the application again.\n")
        if generate_logs:
                logger.error("\nNetwork Error: No Internet Connection Detected.\nPlease check your Network Connection and try running the application again.\n", exc_info=True)
        sys.exit(1)

    except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Unexpected Status Code encountered in Search API's Response:\nError Details: {e}")
            if generate_logs:
                logger.error(f"\nUnexpected Status Code encountered in Search API's Response:\nError Details: ", exc_info=True)
            sys.exit(1)

def fetch_and_download_data(total_files, access_token, refresh_token):
    """Fetches all data from the search endpoint using pagination.""" 

    batch_size = 100
    start_Index = 1
    
    counter = 1
    download_count = 0
    skip_count = 0

    if skip_user_input == False:
        print("\nStarting with Data Download..")

    data = {"datasetId": datasetId}

    try:
        res = requests.get(check_internet_url, json=data)
        if res.status_code == 200:
            list = res.json()
            if list[0][0] == 1:
                print("This Product is not yet Released on Internet. Please try searching for a different 'datasetId'.\nExiting...")
                logout()
                sys.exit(1) # Exit if Product not Released on Internet

    except Exception as e:
        print("Exception occured while retrieving Internet Check: ", e)
        logger.error("Exception occured while retrieving Internet Check: ", exc_info=True)

    optional_parameters = {
    "startTime": startTime,
    "endTime": endTime,
    "count": count,
    "boundingBox": boundingBox,
    "gId": gId
    }
    # Filters out Empty Values
    data.update({k: v for k, v in optional_parameters.items() if v})

    try:
        while counter <= total_files: 
            
            data["startIndex"] = start_Index # Sets 'startIndex' for Pagination 
            try:
                res = requests.get(search_url, params=data)
                
                if res.status_code == 200:
                    list=res.json()
                    
                    if not list: # Stops if No More Results
                        break

                    for item in list['entries']:
                        identifier = item['identifier']
                        record_id = item['id']
                        prod_date = item['updated']
                        file_path = download_data(access_token, record_id, identifier, prod_date, counter, total_files)
                        
                        if file_path == 'NOT_RELEASED':
                            print("This Product is not yet Released on MOSDAC. Please try searching for a different 'datasetId'.\nExiting...")
                            logout()
                            sys.exit(1)

                        if file_path == "Invalid/Expired Token":
                            new_access_token = refresh_access_token(refresh_token)
                            if (new_access_token):
                                access_token = new_access_token['access_token'] # Updates New Access Token Globally
                                refresh_token = new_access_token['refresh_token'] # Updates New Refresh Token Globally
                                file_path = download_data(access_token, record_id, identifier, prod_date, counter, total_files)
                                counter += 1
                            else:
                                print("\n[ERROR] Token could not be Refreshed due to Invalid Refresh Token. Stopping Download...") 
                                if generate_logs:
                                    logger.error("\nThere was an Error encountered to Refresh Access Token due to Invalid Refresh Token provided, and hence, Download cannot proceed.")
                                logout()
                                sys.exit(1) # Exit if Token Refresh Fails

                        counter += 1

                        # Calculating Total Download Statistics
                        if file_path and os.path.exists(file_path):
                            download_count += 1
                        elif not file_path:
                            skip_count += 1

                    # Increments startIndex for Next Batch
                    start_Index += batch_size
                
                else:
                    print(f"\nUnexpected Status Code: {res.status_code}")
                    res.raise_for_status()

            except requests.exceptions.RequestException as e:
                res_json = res.json()
                error_message = res_json['message'][0] 
                print(f"\n\n[ERROR] Error Fetching Data from 'fetch_and_download_data()' method.\n\nError Message: {error_message}\nError Details: {e}")
                if generate_logs:
                    logger.error("\n\nError Fetching Data from 'fetch_and_download_data()' method.\n\nError Message: {error_message}\nError Details: ", exc_info=True)
                break # Exits loop on Error

        if counter == (total_files + 1):
            return True, download_count, skip_count
        else:
            return False, 0, 0
    except KeyboardInterrupt:
        print("\nDownload Interrupted By User. Exiting..")
        return False, 0, 0
    except PermissionError:
        print(f"\n[ERROR]: No Permission to Write on '{download_path}'. Please Check and Update Directory Permissions or use Another Directory.")
        if generate_logs:
            logger.error(f"\nPermission Error encountered: No Permission to Write to {download_path}, hence could not Proceed with Download.\nPlease Check and Update the Permission for Writing files inside: {download_path}")
        print("Logging Out..")
        logout()
        sys.exit(1)
    except Exception as e:
        print(f"\nException encountered in 'fetch_and_download_data()': {e}\n")

def get_user_input():
    try:
        if skip_user_input:
            print(f"\n{GREEN}'skip_user_input' Option Enabled in 'config.json'. Proceeding with Download...{RESET}")
            return "yes"
        
        while True:
            user_response = input().strip().lower()
            if user_response in ("y", "n", 'yes', 'no'):
                return user_response
            print(f"{RED}Invalid Input. Please Input 'Y' or 'N':{RESET}")
    except KeyboardInterrupt:
        print("\nDownload Cancelled By User. Exiting..\n")
        sys.exit(1)
    except Exception as e:
        print("[ERROR] Exception occured in 'get_user_input()': ", e)
        if generate_logs:
            logger.error("There was an Exception encounterd in the 'get_user_input()' method.\nError Details: ", exc_info=True)
        sys.exit(1)

def download_data(bearer_token, record_id, identifier, prod_date, counter, total_files): 
    """Download data using the record ID and collection with resumable chunk streaming."""
    os.makedirs(download_path, exist_ok=True)

    if use_date_structure:
        dataset_download_path = os.path.join(download_path, datasetId)
        if prod_date is None:
            folder_structure = dataset_download_path
        else:
            date_obj = datetime.strptime(prod_date, "%Y-%m-%dT%H:%M:%SZ")
            year = date_obj.strftime("%Y")
            day = date_obj.strftime("%d")
            month_abbr = date_obj.strftime("%b").upper()
            day_month = f"{day}{month_abbr}"
            folder_structure = os.path.join(dataset_download_path, year, day_month)
    else:
        folder_structure = download_path

    os.makedirs(folder_structure, exist_ok=True) 
    file_path = os.path.join(folder_structure, identifier)
    tmp_file_path = file_path + ".part"

    if os.path.exists(file_path):
        print(f"\n[INFO] {identifier} Already Exists in {folder_structure}. Skipping Download..")
        return file_path

    MAX_RETRIES = 10
    session = requests.Session()

    for attempt in range(1, MAX_RETRIES + 1):
        existing_bytes = os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Encoding": "identity",
        }
        params = {"id": record_id}

        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        try:
            response = session.get(download_url, headers=headers, params=params, stream=True, timeout=90)

            if response.status_code == 400:
                resp = response.json()
                print(f"\n[ERROR] Validation Error: {resp.get('error', '')}\n")
                return None
            
            if response.status_code == 401:
                return "Invalid/Expired Token"
            
            if response.status_code == 404:
                return "NOT_RELEASED"
            
            if response.status_code == 429:
                print("\n[INFO] Rate limit reached. Waiting 20 seconds...")
                time.sleep(20)
                continue

            if response.status_code == 416:
                # Requested range not satisfiable -> file already fully downloaded or invalid range
                if os.path.exists(tmp_file_path):
                    os.rename(tmp_file_path, file_path)
                    return file_path
                existing_bytes = 0

            response.raise_for_status()

            is_resumed = (response.status_code == 206)
            content_length = int(response.headers.get('Content-Length', 0))
            total_size = (existing_bytes + content_length) if is_resumed else content_length
            file_mode = "ab" if (is_resumed and existing_bytes > 0) else "wb"
            initial_progress = existing_bytes if is_resumed else 0

            file_size_mb = f"{total_size / (1024 * 1024):.2f} MB" if total_size > 0 else "Unknown Size"
            if attempt == 1 or not is_resumed:
                print(f"\n[{counter}/{total_files}] | Downloading: {identifier} | File Size: {file_size_mb}")
            else:
                print(f"\n[INFO] Resuming {identifier} from {existing_bytes / (1024*1024):.2f} MB / {file_size_mb} (Attempt {attempt}/{MAX_RETRIES})...")

            download_interrupted = False
            try:
                with open(tmp_file_path, file_mode) as file:
                    if HAS_TQDM and total_size > 0:
                        tqdm_kwargs = {"ascii": True} if sys.platform == "win32" else {}
                        with tqdm(
                            desc="Progress",
                            total=total_size,
                            initial=initial_progress,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            smoothing=0.3,
                            dynamic_ncols=True,
                            **tqdm_kwargs
                        ) as bar:
                            for chunk in response.iter_content(chunk_size=1048576):
                                if chunk:
                                    file.write(chunk)
                                    bar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=1048576):
                            if chunk:
                                file.write(chunk)
            except Exception as stream_err:
                print(f"\n[WARNING] Stream interrupted ({stream_err}). Will resume in 5 seconds...")
                download_interrupted = True

            if download_interrupted:
                time.sleep(5)
                continue

            current_downloaded = os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0
            if total_size > 0 and current_downloaded < total_size:
                print(f"\n[WARNING] Incomplete download ({current_downloaded}/{total_size} bytes). Resuming...")
                time.sleep(3)
                continue

            # Full download complete
            if os.path.exists(tmp_file_path):
                os.rename(tmp_file_path, file_path)
                print(f"\n[SUCCESS] Completed download: {identifier}")
                return file_path

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
            print(f"\n[WARNING] Network timeout/drop ({net_err}). Retrying in 5 seconds (Attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(5)
            continue
        except Exception as gen_err:
            print(f"\n[ERROR] Unexpected error in download: {gen_err}")
            time.sleep(5)
            continue

    print(f"\n[ERROR] Failed to download {identifier} after {MAX_RETRIES} attempts.")
    return None


def refresh_access_token(refresh_token):
    data = {"refresh_token": refresh_token}

    try:
        response = requests.post(refresh_url, json=data)

        if response.status_code == 400:
            resp = response.json()
            err_msg = resp['error']
            print(f"\n[ERROR] Validation Error: {err_msg}\n")
            if generate_logs:
                logger.error(f"There was an Error encountered regarding the Validation of Refresh Token.\nError Details: {err_msg}\nSolution: Please make sure the Refresh Token is not Tampered or modified before passing in the download_data() method.")
            return

        response.raise_for_status() 
        return response.json() 
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Invalid Token. Please Login and Try Again.\nError Details: {e}")
        if generate_logs:
            logger.error("[ERROR] Invalid Token Error encountered. Please Login Successfully and Try Again.\nError Details: ", exc_info=True)
        return None

def logout():
    data = {"username":username} 
    retry_delays = [5, 10, 20, 30, 40, 50] # Total Retry Time: 155 Seconds (2.5 mins)

    for attempt, delay in enumerate(retry_delays):
        try:
            response = requests.post(logout_url, json=data, timeout=5)

            if response.status_code == 400:
                resp = response.json()
                err_msg = resp['error']
                print(f"\n[ERROR] Validation Error: {err_msg}\n")
                if generate_logs:
                    logger.error(f"There was an Error encountered regarding the Validation of 'username' in Logout.\nError Details: {err_msg}\nSolution: Please make sure you use the Correct Username associated with your MOSDAC account.")
                return

            response.raise_for_status()
            print(f"\nLogout Successful. {BOLD}Goodbye {username}!{RESET}\n")
            return

        except (requests.ConnectionError, requests.Timeout, OSError):
            if attempt < len(retry_delays) - 1:
                print(f"\n[WARNING] Network Error encountered during Logout. Please check your Internet Connection.")
                print(f"[INFO] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Logout Failed after Multiple Attempts due to Network Error. Please check your Internet connection and Try Again.\n")
                if generate_logs:
                    logger.error("Logout could not be successful even after Multiple Attempts due to the encountered Network Error. Please check your Internet and Try Again to successfully Terminate your session.")
                return

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error encountered in logout()", e)
            if generate_logs:
                logger.error(f"Error Encountered during Logout | Error Details: ", exc_info=True)
        

def main():

    total_files = search_results()  
    user_response = get_user_input()

    # Ending Script if User Response = No
    if user_response == 'n' or user_response == 'no':
        print(f"\n{GREEN}Download Cancelled.{RESET}")
        return
    else:
        print("\nVerifying User Credentials..")

    # Step 2: Login if Prompted for Download Data
    result = get_token()
    if result is None: 
        exit()
    else:
        tokens, username = result

    if tokens:
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token") 

        print(f"\n{GREEN}Login Successful.{RESET} {BOLD}Hello {username}!{RESET}")
    else:
        if generate_logs:
            logger.error("User could not be Authenticated, thus did not Proceed with Download.")
        print("User Authentication Failure, hence cannot Proceed with Download.")
        return

    start_time = time.time()

    download_complete, download_count, skip_count = fetch_and_download_data(total_files, access_token, refresh_token)
    
    end_time = time.time()

    if download_complete:
        print(f"\n{GREEN}Download Complete!{RESET}\n")
        total_time = end_time - start_time
        total_minutes = total_time / 60
        total_hours = total_time / 3600

        print(f"Total No. of Files Downloaded: {download_count}") 

        if (skip_count > 1):
            print(f"Files Skipped for Download: {skip_count}\n") 
        
        if total_hours >= 1:
            print(f"Total Time Taken: {total_hours:.2f} hr")
        elif total_minutes >= 1:
            print(f"Total Time Taken: {total_minutes:.2f} min")
        else:
            print(f"Total Time Taken: {total_time:.2f} sec")
    
    logout()

if __name__ == "__main__":
    main()
