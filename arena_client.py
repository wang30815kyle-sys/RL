#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Arena Client - Refactored for Unified Framework.

This client supports:
1. Automatic authentication via ~/.mlarena/credentials.json.
2. Unified evaluate() method for both Sync and WS modes.
3. Direct object-based evaluation for WS mode (supporting NumPy/PyTorch objects).
4. Robust error handling for common student pitfalls.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import getpass
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Optional: nest_asyncio for Jupyter/IPython/Spyder compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
    _NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    _NEST_ASYNCIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Early Dependency Check
# ---------------------------------------------------------------------------
missing_deps = []
try: import requests
except ImportError: missing_deps.append("requests")
try: import websockets
except ImportError: missing_deps.append("websockets")
try: import pandas as pd
except ImportError: missing_deps.append("pandas")
try: import joblib
except ImportError: missing_deps.append("joblib")

if missing_deps:
    print("-" * 50)
    print("Environment Error: Missing required libraries.")
    print(f"Please install them using: pip install {' '.join(missing_deps)}")
    print("-" * 50)
    sys.exit(1)

# Add current directory to path to ensure mlarena subpackage is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from mlarena.base import ArenaPredictor
except ImportError:
    # Fallback for direct usage without subpackage
    class ArenaPredictor: pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_URL: str = os.getenv("MLARENA_URL", "https://api-mlarena.spkuan.cc")
DEFAULT_WS_URL: str = os.getenv("MLARENA_WS_URL", "wss://api-mlarena.spkuan.cc")
CREDENTIALS_PATH: Path = Path.home() / ".mlarena" / "credentials.json"


class MLArenaClient:
    """A refined client for the ML Arena platform."""

    def __init__(
        self,
        server_url: str = DEFAULT_URL,
        ws_url: str = DEFAULT_WS_URL,
        api_key: Optional[str] = None,
        student_id: Optional[str] = None
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self.api_key = api_key
        self.username = None
        self._student_id_hint = student_id
        
        # Auto-load cached credentials if api_key not provided
        if not self.api_key:
            cached = self._load_cached_credentials()
            if cached:
                cached_username = cached.get("username")
                
                # Check for mismatch if student_id is provided
                if student_id and cached_username and cached_username != student_id:
                    print(f"Account mismatch: Cache is '{cached_username}', but script requested '{student_id}'.")
                    print(f"   Clearing stale cache to allow re-authentication.")
                else:
                    self.api_key = cached["api_key"]
                    self.username = cached_username
                    if self.username:
                        print(f"Authenticated as '{self.username}' via cache.")

    def _load_cached_credentials(self) -> Optional[Dict[str, str]]:
        if CREDENTIALS_PATH.exists():
            try:
                with CREDENTIALS_PATH.open() as fh:
                    return json.load(fh)
            except: pass
        return None

    def _print_result(self, result: Dict[str, Any]) -> None:
        """Helper to pretty-print evaluation result."""
        print("-" * 40)
        print("Evaluation Result Summary")
        print("-" * 40)
        if result.get('error'):
            print(f"ERROR:              {result.get('error')}")
            return # Don't print stats if there's an error
            
        print(f"Status:             {result.get('message', 'Complete')}")
        print(f"Score:              {result.get('score')}")
        print(f"Inference Time:     {result.get('inference_time')}s")
        
        if "stamina_remaining" in result:
            print(f"Stamina Remaining:  {result.get('stamina_remaining')}")
            
        achievements = result.get('unlocked_achievements', [])
        if achievements:
            print(f"Achievements:       {', '.join(achievements)}")
            
        tasks = result.get('unlocked_tasks', [])
        if tasks:
            print(f"Tasks Completed:    {', '.join(tasks)}")
        print("-" * 40)

    def _save_credentials(self, data: Dict[str, Any]) -> None:
        CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CREDENTIALS_PATH.open("w") as fh:
            json.dump({
                "api_key": data["api_key"],
                "username": data["username"],
                "nickname": data.get("nickname", data["username"])
            }, fh)

    def logout(self) -> None:
        """Clears cached credentials."""
        if CREDENTIALS_PATH.exists():
            path_str = str(CREDENTIALS_PATH)
            CREDENTIALS_PATH.unlink()
            print(f"Credentials cleared successfully.")
            print(f"Removed: {path_str}")
        else:
            print(f"No cached credentials found at {CREDENTIALS_PATH}")

    def _reauthenticate(self) -> None:
        """Clear stale credentials and re-authenticate via enroll().

        Works for new students (201), already-claimed students and admin
        accounts (409 → password prompt → login()), and raises on 403.
        """
        print("[Credentials Expired] 本地憑證已過期或無效。")
        print("   正在自動清除快取並引導重新認證...")
        self.logout()
        if self._student_id_hint:
            self.enroll(self._student_id_hint)
        else:
            print("請手動重新執行 run.py 並確認已填寫學號。")
            sys.exit(1)

    def _api_request(
        self,
        method: str,
        url: str,
        *,
        authenticated: bool = True,
        **kwargs,
    ) -> "requests.Response":
        """Central HTTP wrapper: injects X-API-Key and retries once on 401.

        Args:
            method: HTTP verb ("GET", "POST", etc.).
            url: Full URL.
            authenticated: If True, injects X-API-Key and retries on 401.
                           Set False for public endpoints.
            **kwargs: Forwarded to requests.request() (timeout, data, json, etc.).

        Raises:
            RuntimeError: If re-authentication fails (e.g. student_id not whitelisted).
            SystemExit: On connection error or when no student_id hint is available.
        """
        if authenticated and self.api_key is None:
            self._reauthenticate()

        if authenticated:
            kwargs["headers"] = {**kwargs.get("headers", {}), "X-API-Key": self.api_key}

        try:
            resp = requests.request(method, url, **kwargs)
        except requests.exceptions.ConnectionError:
            print("[Connection Error] 無法連線至伺服器。請確認是否已連上網路/VPN。")
            sys.exit(1)

        if authenticated and resp.status_code == 401:
            self._reauthenticate()
            kwargs["headers"] = {**kwargs.get("headers", {}), "X-API-Key": self.api_key}
            try:
                resp = requests.request(method, url, **kwargs)
            except requests.exceptions.ConnectionError:
                print("[Connection Error] 重新認證後無法連線。")
                sys.exit(1)
            if resp.status_code == 401:
                print("[Auth Error] 重新認證後仍收到 401，請確認帳號狀態。")

        return resp

    def enroll(self, student_id: str, nickname: Optional[str] = None) -> str:
        """Enrolls and saves credentials. Falls back to login if already claimed."""
        if not student_id or student_id.strip() == "" or student_id.strip() == "YOUR_ID":
            print("[Validation Error] Student ID is missing or invalid.")
            print("   Please fill in your actual Student ID in your run script (e.g., STUDENT_ID = '11223344').")
            sys.exit(1)

        try:
            url = f"{self.server_url}/api/enroll/claim"
            resp = requests.post(url, json={"student_id": student_id, "nickname": nickname}, timeout=15)
        except requests.exceptions.ConnectionError:
            print("[Connection Error] 無法連線至伺服器。請確認是否已連上網路/VPN，或伺服器是否離線。")
            sys.exit(1)
        
        if resp.status_code == 201:
            data = resp.json()
            self.api_key = data["api_key"]
            self._save_credentials(data)
            print(f"Enrollment successful: {data['username']}")
            return self.api_key
        elif resp.status_code == 409:
            # Smart Fallback: If already claimed, try to log in and recover the key
            print(f"帳號 '{student_id}' 已經領取過 (Already claimed).")
            print(f"   正在嘗試透過登入回復憑證 (Attempting to restore credentials via login)...")
            
            pwd = getpass.getpass(f"請輸入學號 '{student_id}' 的密碼 (Enter password): ")
            try:
                return self.login(student_id, pwd)
            except Exception as e:
                raise RuntimeError(f"Authentication failed during recovery: {e}")
        elif resp.status_code == 403:
            raise RuntimeError(f"Enrollment forbidden: Student ID '{student_id}' is not in the whitelist.")
        else:
            raise RuntimeError(f"Enrollment failed ({resp.status_code}): {resp.text}")

    def login(self, username: str, password: str, mfa_code: Optional[str] = None) -> str:
        """Authenticates via password (and optional MFA) and recovers API key."""
        print(f"Logging in as '{username}'...")

        try:
            # 1. Get JWT Token
            login_url = f"{self.server_url}/api/auth/login"
            payload = {"username": username, "password": password}
            if mfa_code:
                payload["client_secret"] = mfa_code
            resp = requests.post(login_url, data=payload, timeout=15)

            if resp.status_code != 200:
                if resp.status_code == 401:
                    detail = ""
                    try:
                        detail = resp.json().get("detail", "")
                    except Exception:
                        pass
                    if "mfa" in detail.lower() or "code required" in detail.lower():
                        mfa = getpass.getpass("請輸入 MFA 驗證碼 (Authenticator App code): ")
                        return self.login(username, password, mfa_code=mfa)
                    raise RuntimeError("Invalid credentials (密碼錯誤).")
                raise RuntimeError(f"Login failed ({resp.status_code}): {resp.text}")
                
            token = resp.json()["access_token"]
            
            # 2. Get Profile (containing API Key)
            me_url = f"{self.server_url}/api/users/me"
            headers = {"Authorization": f"Bearer {token}"}
            me_resp = requests.get(me_url, headers=headers, timeout=15)
            
            if me_resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch profile: {me_resp.text}")
                
            data = me_resp.json()
            self.api_key = data["api_key"]
            
            if not self.api_key:
                raise RuntimeError("Account found, but no API Key is assigned. Please contact admin.")
                
            # 3. Save to local cache
            self._save_credentials(data)
            print(f"Credentials restored. API Key cached for '{username}'.")
            return self.api_key
        except requests.exceptions.ConnectionError:
            print("[Connection Error] 無法連線至伺服器。")
            sys.exit(1)

    def _get_active_competitions(self) -> Optional[str]:
        """Fetches a list of active competition IDs for error guidance."""
        try:
            url = f"{self.server_url}/api/tavern/competitions/active"
            resp = self._api_request("GET", url, authenticated=False, timeout=10)
            if resp.status_code == 200:
                comps = resp.json()
                active = [f"ID {c['id']}: {c['title']}" for c in comps if c.get("is_active")]
                if active:
                    return "\n".join(active)
            return None
        except Exception:
            return None

    def _handle_competition_not_found(self, competition_id: int, server_error: Optional[str] = None) -> None:
        """Helper to show pretty warnings for invalid competition IDs."""
        if server_error:
            print(f"[Warning] Server error: {server_error}")
        else:
            print(f"[Warning] Competition ID {competition_id} was not found.")
            
        active_list = self._get_active_competitions()
        if active_list:
            print("Available Active Competitions:")
            print("-" * 40)
            print(active_list)
            print("-" * 40)
        print("Please check COMPETITION_ID in your run.py script.")

    def _run_async(self, coro):
        """
        Smart async execution that works in both normal scripts and interactive environments.

        This method handles three scenarios:
        1. Jupyter/IPython/Spyder with nest_asyncio installed → runs in existing loop
        2. No running loop (normal script) → uses asyncio.run()
        3. Running loop without nest_asyncio → shows helpful error message
        """
        try:
            # Try to get the running loop
            loop = asyncio.get_running_loop()

            # We have a running loop - this is an interactive environment
            if not _NEST_ASYNCIO_AVAILABLE:
                print("-" * 50)
                print("[Environment Notice] 偵測到互動式 Python 環境 (Jupyter/Spyder)")
                print("為了在此環境中正常執行，請安裝 nest_asyncio:")
                print("   pip install nest_asyncio")
                print("-" * 50)
                raise RuntimeError(
                    "Cannot run async code in interactive environment without nest_asyncio. "
                    "Please install: pip install nest_asyncio"
                )

            # nest_asyncio is available and already applied, just run the coroutine
            return loop.run_until_complete(coro)

        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No running loop - normal script execution
                return asyncio.run(coro)
            else:
                # Re-raise other RuntimeErrors (like the nest_asyncio message)
                raise

    def evaluate(
        self,
        competition_id: int,
        predictor: Union[ArenaPredictor, str],
        mode: str = "ws",
        _retried: bool = False,
    ) -> Dict[str, Any]:
        """Unified entry point for evaluation."""
        if not self.api_key:
            # Try once more to see if student_id helps
            if self._student_id_hint:
                print("Missing credentials. Triggering automatic enrollment...")
                self.enroll(self._student_id_hint)
            else:
                raise RuntimeError("Not authenticated. Please enroll() first or provide an API key.")

        try:
            if mode == "sync":
                if not isinstance(predictor, str):
                    raise ValueError("In 'sync' mode, 'predictor' must be a file path string.")
                return self._submit_sync(competition_id, predictor)
            elif mode == "ws":
                return self._run_async(self._run_ws_worker(competition_id, predictor))
            else:
                raise ValueError(f"Invalid mode: {mode}")
        except RuntimeError as e:
            if not _retried and ("not authenticated" in str(e).lower() or "unauthorized" in str(e).lower() or "invalid api key" in str(e).lower()):
                self._reauthenticate()
                return self.evaluate(competition_id, predictor, mode, _retried=True)
            raise e

    def _submit_sync(self, competition_id: int, model_path: str) -> Dict[str, Any]:
        """Submits model via HTTP (chunked)."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
            
        file_size = os.path.getsize(model_path)
        filename = os.path.basename(model_path)

        print(f"Initializing upload for Competition {competition_id}...")
        init_resp = self._api_request(
            "POST",
            f"{self.server_url}/api/submissions/upload/init",
            params={"competition_id": competition_id},
            json={"filename": filename},
            timeout=30,
        )

        if init_resp.status_code != 200:
            if init_resp.status_code == 404:
                self._handle_competition_not_found(competition_id)
                sys.exit(0)
            
            # Check for specific error message in body
            try:
                err_msg = init_resp.json().get("detail", init_resp.text)
                err_lower = err_msg.lower()
                if "not found" in err_lower or "dataset" in err_lower:
                    self._handle_competition_not_found(competition_id, server_error=err_msg)
                    sys.exit(0)
                elif "stamina" in err_lower:
                    print(f"[Warning] {err_msg}")
                    print("   體力值不足，請稍後再試，或聯絡管理員確認。")
                    sys.exit(0)
                elif "unauthorized" in err_lower or "api key" in err_lower:
                    raise RuntimeError("Not authenticated")
            except: pass
                
            raise RuntimeError(f"Init failed: {init_resp.text}")
        
        upload_id = init_resp.json()["upload_id"]
        chunk_size = 50 * 1024 * 1024
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        chunk_headers = {"X-API-Key": self.api_key}

        with open(model_path, "rb") as f:
            for i in range(total_chunks):
                print(f"   Chunk {i+1}/{total_chunks}...", end="\r")
                chunk_data = f.read(chunk_size)
                chunk_resp = requests.post(
                    f"{self.server_url}/api/submissions/upload/chunk",
                    headers=chunk_headers,
                    data={"upload_id": upload_id, "chunk_index": i},
                    files={"file": (f"c{i}", chunk_data)},
                    timeout=60
                )
                if chunk_resp.status_code == 401:
                    self._reauthenticate()
                    chunk_headers = {"X-API-Key": self.api_key}
                    chunk_resp = requests.post(
                        f"{self.server_url}/api/submissions/upload/chunk",
                        headers=chunk_headers,
                        data={"upload_id": upload_id, "chunk_index": i},
                        files={"file": (f"c{i}", chunk_data)},
                        timeout=60
                    )
                if chunk_resp.status_code != 200:
                    raise RuntimeError(f"Chunk {i} upload failed ({chunk_resp.status_code}): {chunk_resp.text}")
        
        print("\nUpload complete. Queuing for evaluation...")
        complete_resp = self._api_request(
            "POST",
            f"{self.server_url}/api/submissions/upload/complete",
            json={"upload_id": upload_id},
            timeout=300,
        )

        if complete_resp.status_code != 200:
            raise RuntimeError(f"Failed to finalize upload: {complete_resp.text}")

        result_data = complete_resp.json()
        submission_id = result_data.get("submission_id")

        if not submission_id:
            return result_data

        print(f"Submission {submission_id} queued. Waiting for evaluation...")

        # Poll for status
        status = "pending"
        status_data = {}
        while True:
            status_resp = self._api_request(
                "GET",
                f"{self.server_url}/api/submissions/{submission_id}/status",
                timeout=10,
            )

            if status_resp.status_code != 200:
                print(f"Warning: Failed to check status ({status_resp.status_code})")
                time.sleep(5)
                continue

            status_data = status_resp.json()
            status = status_data.get("status")

            if status in ["completed", "failed"]:
                break
            
            print(f"   Status: {status.capitalize()}...", end="\r")
            time.sleep(2)

        if status == "completed":
            self._print_result(status_data)
        elif status == "failed":
            error_msg = status_data.get("error_message", "Unknown error")
            print(f"\nEvaluation Failed: {error_msg}")
            raise RuntimeError(f"Evaluation failed: {error_msg}")

        return status_data

    async def _run_ws_worker(self, competition_id: int, predictor: Union[ArenaPredictor, str]) -> Dict[str, Any]:
        """Runs evaluation over WebSocket."""
        # Load model if path is provided
        if isinstance(predictor, str):
            if not os.path.exists(predictor):
                 raise FileNotFoundError(f"Model file {predictor} not found.")
            print(f"Loading model from {predictor}...")
            try:
                predictor = joblib.load(predictor)
            except Exception as e:
                print("-" * 50)
                print(f"[Model Load Error] 無法讀取模型檔案: {e}")
                print("   這通常是因為 Python 版本或 sklearn/joblib 版本不一致。")
                print("   請確保訓練與執行環境的套件版本相同。")
                print("-" * 50)
                return {"status": "failed", "error": "Model load fail"}

        ws_endpoint = f"{self.ws_url}/api/ws/execution/{competition_id}?api_key={self.api_key}"
        print(f"Connected to Arena (WS) for Competition {competition_id}...")

        try:
            async with websockets.connect(ws_endpoint, ping_interval=20, ping_timeout=20) as ws:
                # 1. Init
                greeting = json.loads(await ws.recv())
                if "error" in greeting:
                    err_msg = greeting["error"]
                    err_lower = err_msg.lower()
                    if "not found" in err_lower or "dataset" in err_lower:
                        self._handle_competition_not_found(competition_id, server_error=err_msg)
                        return {"status": "failed", "error": err_msg}
                    elif "stamina" in err_lower:
                        print(f"[Warning] {err_msg}")
                        print("   體力值不足，請稍後再試，或聯絡管理員確認。")
                        return {"status": "failed", "error": err_msg}
                    elif "api key" in err_lower or "unauthorized" in err_lower:
                        raise RuntimeError("Unauthorized")
                    raise RuntimeError(err_msg)
                
                total_samples = greeting["total_samples"]
                print(f"Evaluating {total_samples} samples...")

                # 2. Receive Features
                features = []
                while len(features) < total_samples:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        msg_data = json.loads(msg)
                        features.extend(msg_data["data"])
                        print(f"   Received {len(features)}/{total_samples}...", end="\r")
                        if msg_data.get("is_last"): break
                    except asyncio.TimeoutError:
                        print("\n[Network Error] Receive timeout. Closing connection.")
                        return {"status": "failed", "error": "timeout"}

                # 3. Predict
                print(f"\nRunning inference on {type(predictor).__name__}...")
                df = pd.DataFrame(features)
                try:
                    predictions = predictor.predict(df)
                except Exception as e:
                    print(f"\n[Usage Error] Predict function failed: {e}")
                    print("   請檢查您的模型程式碼是否正確讀取 DataFrame 格式。")
                    return {"status": "failed", "error": "predict_fail"}
                
                # Local Validation before sending
                if len(predictions) != total_samples:
                    print(f"\n[Validation Error] 預測數量不符。預期: {total_samples}, 實際: {len(predictions)}")
                    print("   請確認模型是否有正確處理所有輸入樣本。")
                    return {"status": "failed", "error": "count_mismatch"}

                if not isinstance(predictions, list):
                    if hasattr(predictions, "tolist"):
                        predictions = predictions.tolist()
                    else:
                        predictions = list(predictions)

                # 4. Send Predictions
                batch_size = 5000
                for i in range(0, len(predictions), batch_size):
                    chunk = predictions[i : i + batch_size]
                    is_last = (i + batch_size) >= len(predictions)
                    await ws.send(json.dumps({
                        "type": "predictions_chunk",
                        "data": chunk,
                        "is_last": is_last
                    }))

                # 5. Final Result
                result = json.loads(await ws.recv())
                if "error" in result:
                    print(f"\n[Server Error] {result['error']}")
                    return result
                    
                self._print_result(result)
                return result
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 404:
                self._handle_competition_not_found(competition_id)
            elif e.status_code == 401:
                raise RuntimeError("Unauthorized")
            else:
                print(f"WebSocket connection failed: {e}")
            return {"status": "failed", "error": f"Invalid Competition ID: {competition_id}"}
        except Exception as e:
            if "unauthorized" in str(e).lower():
                raise e
            print(f"Evaluation error: {e}")
            return {"status": "failed", "error": str(e)}


    # Weights files larger than this threshold are uploaded in chunks to work
    # around Cloudflare's 100 MB per-request limit.
    _RL_CHUNK_SIZE = 45 * 1024 * 1024  # 45 MB

    def _upload_weights_chunked(self, competition_id: int, weights_path: str) -> str:
        """Upload a large weights file in chunks. Returns the upload_id."""
        chunk_headers = {"X-API-Key": self.api_key}
        filename = os.path.basename(weights_path)
        file_size = os.path.getsize(weights_path)
        total_chunks = (file_size + self._RL_CHUNK_SIZE - 1) // self._RL_CHUNK_SIZE

        start_resp = self._api_request(
            "POST",
            f"{self.server_url}/api/rl/uploads/weights/start",
            data={"competition_id": str(competition_id), "filename": filename},
            timeout=15,
        )
        if start_resp.status_code != 200:
            raise RuntimeError(f"Failed to start weights upload: {start_resp.text}")

        upload_id = start_resp.json()["upload_id"]

        with open(weights_path, "rb") as f:
            for i in range(total_chunks):
                chunk_data = f.read(self._RL_CHUNK_SIZE)
                pct = int((i + 1) / total_chunks * 100)
                print(f"   Uploading weights: chunk {i + 1}/{total_chunks} ({pct}%)...", end="\r")
                chunk_resp = requests.post(
                    f"{self.server_url}/api/rl/uploads/weights/chunk",
                    headers=chunk_headers,
                    data={"upload_id": upload_id, "chunk_index": str(i)},
                    files={"file": (f"chunk_{i}", chunk_data, "application/octet-stream")},
                    timeout=120,
                )
                if chunk_resp.status_code == 401:
                    self._reauthenticate()
                    chunk_headers = {"X-API-Key": self.api_key}
                    chunk_resp = requests.post(
                        f"{self.server_url}/api/rl/uploads/weights/chunk",
                        headers=chunk_headers,
                        data={"upload_id": upload_id, "chunk_index": str(i)},
                        files={"file": (f"chunk_{i}", chunk_data, "application/octet-stream")},
                        timeout=120,
                    )
                if chunk_resp.status_code != 200:
                    raise RuntimeError(f"Chunk {i} upload failed: {chunk_resp.text}")

        print()  # newline after progress
        return upload_id

    def upload_rl_slot(
        self,
        competition_id: int,
        slot_index: int,
        agent_file: Optional[str] = None,
        model_file: Optional[str] = None,
        weights_file: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        _retry: bool = True,
    ) -> Dict[str, Any]:
        """Upload an RL agent slot to the platform.

        Args:
            agent_file:   Path to agent.py (required for new slots).
            model_file:   Path to model.py (optional). When provided, the server
                          inlines it into agent.py so the sandbox has a
                          self-contained file — avoids ImportError from
                          'from model import ...' at runtime.
            weights_file: Path to model weights (optional). Files larger than
                          45 MB are chunked automatically.
            name:         Display name on leaderboard.
            description:  Optional description shown on the slot card.
        """
        if not self.api_key:
            raise RuntimeError("Not authenticated. Run enroll() or login() first.")
        if agent_file and not os.path.exists(agent_file):
            raise FileNotFoundError(f"Agent file not found: {agent_file}")
        if model_file and not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if weights_file and not os.path.exists(weights_file):
            raise FileNotFoundError(f"Weights file not found: {weights_file}")

        url = f"{self.server_url}/api/rl/competitions/{competition_id}/slots"
        headers = {"X-API-Key": self.api_key}

        slot_name = name or (
            os.path.splitext(os.path.basename(agent_file))[0] if agent_file else f"slot_{slot_index}"
        )
        data: Dict[str, Any] = {"slot_index": str(slot_index), "name": slot_name}
        if description:
            data["description"] = description

        # Decide whether to chunk the weights file
        weights_upload_id: Optional[str] = None
        if weights_file:
            size = os.path.getsize(weights_file)
            if size > self._RL_CHUNK_SIZE:
                mb = size / (1024 * 1024)
                print(f"Large weights file ({mb:.1f} MB) — using chunked upload...")
                weights_upload_id = self._upload_weights_chunked(competition_id, weights_file)
                data["weights_upload_id"] = weights_upload_id

        files: Dict[str, Any] = {}
        agent_ctx  = open(agent_file,  "rb") if agent_file  else None
        model_ctx  = open(model_file,  "rb") if model_file  else None
        weights_ctx = open(weights_file, "rb") if (weights_file and not weights_upload_id) else None
        try:
            if agent_ctx:
                files["agent_file"] = (os.path.basename(agent_file), agent_ctx, "text/plain")
            if model_ctx:
                files["model_file"] = (os.path.basename(model_file), model_ctx, "text/plain")
            if weights_ctx:
                files["weights_file"] = (os.path.basename(weights_file), weights_ctx, "application/octet-stream")

            try:
                resp = requests.request(
                    "POST", url,
                    headers={"X-API-Key": self.api_key},
                    data=data,
                    files=files or None,
                    timeout=120,
                )
            except requests.exceptions.ConnectionError:
                print("[Connection Error] 無法連線至伺服器。請確認 SERVER_URL 是否正確，以及是否已連上網路/VPN。")
                sys.exit(1)
        finally:
            if agent_ctx:
                agent_ctx.close()
            if model_ctx:
                model_ctx.close()
            if weights_ctx:
                weights_ctx.close()

        if resp.status_code in (200, 201):
            result = resp.json()
            print(f"Slot {slot_index} uploaded successfully.")
            print(f"  Name: {result.get('name', '-')}")
            print(f"  Elo:  {result.get('elo_rating', 1000):.0f}")
            return result
        elif resp.status_code == 401:
            if not _retry:
                raise RuntimeError("Unauthorized — check your API key.")
            self._reauthenticate()
            return self.upload_rl_slot(
                competition_id, slot_index,
                agent_file, model_file, weights_file,
                name, description,
                _retry=False,
            )
        elif resp.status_code == 400:
            raise RuntimeError(f"Upload rejected: {resp.json().get('detail', resp.text)}")
        else:
            raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text}")

    def list_rl_slots(self, competition_id: int) -> None:
        """Print all RL slots for the current user in a competition."""
        url = f"{self.server_url}/api/rl/competitions/{competition_id}/slots"
        resp = self._api_request("GET", url, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to list slots ({resp.status_code}): {resp.text}")
        slots = resp.json()
        if not slots:
            print(f"No slots uploaded for competition {competition_id}.")
            return
        print(f"{'Slot':>4}  {'Name':<20}  {'Elo':>6}  {'Games':>5}  {'Agent':>10}  Weights")
        print("-" * 70)
        for s in slots:
            print(
                f"{s['slot_index']:>4}  {(s.get('name') or '-'):<20}  "
                f"{s.get('elo_rating', 0):>6.0f}  {s.get('elo_games_played', 0):>5}  "
                f"{'yes' if s.get('agent_file_path') else 'no':>10}  "
                f"{s.get('weights_filename') or '-'}"
            )


def run_arena(
    student_id: str,
    model_path: str,
    competition_id: int,
    mode: str = "sync",
    nickname: Optional[str] = None
) -> None:
    """One-click helper for students."""
    # Basic validation before starting
    if not student_id or student_id.strip() == "" or student_id.strip() == "YOUR_ID":
        print("[Validation Error] 學號 (Student ID) 尚未填寫或仍為預設值。")
        print("   請修改 run.py 中的 STUDENT_ID 變數，填入您的真實學號。")
        return

    client = MLArenaClient(student_id=student_id)
    try:
        if not client.api_key:
            client.enroll(student_id, nickname)
        
        client.evaluate(competition_id, model_path, mode=mode)
    except Exception as e:
        print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Arena Client")
    subparsers = parser.add_subparsers(dest="command")

    # --- evaluate (default) ---
    eval_p = subparsers.add_parser("evaluate", help="Submit a classic ML model for evaluation")
    eval_p.add_argument("--mode", choices=["sync", "ws"], default="ws")
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--competition-id", type=int, required=True)
    eval_p.add_argument("--student-id", default=None)
    eval_p.add_argument("--nickname", default=None)

    # --- upload-slot ---
    slot_p = subparsers.add_parser("upload-slot", help="Upload an RL agent slot")
    slot_p.add_argument("--competition-id", type=int, required=True)
    slot_p.add_argument("--slot-index", type=int, required=True, help="0-based slot index")
    slot_p.add_argument("--agent-file", required=True, help="Path to agent.py")
    slot_p.add_argument("--model-file", default=None, help="Path to model.py (inlined into agent.py server-side)")
    slot_p.add_argument("--weights-file", default=None, help="Path to weights file (optional)")
    slot_p.add_argument("--name", default=None, help="Slot display name")
    slot_p.add_argument("--description", default=None, help="Optional slot description")
    slot_p.add_argument("--student-id", default=None)

    # --- list-slots ---
    list_p = subparsers.add_parser("list-slots", help="List your RL slots for a competition")
    list_p.add_argument("--competition-id", type=int, required=True)
    list_p.add_argument("--student-id", default=None)

    # --- logout ---
    subparsers.add_parser("logout", help="Clear cached credentials")

    # Legacy flat args for backwards compatibility
    parser.add_argument("--mode", choices=["sync", "ws"], default="ws", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--competition-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--student-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--nickname", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--logout", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Handle legacy flat invocation
    if args.command is None:
        if args.logout:
            MLArenaClient().logout()
            return
        if args.model and args.competition_id:
            run_arena(args.student_id, args.model, args.competition_id, args.mode, args.nickname)
            return
        parser.print_help()
        return

    if args.command == "logout":
        MLArenaClient().logout()

    elif args.command == "evaluate":
        run_arena(args.student_id, args.model, args.competition_id, args.mode, args.nickname)

    elif args.command == "upload-slot":
        client = MLArenaClient(student_id=args.student_id)
        if not client.api_key:
            if not args.student_id:
                parser.error("Not authenticated. Provide --student-id to auto-enroll.")
            client.enroll(args.student_id)
        client.upload_rl_slot(
            competition_id=args.competition_id,
            slot_index=args.slot_index,
            agent_file=args.agent_file,
            model_file=args.model_file,
            weights_file=args.weights_file,
            name=args.name,
            description=args.description,
        )

    elif args.command == "list-slots":
        client = MLArenaClient(student_id=args.student_id)
        if not client.api_key:
            parser.error("Not authenticated.")
        client.list_rl_slots(args.competition_id)


if __name__ == "__main__":
    main()
