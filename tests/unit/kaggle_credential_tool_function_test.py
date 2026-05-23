import httpx

def validate_kaggle_credentials(username: str, key: str) -> tuple[bool, str | None]:
        """
        Calls Kaggle API to verify username + API key combination.
        Uses the competitions list endpoint as a lightweight auth check.
        """
        if not username or not key:
            return False, "Both Kaggle username and API key are required."

        try:
            resp = httpx.get(
                "https://www.kaggle.com/api/v1/competitions/list",
                auth=(username, key),
                timeout=10,
            )
            if resp.status_code == 200:
                return True, None
            elif resp.status_code == 401:
                return False, (
                    "Kaggle credentials are invalid. "
                    "Get your API key at https://www.kaggle.com/settings → API → Create New Token"
                )
            else:
                return False, f"Kaggle API returned status {resp.status_code}."
        except httpx.TimeoutException:
            return False, "Kaggle API timed out. Check your internet connection."
        except Exception as e:
            return False, f"Could not reach Kaggle API: {e}"
        

valid,stri=validate_kaggle_credentials(username="faisalishfaq", key="602af2086b817cbd5bd52cdaf949b726")
print(valid)
print(stri)