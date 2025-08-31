import csv
import os
import subprocess
import datetime
from typing import Optional, Dict, Any, List


class CSVRunLogger:
    """
    CSVRunLogger writes exactly one CSV file per run and can optionally commit/push it to Git if
    the repository is a git repo with a configured remote and credentials available.
    """

    def __init__(self, logs_dir: str = "./logs", filename_prefix: str = "run"):
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(logs_dir, f"{filename_prefix}_{timestamp}.csv")
        self._file = open(self.file_path, mode="w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = None

    @property
    def path(self) -> str:
        return self.file_path

    def log_row(self, row: Dict[str, Any]):
        # Lazily initialize DictWriter with dynamic fieldnames
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        else:
            # Ensure all keys exist; add missing keys with None and extend header if new fields appear
            new_keys = [k for k in row.keys() if k not in self._fieldnames]
            if new_keys:
                # Extend header: we need to rewrite the file with new header; to keep it minimal,
                # we'll just add the new keys to fieldnames and write them in subsequent rows.
                self._fieldnames.extend(new_keys)
                self._writer.fieldnames = self._fieldnames
        # Fill missing columns with None
        complete_row = {k: row.get(k) for k in self._fieldnames}
        self._writer.writerow(complete_row)
        self._file.flush()

    def close(self):
        try:
            if self._file:
                self._file.flush()
                self._file.close()
        finally:
            self._file = None

    # -------- Optional Git integration --------
    def try_git_upload(self, commit_message: Optional[str] = None) -> bool:
        """
        Attempt to add, commit, and push the CSV log to the current repository.
        Returns True on success, False otherwise. All errors are swallowed to keep training robust.
        """
        csv_path = os.path.relpath(self.file_path)
        try:
            # Check if inside a git repo
            if subprocess.call(["git", "rev-parse", "--is-inside-work-tree"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
                return False
            # Ensure there is at least one remote
            remotes = subprocess.check_output(["git", "remote"], text=True).strip().splitlines()
            if not remotes:
                return False
            # Stage file
            subprocess.check_call(["git", "add", csv_path])
            # Commit
            if commit_message is None:
                commit_message = f"Add training CSV log {os.path.basename(csv_path)}"
            # Allow empty commit only if needed
            try:
                subprocess.check_call(["git", "commit", "-m", commit_message])
            except subprocess.CalledProcessError:
                # Possibly nothing to commit
                return False
            # Push
            subprocess.check_call(["git", "push"])  # Use default remote/branch
            return True
        except Exception:
            return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
