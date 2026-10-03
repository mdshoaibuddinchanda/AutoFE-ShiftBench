import sqlite3
import pytest
from src.task_manifest import ManifestStore
from tests.test_critical_recovery import fixture


def test_sqlite_context_releases_connection_and_windows_handle(tmp_path):
    store,records = fixture(tmp_path)
    with store._connect() as connection:
        connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    moved = tmp_path/"moved.db"
    store.db_path.rename(moved)
    moved.unlink()


def test_readonly_missing_database_is_not_created(tmp_path):
    path=tmp_path/"absent"/"manifest.db"
    with pytest.raises(FileNotFoundError):
        ManifestStore(path,read_only=True)
    assert not path.parent.exists()


def test_readonly_existing_store_rejects_writes(tmp_path):
    store,records=fixture(tmp_path)
    reader=ManifestStore(store.db_path,read_only=True)
    assert len(reader.snapshot("r")["tasks"]) == 2
    with pytest.raises(sqlite3.OperationalError):
        reader.set_run_status("r","changed")
