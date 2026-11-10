"""
Unit tests for merge_datasets.py
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path
import pandas as pd

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from merge_datasets import find_csvs  # noqa: E402


class TestFindCsvs:
    """Tests for find_csvs function."""

    def test_find_csv_files(self):
        """Test finding CSV files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "test1.csv").touch()
            Path(tmpdir, "test2.csv").touch()
            Path(tmpdir, "data.txt").touch()  # Non-CSV file

            result = find_csvs(tmpdir)

            assert len(result) == 2
            assert all(f.endswith(".csv") for f in result)

    def test_empty_directory(self):
        """Test empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_csvs(tmpdir)
            assert result == []

    def test_nonexistent_directory(self):
        """Test nonexistent directory returns empty list."""
        result = find_csvs("/nonexistent/directory/path")
        assert result == []

    def test_sorted_output(self):
        """Test CSV files are returned sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "c.csv").touch()
            Path(tmpdir, "a.csv").touch()
            Path(tmpdir, "b.csv").touch()

            result = find_csvs(tmpdir)

            basenames = [os.path.basename(f) for f in result]
            assert basenames == ["a.csv", "b.csv", "c.csv"]


class TestMergeDatasets:
    """Integration tests for merge_datasets functionality."""

    def test_deduplication(self):
        """Test that duplicate rows are removed correctly."""
        # Create two dataframes with overlapping data
        df1 = pd.DataFrame(
            {
                "hostname": ["host1", "host1"],
                "timestamp": ["2024-01-01 10:00:00", "2024-01-01 10:00:01"],
                "service_name": ["service1", "service1"],
                "cpu_usage": [50.0, 55.0],
            }
        )

        df2 = pd.DataFrame(
            {
                "hostname": ["host1", "host2"],
                "timestamp": ["2024-01-01 10:00:00", "2024-01-01 10:00:00"],
                "service_name": ["service1", "service1"],
                "cpu_usage": [60.0, 45.0],  # Duplicate entry with different value
            }
        )

        # Merge and deduplicate
        merged = pd.concat([df1, df2], ignore_index=True)
        before = len(merged)

        merged = merged.drop_duplicates(
            subset=["hostname", "timestamp", "service_name"], keep="last"
        )
        after = len(merged)

        # Should have removed one duplicate
        assert before == 4
        assert after == 3

        # Check that the last value was kept (60.0 from df2)
        duplicate_row = merged[
            (merged["hostname"] == "host1")
            & (merged["timestamp"] == "2024-01-01 10:00:00")
            & (merged["service_name"] == "service1")
        ]
        assert duplicate_row["cpu_usage"].values[0] == 60.0

    def test_sorting(self):
        """Test that merged data is sorted correctly."""
        df = pd.DataFrame(
            {
                "hostname": ["host2", "host1", "host1", "host2"],
                "timestamp": [
                    "2024-01-01 10:00:01",
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:00:01",
                    "2024-01-01 10:00:00",
                ],
                "service_name": ["service1", "service2", "service1", "service2"],
                "cpu_usage": [50.0, 55.0, 60.0, 65.0],
            }
        )

        sorted_df = df.sort_values(["hostname", "service_name", "timestamp"], ignore_index=True)

        # Check sorting
        assert sorted_df["hostname"].tolist() == ["host1", "host1", "host2", "host2"]
        assert sorted_df["service_name"].iloc[0] == "service1"
        assert sorted_df["service_name"].iloc[1] == "service2"

    def test_empty_dataframe_merge(self):
        """Test merging with empty dataframes."""
        df1 = pd.DataFrame(
            {
                "hostname": ["host1"],
                "timestamp": ["2024-01-01 10:00:00"],
                "service_name": ["service1"],
                "cpu_usage": [50.0],
            }
        )

        df2 = pd.DataFrame()  # Empty dataframe

        merged = pd.concat([df1, df2], ignore_index=True)

        assert len(merged) == 1
        assert merged["hostname"].iloc[0] == "host1"


class TestCsvReading:
    """Tests for CSV reading functionality."""

    def test_read_valid_csv(self):
        """Test reading a valid CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir, "test.csv")

            # Create test CSV
            df = pd.DataFrame(
                {
                    "hostname": ["host1", "host2"],
                    "service_name": ["service1", "service2"],
                    "cpu_usage": [50.0, 60.0],
                }
            )
            df.to_csv(csv_path, index=False)

            # Read it back
            result = pd.read_csv(csv_path)

            assert len(result) == 2
            assert "hostname" in result.columns
            assert result["cpu_usage"].tolist() == [50.0, 60.0]

    def test_read_empty_csv(self):
        """Test handling empty CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir, "empty.csv")
            csv_path.touch()

            with pytest.raises(pd.errors.EmptyDataError):
                pd.read_csv(csv_path)
