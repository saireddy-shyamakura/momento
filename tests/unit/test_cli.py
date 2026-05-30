"""Unit tests for cli.py — command-line interface.

Tests argument parsing, threshold validation,
deprecation warnings, and config subcommands.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch


class TestParseArguments:
    """Argument parsing."""

    def test_default_arguments(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento"]):
            args = parse_arguments()
            assert args.dir is None
            assert args.version is False
            assert args.reset is False
            assert args.count is False
            assert args.verify is False

    def test_dir_argument(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento", "--dir", "/path/to/images"]):
            args = parse_arguments()
            assert args.dir == "/path/to/images"

    def test_version_flag(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento", "--version"]):
            args = parse_arguments()
            assert args.version is True

    def test_log_format(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento", "--log-format", "json"]):
            args = parse_arguments()
            assert args.log_format == "json"

    def test_v3_flags(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento", "--rerank", "--no-hybrid", "--no-query-expansion"]):
            args = parse_arguments()
            assert args.rerank is True
            assert args.no_hybrid is True
            assert args.no_query_expansion is True

    def test_feature_toggles(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento", "--no-multi-embed", "--no-video", "--no-yolo", "--no-ocr"]):
            args = parse_arguments()
            assert args.no_multi_embed is True
            assert args.no_video is True
            assert args.no_yolo is True
            assert args.no_ocr is True

    def test_dry_run_flag(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento", "--dry-run"]):
            args = parse_arguments()
            assert args.dry_run is True

    def test_subcommand_config(self):
        from momento.cli import parse_arguments
        with patch.object(sys, "argv", ["momento", "config", "show"]):
            args = parse_arguments()
            assert args.command == "config"
            assert args.config_action == "show"


class TestValidateThreshold:
    """Threshold validation."""

    def test_valid_threshold(self):
        from momento.cli import validate_threshold
        assert validate_threshold(0.5) is True
        assert validate_threshold(0.0) is True
        assert validate_threshold(1.0) is True

    def test_invalid_threshold_low(self):
        from momento.cli import validate_threshold
        assert validate_threshold(-0.1) is False

    def test_invalid_threshold_high(self):
        from momento.cli import validate_threshold
        assert validate_threshold(1.1) is False


class TestShowDeprecationWarnings:
    """Deprecation warning display."""

    def test_no_deprecated_flags(self):
        from momento.cli import show_deprecation_warnings
        args = MagicMock()
        args.multi_embed = False
        args.include_video = False
        args.yolo = False
        args.ocr = False
        args.all_features = False
        show_deprecation_warnings(args)

    def test_with_deprecated_flag(self):
        from momento.cli import show_deprecation_warnings
        args = MagicMock()
        args.multi_embed = True
        args.include_video = False
        args.yolo = False
        args.ocr = False
        args.all_features = False
        show_deprecation_warnings(args)


class TestMain:
    """Main entry point."""

    def test_main_without_lock_commands(self):
        from momento.cli import main
        with patch("momento.cli.run_cli") as mock_run, \
             patch.object(sys, "argv", ["momento", "doctor"]), \
             patch("momento.cli.set_log_format"):
            main()
            mock_run.assert_called_once()

    def test_main_with_lock(self):
        from momento.cli import main
        with patch("momento.cli.run_cli") as mock_run, \
             patch.object(sys, "argv", ["momento", "--dir", "/tmp"]), \
             patch("momento.cli.set_log_format"), \
             patch("momento.cli.LockFile") as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock_instance.acquire.return_value = True
            mock_lock.return_value = mock_lock_instance
            main()
            mock_run.assert_called_once()