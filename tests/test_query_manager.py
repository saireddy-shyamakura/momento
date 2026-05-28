"""Unit tests for query_manager.py — search menu, threshold changes, pagination, result display.

Covers:
- QueryManager initialization and state
- show_menu — choice validation
- image_search / text_search routing (mocked)
- change_threshold — valid/invalid values
- show_index_info
- _display_results — pagination, navigation
- _handle_open_result
- run_interactive_loop edge cases
"""

from unittest.mock import patch, MagicMock
import pytest

from momento.query_manager import QueryManager


class TestQueryManagerInit:
    """Tests for QueryManager initialization."""

    def test_init_sets_defaults(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test/folder", threshold=0.25)
        assert manager.index is mock_index
        assert manager.current_folder == "/test/folder"
        assert manager.threshold == 0.25
        assert manager.use_aggregation is True

    def test_init_uses_default_threshold(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test/folder")
        from momento.config import SIMILARITY_THRESHOLD
        assert manager.threshold == SIMILARITY_THRESHOLD


class TestShowMenu:
    """Tests for show_menu."""

    def test_valid_choice_returns_choice(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test/folder")
        with patch("builtins.input", return_value="1"):
            choice = manager.show_menu()
        assert choice == "1"

    def test_valid_quit_returns_q(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test/folder")
        with patch("builtins.input", return_value="q"):
            choice = manager.show_menu()
        assert choice == "q"

    def test_invalid_then_valid_choice(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test/folder")
        inputs = ["invalid", "3"]
        with patch("builtins.input", side_effect=inputs):
            choice = manager.show_menu()
        assert choice == "3"


class TestImageSearch:
    """Tests for image_search."""

    def test_image_search_calls_search_with_path(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        mock_index = MagicMock()
        manager = QueryManager(mock_index, str(tmp_path))
        with patch("builtins.input", return_value=str(img_file)), \
             patch("momento.query_manager.image_search") as mock_search:
            mock_search.return_value = [(0.85, str(img_file))]
            manager.image_search()
            mock_search.assert_called_once()

    def test_image_search_handles_invalid_path(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", return_value="/nonexistent/path.jpg"), \
             patch("momento.query_manager.image_search") as mock_search:
            manager.image_search()
            mock_search.assert_not_called()

    def test_image_search_handles_search_error(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", return_value="/tmp/test.jpg"), \
             patch("momento.query_manager.validate_image_path", return_value=(True, "")), \
             patch("momento.query_manager.image_search", side_effect=RuntimeError("search failed")):
            # Should not raise — error is caught and logged
            manager.image_search()


class TestTextSearch:
    """Tests for text_search."""

    def test_text_search_calls_search_with_query(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", return_value="a cat"), \
             patch("momento.query_manager.text_search") as mock_search:
            mock_search.return_value = [(0.9, "/img/cat.jpg")]
            manager.text_search()
            mock_search.assert_called_once()

    def test_text_search_handles_empty_query(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", return_value=""), \
             patch("momento.query_manager.text_search") as mock_search:
            manager.text_search()
            mock_search.assert_not_called()

    def test_text_search_handles_search_error(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", return_value="query"), \
             patch("momento.query_manager.validate_text_query", return_value=(True, "")), \
             patch("momento.query_manager.text_search", side_effect=RuntimeError("error")):
            # Should not raise
            manager.text_search()


class TestChangeThreshold:
    """Tests for change_threshold."""

    def test_change_to_valid_threshold(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test", threshold=0.20)
        with patch("builtins.input", return_value="0.50"):
            manager.change_threshold()
        assert manager.threshold == 0.50

    def test_change_to_zero(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", return_value="0.0"):
            manager.change_threshold()
        assert manager.threshold == 0.0

    def test_change_to_one(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", return_value="1.0"):
            manager.change_threshold()
        assert manager.threshold == 1.0

    def test_invalid_then_valid_threshold(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test", threshold=0.20)
        inputs = ["invalid", "1.5", "0.30"]
        with patch("builtins.input", side_effect=inputs):
            manager.change_threshold()
        assert manager.threshold == 0.30

    def test_quit_with_q(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test", threshold=0.20)
        with patch("builtins.input", return_value="q"):
            manager.change_threshold()
        assert manager.threshold == 0.20  # Unchanged


class TestShowIndexInfo:
    """Tests for show_index_info."""

    def test_displays_vector_count(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 500
        manager = QueryManager(mock_index, "/photos")
        with patch("builtins.input", return_value="q"):  # Consume any pending input
            pass
        manager.show_index_info()
        mock_index.get_vector_count.assert_called_once()


class TestDisplayResults:
    """Tests for _display_results."""

    def test_no_results_shows_message(self, capsys):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        manager._display_results([])
        captured = capsys.readouterr()
        assert "No matches found" in captured.out

    def test_single_page_results(self, capsys):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        results = [(0.95, "/img/a.jpg"), (0.85, "/img/b.jpg")]
        with patch("builtins.input", return_value="q"):
            manager._display_results(results)
        captured = capsys.readouterr()
        assert "92%" in captured.out  # format_bar for 0.95 is 10 chars, so block + pct

    def test_multi_page_navigation(self, capsys):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        # Create 12 results so we have 3 pages (5 per page)
        results = [(0.9, f"/img/{i}.jpg") for i in range(12)]
        inputs = ["n", "q"]  # Go to next page, then quit
        with patch("builtins.input", side_effect=inputs):
            manager._display_results(results)
        captured = capsys.readouterr()
        # Should mention page numbers
        assert "Page" in captured.out

    def test_handle_open_result_valid(self):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        results = [(0.9, "/img/photo.jpg")]
        with patch("builtins.input", return_value="1"), \
             patch("momento.query_manager.open_file") as mock_open, \
             patch("os.path.exists", return_value=True):
            manager._handle_open_result(results, 0, 1)
            mock_open.assert_called_once_with("/img/photo.jpg")

    def test_handle_open_result_invalid_number(self, capsys):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        results = [(0.9, "/img/photo.jpg")]
        with patch("builtins.input", return_value="5"):
            manager._handle_open_result(results, 0, 1)
        captured = capsys.readouterr()
        assert "Invalid result number" in captured.out

    def test_handle_open_result_file_not_found(self, capsys):
        mock_index = MagicMock()
        manager = QueryManager(mock_index, "/test")
        results = [(0.9, "/img/nonexistent.jpg")]
        with patch("builtins.input", return_value="1"), \
             patch("os.path.exists", return_value=False):
            manager._handle_open_result(results, 0, 1)
        captured = capsys.readouterr()
        assert "File not found" in captured.out


class TestRunInteractiveLoop:
    """Tests for run_interactive_loop."""

    def test_quit_breaks_loop(self, capsys):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", side_effect=["q"]):
            manager.run_interactive_loop()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_index_info_then_quit(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", side_effect=["5", "q"]):
            manager.run_interactive_loop()

    def test_reindex_breaks_loop(self):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", side_effect=["4"]):
            manager.run_interactive_loop()

    def test_keyboard_interrupt_handled(self, capsys):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            manager.run_interactive_loop()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_eof_error_handled(self, capsys):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", side_effect=EOFError):
            manager.run_interactive_loop()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_unexpected_error_handled(self, capsys):
        mock_index = MagicMock()
        mock_index.get_vector_count.return_value = 100
        manager = QueryManager(mock_index, "/test")
        with patch("builtins.input", side_effect=["1", Exception("unexpected")]):
            manager.run_interactive_loop()
        captured = capsys.readouterr()
        assert "unexpected" in captured.out