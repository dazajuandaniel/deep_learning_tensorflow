"""
Unit Tests for Utils.py
"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
import pytest

from utils import get_run_logdir, get_fashion_mnist_data

class TestGetRunLogDir():
    """
    Test the get_run_logdir function
    """

    def test_is_string_instance(self):
        """
        Tests that the output is a string
        """
        assert isinstance(get_run_logdir(), str)

    def test_name_in_path(self):
        """
        Tests that the name given is in path
        """
        path = get_run_logdir(folder_name = "test_string")
        assert "test_string" in path


class TestGetFashionMnistData():
    """
    Test the get_run_logdir function
    """

    def test_is_tuple(self):
        """
        Tests that the output is a tuple
        """
        assert isinstance(get_fashion_mnist_data(), tuple)

    def test_complete_set(self):
        """
        Tests that the tuple is complete
        """
        data = get_fashion_mnist_data()
        assert len(data) == 6
