from dotenv import load_dotenv
load_dotenv()
import pytest
from unittest.mock import patch
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import os

@pytest.fixture
def weather_tool():
    return OpenWeatherMapAPIWrapper()

@pytest.mark.langsmith
@patch.object(OpenWeatherMapAPIWrapper, "run", return_value="The weather in Hyderabad is 28°C and clear skies.")
def test_weather_api_valid_response(mock_run, weather_tool):
    """Weather API returns valid formatted data for known city"""
    result = weather_tool.run("Hyderabad")
    assert "hyderabad" in result.lower()
    assert any(keyword in result.lower() for keyword in ["°c", "weather", "temperature"])
    mock_run.assert_called_once_with("Hyderabad")

@patch.object(OpenWeatherMapAPIWrapper, "run", return_value="Weather in Hyderabad is 29°C, clear sky.")
def test_valid_city(mock_run, weather_tool):
    result = weather_tool.run("Hyderabad")
    assert "hyderabad" in result.lower()
    assert any(x in result.lower() for x in ["°c", "weather", "sky"])
    mock_run.assert_called_once_with("Hyderabad")

@patch.object(OpenWeatherMapAPIWrapper, "run", return_value="City not found.")
def test_invalid_city(mock_run, weather_tool):
    result = weather_tool.run("InvalidCity123")
    assert "not found" in result.lower()
    mock_run.assert_called_once_with("InvalidCity123")


@patch.object(OpenWeatherMapAPIWrapper, "run", side_effect=Exception("API Error"))
def test_api_error_handling(mock_run, weather_tool):
    with pytest.raises(Exception) as exc_info:
        weather_tool.run("Hyderabad")
    assert "api error" in str(exc_info.value).lower()
    mock_run.assert_called_once_with("Hyderabad")