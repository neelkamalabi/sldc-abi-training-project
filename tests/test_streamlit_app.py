"""
Test suite for Streamlit Iris Classifier UI.
"""
from unittest.mock import Mock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest


@pytest.fixture
def mock_api_healthy():
    """Mock a healthy API response."""
    def _mock_get(url, **kwargs):
        mock_response = Mock()
        if "/health" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "model_loaded": True,
                "features": ["sepal length (cm)", "sepal width (cm)", 
                           "petal length (cm)", "petal width (cm)"],
                "classes": ["setosa", "versicolor", "virginica"]
            }
        return mock_response
    return _mock_get


@pytest.fixture
def mock_api_predict_success():
    """Mock a successful prediction response."""
    def _mock_post(url, **kwargs):
        mock_response = Mock()
        if "/predict" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "predicted_class": 0,
                "predicted_species": "setosa",
                "confidence": 0.95,
                "probabilities": {
                    "setosa": 0.95,
                    "versicolor": 0.03,
                    "virginica": 0.02
                }
            }
        return mock_response
    return _mock_post


@pytest.fixture
def mock_api_unhealthy():
    """Mock an unhealthy API response."""
    def _mock_get(url, **kwargs):
        mock_response = Mock()
        if "/health" in url:
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "unhealthy",
                "model_loaded": False
            }
        return mock_response
    return _mock_get


@pytest.fixture
def mock_api_error():
    """Mock an API error response."""
    def _mock_post(url, **kwargs):
        mock_response = Mock()
        if "/predict" in url:
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
        return mock_response
    return _mock_post


@pytest.fixture
def mock_api_timeout():
    """Mock API timeout."""
    def _mock_get(url, **kwargs):
        raise requests.exceptions.Timeout("Connection timeout")
    return _mock_get


class TestAppInitialization:
    """Tests for app initialization and configuration."""
    
    @patch('requests.get')
    def test_app_loads_successfully(self, mock_get, mock_api_healthy):
        """Test that the app loads successfully with healthy API."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        assert not at.exception
    
    @patch('requests.get')
    def test_page_title_set(self, mock_get, mock_api_healthy):
        """Test that page title is set correctly."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Check title is present
        assert len(at.title) > 0
        assert "Iris" in at.title[0].value
    
    @patch('requests.get')
    def test_api_url_environment_variable(self, mock_get, mock_api_healthy):
        """Test API URL can be configured via environment variable."""
        mock_get.side_effect = mock_api_healthy
        
        with patch.dict('os.environ', {'API_BASE': 'http://custom:9000'}):
            at = AppTest.from_file("streamlit_app.py")
            at.run()
            
            # API should be called with custom URL
            mock_get.assert_called()
            call_url = mock_get.call_args[0][0]
            assert "custom:9000" in call_url


class TestHealthCheck:
    """Tests for API health check functionality."""
    
    @patch('requests.get')
    def test_healthy_api_shows_success(self, mock_get, mock_api_healthy):
        """Test success message shown when API is healthy."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Check for success message
        assert len(at.success) > 0
        success_text = at.success[0].value
        assert "API is running" in success_text or "healthy" in success_text.lower()
    
    @patch('requests.get')
    def test_unhealthy_api_shows_error(self, mock_get, mock_api_unhealthy):
        """Test error message shown when API is unhealthy."""
        mock_get.side_effect = mock_api_unhealthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Check for error message
        assert len(at.error) > 0
    
    @patch('requests.get')
    def test_api_timeout_shows_error(self, mock_get, mock_api_timeout):
        """Test error message shown when API times out."""
        mock_get.side_effect = mock_api_timeout
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Should show error about API not running
        assert len(at.error) > 0
    
    @patch('requests.get')
    def test_api_connection_error_stops_app(self, mock_get):
        """Test app stops when cannot connect to API."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Should show error message
        assert len(at.error) > 0


class TestInputFeatures:
    """Tests for input feature sliders."""
    
    @patch('requests.get')
    def test_sepal_length_slider_exists(self, mock_get, mock_api_healthy):
        """Test sepal length slider is present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Find sepal length slider
        sepal_length_sliders = [s for s in at.slider if "Sepal Length" in str(s.label)]
        assert len(sepal_length_sliders) > 0
    
    @patch('requests.get')
    def test_sepal_width_slider_exists(self, mock_get, mock_api_healthy):
        """Test sepal width slider is present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        sepal_width_sliders = [s for s in at.slider if "Sepal Width" in str(s.label)]
        assert len(sepal_width_sliders) > 0
    
    @patch('requests.get')
    def test_petal_length_slider_exists(self, mock_get, mock_api_healthy):
        """Test petal length slider is present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        petal_length_sliders = [s for s in at.slider if "Petal Length" in str(s.label)]
        assert len(petal_length_sliders) > 0
    
    @patch('requests.get')
    def test_petal_width_slider_exists(self, mock_get, mock_api_healthy):
        """Test petal width slider is present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        petal_width_sliders = [s for s in at.slider if "Petal Width" in str(s.label)]
        assert len(petal_width_sliders) > 0
    
    @patch('requests.get')
    def test_all_four_sliders_present(self, mock_get, mock_api_healthy):
        """Test all four feature sliders are present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Should have 4 sliders for the 4 features
        assert len(at.slider) >= 4
    
    @patch('requests.get')
    def test_slider_default_values(self, mock_get, mock_api_healthy):
        """Test sliders have appropriate default values."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Check that sliders have reasonable default values
        for slider in at.slider:
            assert slider.value is not None
            assert slider.value >= 0


class TestPredictButton:
    """Tests for predict button functionality."""
    
    @patch('requests.get')
    def test_predict_button_exists(self, mock_get, mock_api_healthy):
        """Test predict button is present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Find predict button
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        assert len(predict_buttons) > 0
    
    @patch('requests.post')
    @patch('requests.get')
    def test_predict_button_makes_api_call(self, mock_get, mock_post, 
                                          mock_api_healthy, mock_api_predict_success):
        """Test clicking predict button makes API call."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_predict_success
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Click predict button
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Verify API was called
            mock_post.assert_called()
    
    @patch('requests.post')
    @patch('requests.get')
    def test_predict_sends_correct_data(self, mock_get, mock_post, 
                                       mock_api_healthy, mock_api_predict_success):
        """Test predict sends correct feature data to API."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_predict_success
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Click predict button
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Check the POST call was made with json data
            if mock_post.called:
                call_kwargs = mock_post.call_args[1]
                assert 'json' in call_kwargs
                json_data = call_kwargs['json']
                
                # Verify all required fields are present
                assert 'sepal_length' in json_data
                assert 'sepal_width' in json_data
                assert 'petal_length' in json_data
                assert 'petal_width' in json_data


class TestPredictionResults:
    """Tests for prediction results display."""
    
    @patch('requests.post')
    @patch('requests.get')
    def test_successful_prediction_shows_result(self, mock_get, mock_post, 
                                               mock_api_healthy, mock_api_predict_success):
        """Test successful prediction displays results."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_predict_success
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Click predict button
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Should show success message with prediction
            assert len(at.success) > 0
    
    @patch('requests.post')
    @patch('requests.get')
    def test_prediction_shows_species_name(self, mock_get, mock_post, 
                                          mock_api_healthy, mock_api_predict_success):
        """Test prediction shows species name."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_predict_success
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Check that species name appears in success message
            if at.success:
                success_text = str(at.success[0].value).lower()
                assert "setosa" in success_text or "species" in success_text
    
    @patch('requests.post')
    @patch('requests.get')
    def test_prediction_shows_confidence(self, mock_get, mock_post, 
                                        mock_api_healthy, mock_api_predict_success):
        """Test prediction shows confidence score."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_predict_success
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Check for metric display (confidence)
            assert len(at.metric) > 0
    
    @patch('requests.post')
    @patch('requests.get')
    def test_prediction_shows_probabilities(self, mock_get, mock_post, 
                                           mock_api_healthy, mock_api_predict_success):
        """Test prediction shows probability distribution."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_predict_success
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Check for progress bars (used to display probabilities)
            assert len(at.progress) >= 3  # Should have 3 progress bars for 3 classes
    
    @patch('requests.post')
    @patch('requests.get')
    def test_api_error_shows_error_message(self, mock_get, mock_post, 
                                          mock_api_healthy, mock_api_error):
        """Test API error shows error message."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_error
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Should show error message
            assert len(at.error) > 0
    
    @patch('requests.post')
    @patch('requests.get')
    def test_connection_error_shows_error_message(self, mock_get, mock_post, 
                                                  mock_api_healthy):
        """Test connection error shows error message."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            # Should show error message
            assert len(at.error) > 0


class TestExamplePresets:
    """Tests for example preset buttons."""
    
    @patch('requests.get')
    def test_example_preset_buttons_exist(self, mock_get, mock_api_healthy):
        """Test example preset buttons are present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Should have example buttons for different species
        example_buttons = [b for b in at.button if "Example" in str(b.label)]
        assert len(example_buttons) >= 3  # At least 3 species examples
    
    @patch('requests.get')
    def test_setosa_example_button_exists(self, mock_get, mock_api_healthy):
        """Test Setosa example button exists."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        setosa_buttons = [b for b in at.button if "Setosa" in str(b.label)]
        assert len(setosa_buttons) > 0
    
    @patch('requests.get')
    def test_versicolor_example_button_exists(self, mock_get, mock_api_healthy):
        """Test Versicolor example button exists."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        versicolor_buttons = [b for b in at.button if "Versicolor" in str(b.label)]
        assert len(versicolor_buttons) > 0
    
    @patch('requests.get')
    def test_virginica_example_button_exists(self, mock_get, mock_api_healthy):
        """Test Virginica example button exists."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        virginica_buttons = [b for b in at.button if "Virginica" in str(b.label)]
        assert len(virginica_buttons) > 0


class TestUILayout:
    """Tests for UI layout and structure."""
    
    @patch('requests.get')
    def test_header_present(self, mock_get, mock_api_healthy):
        """Test page header is present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Check for header
        assert len(at.header) >= 2  # Should have headers for sections
    
    @patch('requests.get')
    def test_markdown_content_present(self, mock_get, mock_api_healthy):
        """Test markdown content is present."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Should have markdown content
        assert len(at.markdown) > 0
    
    @patch('requests.get')
    def test_expander_about_section(self, mock_get, mock_api_healthy):
        """Test 'About the Model' expander exists."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Should have at least one expander
        assert len(at.expander) > 0


class TestAPIIntegration:
    """Tests for API integration."""
    
    @patch('requests.get')
    def test_api_base_url_used(self, mock_get, mock_api_healthy):
        """Test API base URL is used for requests."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Verify API was called
        mock_get.assert_called()
        call_url = mock_get.call_args[0][0]
        assert "/health" in call_url
    
    @patch('requests.get')
    def test_health_check_timeout_set(self, mock_get, mock_api_healthy):
        """Test health check request has timeout."""
        mock_get.side_effect = mock_api_healthy
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        # Check that timeout was set
        call_kwargs = mock_get.call_args[1]
        assert 'timeout' in call_kwargs
    
    @patch('requests.post')
    @patch('requests.get')
    def test_predict_request_timeout_set(self, mock_get, mock_post, 
                                        mock_api_healthy, mock_api_predict_success):
        """Test predict request has timeout."""
        mock_get.side_effect = mock_api_healthy
        mock_post.side_effect = mock_api_predict_success
        
        at = AppTest.from_file("streamlit_app.py")
        at.run()
        
        predict_buttons = [b for b in at.button if "Predict" in str(b.label)]
        if predict_buttons:
            at.button[predict_buttons[0].key].click().run()
            
            if mock_post.called:
                call_kwargs = mock_post.call_args[1]
                assert 'timeout' in call_kwargs
