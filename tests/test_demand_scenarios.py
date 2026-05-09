"""Tests for demand agent scenario injector logic."""
import pytest
from unittest.mock import patch, MagicMock


def make_forecast(baseline=500.0, days=7):
    return {
        "baseline_avg": baseline,
        "values": [baseline] * days,
        "events": [None] * days,
        "dates": [f"2026-05-{i+1:02d}" for i in range(days)],
        "riders_needed": [10] * days,
    }


def run_scenario(scenario_id, baseline=500.0, custom_multiplier=None):
    """Run demand agent with a mocked model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = make_forecast(baseline)

    state = {
        "forecast_horizon": 7,
        "scenario_id": scenario_id,
        "custom_multiplier": custom_multiplier,
    }

    with patch("src.agents.demand_agent.MODEL_PATH") as mock_path, \
         patch("joblib.load", return_value=mock_model):
        mock_path.exists.return_value = True
        from src.agents.demand_agent import run_demand_agent
        return run_demand_agent(state)


class TestScenarioMultipliers:
    def test_normal_scenario_leaves_values_unchanged(self):
        result = run_scenario("normal")
        fc = result["demand_result"]["forecast"]
        assert fc["values"][0] == pytest.approx(500.0)

    def test_1111_spike_multiplies_day0_by_2point5(self):
        result = run_scenario("spike")
        fc = result["demand_result"]["forecast"]
        assert fc["values"][0] == pytest.approx(500.0 * 2.5)

    def test_1212_sale_multiplies_day0_by_2point2(self):
        result = run_scenario("sale_1212")
        fc = result["demand_result"]["forecast"]
        assert fc["values"][0] == pytest.approx(500.0 * 2.2)

    def test_flash_sale_multiplies_day0_by_3(self):
        result = run_scenario("flash")
        fc = result["demand_result"]["forecast"]
        assert fc["values"][0] == pytest.approx(500.0 * 3.0)

    def test_cny_multiplies_first_3_days(self):
        result = run_scenario("cny")
        fc = result["demand_result"]["forecast"]
        assert fc["values"][0] == pytest.approx(500.0 * 1.8)
        assert fc["values"][1] == pytest.approx(500.0 * 1.8)
        assert fc["values"][2] == pytest.approx(500.0 * 1.8)
        assert fc["values"][3] == pytest.approx(500.0)  # day 4 unchanged

    def test_lockdown_reduces_all_days_by_85pct(self):
        result = run_scenario("lockdown")
        fc = result["demand_result"]["forecast"]
        for v in fc["values"]:
            assert v == pytest.approx(500.0 * 0.15)

    def test_custom_multiplier_uses_baseline(self):
        result = run_scenario("custom", baseline=400.0, custom_multiplier=10.0)
        fc = result["demand_result"]["forecast"]
        assert fc["values"][0] == pytest.approx(400.0 * 10.0)

    def test_custom_20x_gives_more_demand_than_1111_2point5x(self):
        r_custom = run_scenario("custom", baseline=500.0, custom_multiplier=20.0)
        r_spike = run_scenario("spike", baseline=500.0)
        custom_day0 = r_custom["demand_result"]["forecast"]["values"][0]
        spike_day0 = r_spike["demand_result"]["forecast"]["values"][0]
        assert custom_day0 > spike_day0

    def test_scenario_labels_match_multiplier(self):
        result = run_scenario("spike")
        fc = result["demand_result"]["forecast"]
        assert "11.11" in fc["events"][0] or "MEGA" in fc["events"][0]

    def test_spike_detected_flag_set_on_high_demand(self):
        result = run_scenario("spike")
        assert result["demand_result"]["spike_detected"] is True

    def test_no_spike_on_normal_scenario(self):
        result = run_scenario("normal")
        assert result["demand_result"]["spike_detected"] is False


class TestDemandSpikeMath:
    def test_spike_percentage_1111_is_150pct(self):
        """2.5x multiplier = +150% above baseline."""
        baseline = 500.0
        multiplied = baseline * 2.5
        pct_increase = (multiplied - baseline) / baseline * 100
        assert pct_increase == pytest.approx(150.0)

    def test_spike_percentage_flash_is_200pct(self):
        """3.0x multiplier = +200% above baseline."""
        baseline = 500.0
        multiplied = baseline * 3.0
        pct_increase = (multiplied - baseline) / baseline * 100
        assert pct_increase == pytest.approx(200.0)

    def test_custom_20x_is_1900pct_increase(self):
        baseline = 500.0
        multiplied = baseline * 20.0
        pct_increase = (multiplied - baseline) / baseline * 100
        assert pct_increase == pytest.approx(1900.0)
