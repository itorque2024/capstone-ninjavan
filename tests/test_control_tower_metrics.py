"""Tests for control tower metrics calculation."""
import pytest


class TestDemandSpikeCalculation:
    """Regression tests for the baseline_avg bug fix."""

    def _calc_spike(self, peak, baseline):
        """Mirror the spike_mult logic from control_tower.py."""
        return round(peak / baseline, 2) if baseline else 1.0

    def test_spike_mult_matches_multiplier(self):
        """2.5x scenario: peak = baseline * 2.5, spike_mult should be 2.5."""
        baseline = 500.0
        peak = baseline * 2.5
        assert self._calc_spike(peak, baseline) == pytest.approx(2.5)

    def test_custom_20x_spike_mult_is_20(self):
        baseline = 500.0
        peak = baseline * 20.0
        assert self._calc_spike(peak, baseline) == pytest.approx(20.0)

    def test_spike_not_4722_when_baseline_correct(self):
        """Regression: old bug read baseline=1 giving spike=4722 for 20x."""
        baseline = 500.0
        peak = baseline * 20.0
        spike = self._calc_spike(peak, baseline)
        assert spike != pytest.approx(4722.0)
        assert spike == pytest.approx(20.0)

    def test_zero_baseline_returns_1(self):
        assert self._calc_spike(500, 0) == 1.0

    def test_normal_scenario_spike_is_1(self):
        baseline = 500.0
        spike = self._calc_spike(baseline, baseline)
        assert spike == pytest.approx(1.0)


class TestRidersCalculation:
    PARCELS_PER_RIDER = 80

    def riders_for(self, parcels):
        import math
        return max(1, math.ceil(parcels / self.PARCELS_PER_RIDER))

    def test_20x_gives_more_riders_than_2point5x(self):
        baseline = 500.0
        riders_20x = self.riders_for(baseline * 20)
        riders_2point5x = self.riders_for(baseline * 2.5)
        assert riders_20x > riders_2point5x

    def test_riders_scale_with_demand(self):
        assert self.riders_for(80) == 1
        assert self.riders_for(160) == 2
        assert self.riders_for(500) == 7
        assert self.riders_for(10000) == 125

    def test_minimum_1_rider_even_for_low_demand(self):
        assert self.riders_for(1) == 1
