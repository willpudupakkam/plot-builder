from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from examples.astro_features import (
    CLUSTER_ROLE_COLUMN,
    MERGER_AXIS_DISTANCE_COLUMN,
    MERGER_AXIS_DISTANCE_NORMALIZED_COLUMN,
    MERGER_AXIS_PERPENDICULAR_RADIUS_COLUMN,
    augment_catalogue,
)


class AstroFeaturesTest(unittest.TestCase):
    def test_compatible_catalogue_is_augmented(self) -> None:
        data = pd.DataFrame(
            {
                "#ID(1)": [100, 200, 300],
                "Xc(6)": [0.0, 2.0, 1.0],
                "Yc(7)": [0.0, 0.0, 3.0],
                "Zc(8)": [0.0, 0.0, 0.0],
                "theta_200": [0.0, 0.0, 0.0],
            }
        )

        result = augment_catalogue(data, Path("catalogue.csv"))

        np.testing.assert_allclose(result[MERGER_AXIS_DISTANCE_COLUMN], [0.0, 2.0, 1.0])
        np.testing.assert_allclose(
            result[MERGER_AXIS_DISTANCE_NORMALIZED_COLUMN],
            [0.0, 1.0, 0.5],
        )
        np.testing.assert_allclose(
            result[MERGER_AXIS_PERPENDICULAR_RADIUS_COLUMN],
            [0.0, 0.0, 3.0],
        )
        self.assertEqual(result[CLUSTER_ROLE_COLUMN].tolist(), ["main", "bullet", ""])

    def test_incompatible_catalogue_is_returned_unchanged(self) -> None:
        data = pd.DataFrame({"x": [1.0], "y": [2.0]})

        result = augment_catalogue(data, Path("catalogue.csv"))

        pd.testing.assert_frame_equal(result, data)


if __name__ == "__main__":
    unittest.main()
