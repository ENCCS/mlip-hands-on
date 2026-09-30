"""CPU-only checks for the two-engine starting-state generator."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("generate_starts.py")
SPEC = importlib.util.spec_from_file_location("generate_starts", SOURCE)
STARTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STARTS)


class MatchedStartsTests(unittest.TestCase):
    def test_same_atoms_and_velocity_units_for_both_engines(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "starts"
            output.mkdir()
            STARTS.write_state(output, 2, 0, 300.0)
            record = json.loads((output / "replica-0.json").read_text())
            data = (output / "replica-0.data").read_text()
            atoms = data.split("Atoms # atomic\n\n", 1)[1].split("\n\nVelocities", 1)[0]
            metal = data.split("Velocities\n\n", 1)[1].strip().splitlines()
            self.assertEqual(record["atoms"], 64)
            self.assertEqual(len(atoms.splitlines()), 64)
            self.assertEqual(len(metal), 64)
            for row, position in zip(atoms.splitlines(), record["positions_a"]):
                self.assertEqual([float(x) for x in row.split()[2:]], position)
            for row, internal in zip(metal, record["velocities_internal"]):
                observed = [float(x) for x in row.split()[1:]]
                expected = [v * 1000 / STARTS.FS_PER_INTERNAL_TIME for v in internal]
                for left, right in zip(observed, expected):
                    self.assertTrue(math.isclose(left, right, rel_tol=1e-14))
            self.assertTrue(all(abs(sum(v[axis] for v in record["velocities_internal"])) < 1e-12
                                for axis in range(3)))

    def test_replica_seed_is_deterministic_and_distinct(self):
        first = STARTS.make_state(2, 0, 300.0)
        second = STARTS.make_state(2, 0, 300.0)
        other = STARTS.make_state(2, 1, 300.0)
        self.assertEqual(first, second)
        self.assertEqual(first[0], other[0])
        self.assertNotEqual(first[1], other[1])


if __name__ == "__main__":
    unittest.main()
