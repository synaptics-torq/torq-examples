import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.deps import MissingRequirementsError, _requirement_name, check_requirements


class DependencyCheckTest(unittest.TestCase):
    def test_requirement_name_ignores_comments_options_and_paths(self):
        self.assertIsNone(_requirement_name(""))
        self.assertIsNone(_requirement_name("# comment"))
        self.assertIsNone(_requirement_name("-r other.txt"))
        self.assertIsNone(_requirement_name("./wheelhouse/pkg.whl"))
        self.assertIsNone(_requirement_name("https://example.com/pkg.whl"))

    def test_requirement_name_strips_specifiers_and_extras(self):
        self.assertEqual(_requirement_name("numpy<2.0"), "numpy")
        self.assertEqual(_requirement_name("tokenizers==0.23.1"), "tokenizers")
        self.assertEqual(_requirement_name("requests[socks]>=2"), "requests")

    def test_check_requirements_uses_installed_distributions(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("Pillow\nnumpy<2.0\n")

            with mock.patch(
                "utils.deps.importlib.metadata.distribution",
                return_value=object(),
            ) as distribution:
                check_requirements(req)

            self.assertEqual(
                [call.args[0] for call in distribution.call_args_list],
                ["Pillow", "numpy"],
            )

    def test_check_requirements_raises_setup_error_for_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("missing-pkg\n")

            with mock.patch(
                "utils.deps.importlib.metadata.distribution",
                side_effect=importlib.metadata.PackageNotFoundError,
            ):
                with self.assertRaises(MissingRequirementsError):
                    check_requirements(req)


if __name__ == "__main__":
    unittest.main()
