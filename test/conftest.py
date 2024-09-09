import doctest
import pytest
import re


class NumpyOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        # Remove the np.float64 wrapper from both want and got
        want = re.sub(r'np\.float64\(([\d.-]*)\)', r'\1', want)
        got = re.sub(r'np\.float64\(([\d.-]*)\)', r'\1', got)

        # numpy.set_printoptions started producing output in some version
        if got.strip().startswith("<Token") and want == "":
            return True

        return super().check_output(want, got, optionflags)

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    doctest.OutputChecker = NumpyOutputChecker
