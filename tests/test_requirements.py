from pathlib import Path

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

from tests.utils import wrap_test_forked


def get_requirements():
    req_file = "requirements.txt"
    req_tmp_file = f'{req_file}.tmp.txt'

    reqs_http = []

    with open(req_file, 'rt') as f:
        contents = f.readlines()
        with open(req_tmp_file, 'wt') as g:
            for line in contents:
                if 'http://' not in line and 'https://' not in line:
                    g.write(line)
                else:
                    reqs_http.append(line.replace('\n', ''))
    reqs_http = [x for x in reqs_http if x]
    print(f'reqs_http: {reqs_http}', flush=True)

    _REQUIREMENTS_PATH = Path(__file__).parent.with_name(req_tmp_file)
    requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH.open())
    return requirements, reqs_http


@wrap_test_forked
def test_requirements():
    """Test that each required package is available."""
    packages_all = []
    packages_dist = []
    packages_version = []
    packages_unkn = []

    requirements, reqs_http = get_requirements()

    for requirement in requirements:
        try:
            requirement = str(requirement)
            pkg_resources.require(requirement)
        except DistributionNotFound:
            packages_all.append(requirement)
            packages_dist.append(requirement)
        except VersionConflict:
            packages_all.append(requirement)
            packages_version.append(requirement)
        except pkg_resources.extern.packaging.requirements.InvalidRequirement:
            packages_all.append(requirement)
            packages_unkn.append(requirement)

    packages_all.extend(reqs_http)
    if packages_dist or packages_version:
        print(f'Missing packages: {packages_dist}', flush=True)
        print(f'Wrong version of packages: {packages_version}', flush=True)
        print(f"Can't determine (e.g. http) packages: {packages_unkn}", flush=True)
        print('\n\nRUN THIS:\n\n', flush=True)
        print(
            f"pip uninstall peft transformers accelerate -y ; CUDA_HOME=/usr/local/cuda-11.7 pip install {' '.join(packages_all)} --upgrade",
            flush=True,
        )
        print('\n\n', flush=True)

        raise ValueError(packages_all)


import requests
import json
try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

URL_PATTERN = 'https://pypi.python.org/pypi/{package}/json'


def get_version(package, url_pattern=URL_PATTERN):
    """Return version of package on pypi.python.org using json."""
    req = requests.get(url_pattern.format(package=package))
    version = parse('0')
    if req.status_code == requests.codes.ok:
        j = json.loads(req.text.encode(req.encoding))
        releases = j.get('releases', [])
        for release in releases:
            ver = parse(release)
            if not ver.is_prerelease:
                version = max(version, ver)
    return version


@wrap_test_forked
def test_what_latest_packages():
    # pip install requirements-parser
    import requirements
    import glob
    for req_name in ['requirements.txt'] + glob.glob('reqs_optional/req*.txt'):
        print("\n File: %s" % req_name, flush=True)
        with open(req_name, 'rt') as fd:
            for req in requirements.parse(fd):
                try:
                    print(f"{req.name}=={get_version(req.name)}", flush=True)
                except Exception as e:
                    print(f"Exception: {str(e)}", flush=True)

