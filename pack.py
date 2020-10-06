"""Utils to package and publish.



The typical sequence of the methodic and paranoid could be something like this:

```
python pack.py current-configs  # see what you got
python pack.py increment-configs-version  # update (increment the version and write that in setup.cfg
python pack.py current-configs-version  # see that it worked
python pack.py current-configs  # ... if you really want to see the whole configs again (you're really paranoid)
python pack.py run-setup  # see that it worked
python pack.py twine-upload-dist  # publish
# and then go check things work...
```


If you're crazy (or know what you're doing) just do

```
python pack.py go
```


"""
import subprocess
from setuptools import find_packages
import json
import re
from pprint import pprint
from typing import Union, Mapping, Iterable, Generator
from configparser import ConfigParser
import os

DFLT_CONFIG_FILE = 'setup.cfg'
DFLT_CONFIG_SECTION = 'metadata'


def get_local_name():
    """Get name from local setup.cfg (metadata section)"""
    configs = read_configs()
    return configs['name']


def go(version=None, verbose=True):
    """Update version, package and deploy:
    Runs in a sequence: update_setup_cfg, run_setup, twine_upload_dist

    :param version: The desired version (if not given, will increment the current version
    :param verbose: Whether to print stuff or not

    """

    increment_configs_version()
    run_setup()
    twine_upload_dist()


def clog(condition, *args, **kwargs):
    if condition:
        pprint(*args, **kwargs)


def update_version(version):
    """Updates version (writes to setup.cfg)"""
    pass


def current_configs():
    configs = read_configs()
    pprint(configs)


def current_configs_version():
    return read_configs()['version']


# TODO: Both setup and twine are python. Change to use python objects directly.
# def update_setup_cfg(new_deploy=False, version=None, verbose=True):
#     """Update setup.cfg (at this point, just updates the version).
#     If version is not given, will ask pypi (via http request) what the current version is, and increment that.
#     """
#     configs = read_and_resolve_setup_configs(new_deploy=new_deploy, version=version)
#     clog(verbose, pprint(configs))
#     write_configs(configs)


def set_version(version):
    """Update setup.cfg (at this point, just updates the version).
    If version is not given, will ask pypi (via http request) what the current version is, and increment that.
    """
    configs = read_configs()
    assert isinstance(version, str), "version should be a string"
    configs['version'] = version
    write_configs(configs)


def increment_configs_version(version=None):
    """Update setup.cfg (at this point, just updates the version).
    If version is not given, will ask pypi (via http request) what the current version is, and increment that.
    """
    configs = read_configs()
    version = _get_version(version=version, configs=configs, new_deploy=False)
    version = increment_version(version)
    configs['version'] = version
    write_configs(configs)


def run_setup():
    """Run ``python setup.py sdist bdist_wheel``"""
    print('--------------------------- setup_output ---------------------------')
    setup_output = subprocess.run('python setup.py sdist bdist_wheel'.split(' '))
    # print(f"{setup_output}\n")


def twine_upload_dist():
    """Publish to pypi. Runs ``python -m twine upload dist/*``"""
    print('--------------------------- upload_output ---------------------------')
    # TODO: dist/*? How to publish just last on
    upload_output = subprocess.run('python -m twine upload dist/*'.split(' '))
    # print(f"{upload_output.decode()}\n")


# TODO: A lot of work done here to read setup.cfg. setup function apparently does it for you. How to use that?

# TODO: postprocess_ini_section_items and preprocess_ini_section_items: Add comma separated possibility?
# TODO: Find out if configparse has an option to do this processing alreadys
def postprocess_ini_section_items(items: Union[Mapping, Iterable]) -> Generator:
    r"""Transform newline-separated string values into actual list of strings (assuming that intent)

    >>> section_from_ini = {
    ...     'name': 'epythet',
    ...     'keywords': '\n\tdocumentation\n\tpackaging\n\tpublishing'
    ... }
    >>> section_for_python = dict(postprocess_ini_section_items(section_from_ini))
    >>> section_for_python
    {'name': 'epythet', 'keywords': ['documentation', 'packaging', 'publishing']}

    """
    splitter_re = re.compile('[\n\r\t]+')
    if isinstance(items, Mapping):
        items = items.items()
    for k, v in items:
        if v.startswith('\n'):
            v = splitter_re.split(v[1:])
            v = [vv.strip() for vv in v if vv.strip()]
            v = [vv for vv in v if not vv.startswith('#')]  # remove commented lines
        yield k, v


# TODO: Find out if configparse has an option to do this processing alreadys
def preprocess_ini_section_items(items: Union[Mapping, Iterable]) -> Generator:
    """Transform list values into newline-separated strings, in view of writing the value to a ini formatted section
    >>> section = {
    ...     'name': 'epythet',
    ...     'keywords': ['documentation', 'packaging', 'publishing']
    ... }
    >>> for_ini = dict(preprocess_ini_section_items(section))
    >>> print('keywords =' + for_ini['keywords'])  # doctest: +NORMALIZE_WHITESPACE
    keywords =
        documentation
        packaging
        publishing

    """
    if isinstance(items, Mapping):
        items = items.items()
    for k, v in items:
        if isinstance(v, list):
            v = '\n\t' + '\n\t'.join(v)
        yield k, v


def read_configs(
        config_file=DFLT_CONFIG_FILE,
        section=DFLT_CONFIG_SECTION,
        postproc=postprocess_ini_section_items):
    c = ConfigParser()
    c.read_file(open(config_file, 'r'))
    if section is None:
        d = dict(c)
        if postproc:
            d = {k: dict(postproc(v)) for k, v in c}
    else:
        d = dict(c[section])
        if postproc:
            d = dict(postproc(d))
    return d


def write_configs(
        configs,
        config_file=DFLT_CONFIG_FILE,
        section=DFLT_CONFIG_SECTION,
        preproc=preprocess_ini_section_items
):
    c = ConfigParser()
    if os.path.isfile(config_file):
        c.read_file(open(config_file, 'r'))
    c[section] = dict(preproc(configs))
    with open(config_file, 'w') as fp:
        c.write(fp)


# dflt_formatter = Formatter()

def increment_version(version_str):
    version_nums = list(map(int, version_str.split('.')))
    version_nums[-1] += 1
    return '.'.join(map(str, version_nums))


try:
    import requests

    requests_is_installed = True
except ModuleNotFoundError:
    requests_is_installed = False


def http_get_json(url, use_requests=requests_is_installed) -> Union[dict, None]:
    """Make ah http request to url and get json, and return as python dict
    """

    if use_requests:
        import requests
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            raise ValueError(f"response code was {r.status_code}")
    else:
        import urllib.request
        from urllib.error import HTTPError
        req = urllib.request.Request(url)
        try:
            r = urllib.request.urlopen(req)
            if r.code == 200:
                return json.loads(r.read())
            else:
                raise ValueError(f"response code was {r.code}")
        except HTTPError:
            return None  # to indicate (hopefully) that name doesn't exist
        except Exception:
            raise


DLFT_PYPI_PACKAGE_JSON_URL_TEMPLATE = 'https://pypi.python.org/pypi/{package}/json'


# TODO: Perhaps there's a safer way to analyze errors (and determine if the package exists or other HTTPError)
def current_pypi_version(
        name: Union[None, str] = None,
        url_template=DLFT_PYPI_PACKAGE_JSON_URL_TEMPLATE,
        use_requests=requests_is_installed
) -> Union[str, None]:
    """
    Return version of package on pypi.python.org using json.

    ```
    > current_pypi_version('py2store')
    '0.0.7'
    ```

    :param package: Name of the package
    :return: A version (string) or None if there was an exception (usually means there
    """
    name = name or get_local_name()
    url = url_template.format(package=name)
    t = http_get_json(url, use_requests=use_requests)
    releases = t.get('releases', [])
    if releases:
        return sorted(releases, key=lambda r: tuple(map(int, r.split('.'))))[-1]


def next_version_for_package(
        name: Union[None, str] = None,
        url_template=DLFT_PYPI_PACKAGE_JSON_URL_TEMPLATE,
        version_if_current_version_none='0.0.1',
        use_requests=requests_is_installed
) -> str:
    name = name or get_local_name()
    current_version = current_pypi_version(name, url_template, use_requests=use_requests)
    if current_version is not None:
        return increment_version(current_version)
    else:
        return version_if_current_version_none


def _get_version(version,
                 configs,
                 name: Union[None, str] = None,
                 new_deploy=False):
    version = version or configs.get('version', None)
    if version is None:
        try:
            if new_deploy:
                version = next_version_for_package(name)  # when you want to make a new package
            else:
                version = current_pypi_version(name)  # when you want to make a new package
        except Exception as e:
            print(
                f"Got an error trying to get the new version of {name} so will try to get the version from setup.cfg...")
            print(f"{e}")
            version = configs.get('version', None)
            if version is None:
                raise ValueError(f"Couldn't fetch the next version from PyPi (no API token?), "
                                 f"nor did I find a version in setup.cfg (metadata section).")
    return version


def read_and_resolve_setup_configs(new_deploy=False, version=None):
    """make setup params and call setup

    :param new_deploy: whether this setup for a new deployment (publishing to pypi) or not
    :param version: The version number to set this up as.
                    If not given will look at setup.cfg[metadata] for one,
                    and if not found there will use the current version (requesting pypi.org)
                    and bump it if the new_deploy flag is on
    """
    # read the config file (get a dict with it's contents)
    root_dir = os.path.dirname(__file__)
    config_file = os.path.join(root_dir, 'setup.cfg')
    configs = read_configs(config_file, section='metadata')

    # parse out name and root_url
    name = configs['name']
    root_url = configs['root_url']

    # Note: if version is not in config, version will be None,
    #  resulting in bumping the version or making it be 0.0.1 if the package is not found (i.e. first deploy)

    meta_data_dict = {k: v for k, v in configs.items()}

    # make the setup_kwargs
    setup_kwargs = dict(
        meta_data_dict,
        # You can add more key=val pairs here if they're missing in config file
    )

    # import os
    # name = os.path.split(os.path.dirname(__file__))[-1]

    version = _get_version(version, configs, name, new_deploy)

    def text_of_readme_md_file():
        try:
            with open('README.md') as f:
                return f.read()
        except:
            return ""

    if root_url.endswith('/'):
        root_url = root_url[:-1]

    dflt_kwargs = dict(
        name=f"{name}",
        version=f'{version}',
        url=f"{root_url}/{name}",
        packages=find_packages(),
        include_package_data=True,
        platforms='any',
        long_description=text_of_readme_md_file(),
        long_description_content_type="text/markdown",
    )

    configs = dict(dflt_kwargs, **setup_kwargs)

    return configs


if __name__ == '__main__':
    import argh  # pip install argh

    funcs = [
        current_configs,
        increment_configs_version,
        current_configs_version,
        twine_upload_dist,
        read_and_resolve_setup_configs,
        go,
        get_local_name,
        run_setup,
        current_pypi_version
    ]
    argh.dispatch_commands(funcs)
