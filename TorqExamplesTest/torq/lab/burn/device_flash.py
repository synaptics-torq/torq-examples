#!/usr/bin/env python3
"""Prepare firmware images and configuration for device flashing workflows.

This module provides shared helpers used by the flashing case
runners: parsing per-board INI configuration sections,
downloading and staging firmware images for TFTP, and generating a
platform-info JSON artifact describing the resolved board configuration.

Public interface:
- `list_config_sections`: list the section names defined in an INI config file.
- `resolve_values`: merge CLI arguments with a config section, enforcing any
  required keys.
- `resolve_tftp_path`: resolve the image target and TFTP path without downloading.
- `prepare_tftp_image`: download (or reuse a staged local) eMMC image and
  return the resolved TFTP path.
- `generate_platform_info`: write a JSON summary of resolved board
  configurations for CI artifact upload.

Environment:
- `BUILD_VERSION`/`PROFILE`: used to build the default image URL, local image
  path, and TFTP path when not passed explicitly.
- `IMAGE_BASE_URL`: base URL of the firmware file server used to build the
  default image URL; required when `--image-url` is not passed explicitly.
- `IMAGE_LOCAL`/`TFTP_ROOT`: override the local directory used to stage
  downloaded images.
"""
from html.parser import HTMLParser
import configparser
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote, unquote, urldefrag, urljoin, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

from .runner_heartbeat import suppress_heartbeat

LOGGER = logging.getLogger(__name__)


# --- config parsing ---------------------------------------------------------

def _read_config(config_path, section=None):
    config = configparser.ConfigParser()
    config.optionxform = str
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f'Configuration file does not exist: {config_path}')
    config.read(config_path)
    if section is None:
        raise ValueError('Configuration section must be selected')
    if section not in config:
        raise ValueError(f'Configuration section does not exist: {section}')
    values = {key.upper(): value for key, value in config.items(section)}
    return values


def list_config_sections(config_path):
    config = configparser.ConfigParser()
    config.optionxform = str
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f'Configuration file does not exist: {config_path}')
    config.read(config_path)
    return config.sections()


def resolve_values(args, required_keys=None, config_keys=None):
    config_keys = required_keys if config_keys is None else config_keys
    config_environment_keys = (
        None if config_keys is None else [key.upper() for key in config_keys]
    )
    required_environment_keys = (
        None if required_keys is None else [key.upper() for key in required_keys]
    )
    values = {
        argument.upper(): value
        for argument, value in vars(args).items()
        if value is not None and (
            config_environment_keys is not None
            and argument.upper() in config_environment_keys
        )
    }
    values.update(_read_config(args.config, args.section))
    for argument, argument_value in vars(args).items():
        if argument_value is not None and (
            config_environment_keys is not None
            and argument.upper() in config_environment_keys
        ):
            values[argument.upper()] = argument_value
    if required_environment_keys is None:
        return values
    missing = [name for name in required_environment_keys if not values.get(name)]
    if missing:
        all_missings = ", ".join(missing)
        raise ValueError(f'Missing configuration: {all_missings}')
    return {name: values[name] for name in required_environment_keys}


# --- U-Boot version check ---------------------------------------------------

# Requires a trailing '-' or line break so a partially received line is not matched.
UBOOT_VERSION_PATTERN = re.compile(r'U-Boot\s+([^\s-]+)(?=[-\r\n])')
# Written to the section log folder when flashing is skipped.
BOARD_BUILD_VERSION_FILENAME = 'board_build_version.txt'


def parse_uboot_build_version(version_output):
    match = UBOOT_VERSION_PATTERN.search(version_output)
    if not match:
        return None
    build_version = re.search(r'(\d{12})$', match.group(1))
    return build_version.group(1) if build_version else None


def image_needs_upgrade(current_version, build_version=None):
    build_version = build_version or os.environ.get('BUILD_VERSION')
    if not build_version or len(build_version) != 12 or not build_version.isdigit():
        raise ValueError('BUILD_VERSION must use the YYYYMMDDHHMM format')
    if current_version is None:
        return True
    return int(current_version) < int(build_version)


# --- firmware image preparation ---------------------------------------------

def _get_tftp_ip():
    for interface in ('eth1', 'eth0'):
        ip_command = 'ip'
        if not Path('/usr/bin/ip').exists() and Path('/sbin/ip').exists():
            ip_command = '/sbin/ip'
        result = subprocess.run(
            [ip_command, '-f', 'inet', 'addr', 'show', interface],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r'inet ([0-9.]+)', result.stdout)
        if match:
            return match.group(1)
    raise ValueError('Unable to determine the TFTP interface IP address')


def _build_tftppath(build_version=None, profile=None):
    build_version = build_version or os.environ.get('BUILD_VERSION')
    profile = profile or os.environ.get('PROFILE')
    missing = [
        name for name, value in (
            ('BUILD_VERSION', build_version),
            ('PROFILE', profile),
        )
        if not value
    ]
    if missing:
        all_missings = ", ".join(missing)
        raise ValueError(
            f'Missing TFTP configuration: {all_missings}'
        )
    ip = _get_tftp_ip()
    return f"{ip}:{build_version}/{profile}/eMMCimg"


def _resolve_image_target(image_url=None, update_environment=True):
    build_version = profile = None
    if image_url:
        build_version, profile = _parse_image_url(image_url)
        if update_environment and build_version and profile:
            os.environ['BUILD_VERSION'] = build_version
            os.environ['PROFILE'] = profile
    return build_version, profile


def resolve_tftp_path(
    image_url=None,
    build_version=None,
    profile=None,
    update_environment=True,
):
    image_build_version, image_profile = _resolve_image_target(
        image_url, update_environment=update_environment
    )
    build_version = image_build_version or build_version
    profile = image_profile or profile
    return _build_tftppath(build_version, profile)


def _build_image_url(build_version=None, profile=None, image_base_url=None):
    build_version = build_version or os.environ.get('BUILD_VERSION')
    profile = profile or os.environ.get('PROFILE')
    base_url = image_base_url or os.environ.get('IMAGE_BASE_URL')
    missing = [
        name for name, value in (
            ('BUILD_VERSION', build_version),
            ('PROFILE', profile),
            ('IMAGE_BASE_URL', base_url),
        )
        if not value
    ]
    if missing:
        all_missings = ", ".join(missing)
        raise ValueError(f'Missing image URL configuration: {all_missings}')
    if len(build_version) != 12 or not build_version.isdigit():
        raise ValueError('BUILD_VERSION must use the YYYYMMDDHHMM format')

    month = build_version[:6]
    build_date = build_version[:8]
    encoded_profile = quote(profile, safe='')
    fixed_url = base_url.rstrip('/')
    return (
        f"{fixed_url}/{month}/{build_date}/{build_version}/"
        f"LinuxSDK-Serpens/{encoded_profile}/Eng_Build/"
    )


def _parse_image_url(image_url):
    match = re.search(
        r'/(\d{12})/LinuxSDK-Serpens/([^/]+)/',
        unquote(urlparse(image_url).path),
    )
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _build_image_local(build_version=None, profile=None, tftp_root=None):
    build_version = build_version or os.environ.get('BUILD_VERSION')
    profile = profile or os.environ.get('PROFILE')
    missing = [
        name for name, value in (
            ('BUILD_VERSION', build_version),
            ('PROFILE', profile),
        )
        if not value
    ]
    if missing:
        all_missings = ", ".join(missing)
        raise ValueError(
            f'Missing image destination configuration: {all_missings}'
        )
    tftp_root = tftp_root or os.environ.get('TFTP_ROOT', '/tftproot')
    return Path(tftp_root).expanduser() / build_version / profile


class _DirectoryLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def _download_url(
    url, download_root, base_url, depth=0, visited=None, filename_prefix=None,
    retries=3, timeout=60,
):
    if visited is None:
        visited = set()
    url, _ = urldefrag(url)
    if url in visited:
        return
    visited.add(url)
    if depth > 4:
        return

    parsed_url = urlparse(url)
    if Path(parsed_url.path).suffix:
        relative_path = unquote(parsed_url.path)
        if depth:
            root_path = urlparse(base_url).path.rstrip('/') + '/'
            relative_path = relative_path[len(root_path):]
        else:
            relative_path = Path(relative_path).name
        relative_path = relative_path.lstrip('/')
        destination = (download_root / relative_path).resolve()
        if os.path.commonpath((str(download_root.resolve()), str(destination))) != str(download_root.resolve()):
            raise ValueError('Image URL contains an unsafe download path')
        _download_file(url, destination, retries=retries, timeout=timeout)
        return

    parser, final_url = _download_directory_listing(
        url, retries=retries, timeout=timeout
    )
    base = urlparse(base_url)
    for href in parser.links:
        linked_url, _ = urldefrag(urljoin(final_url, href))
        linked = urlparse(linked_url)
        if linked.netloc != base.netloc or not linked.path.startswith(base.path):
            continue
        if linked.path.rstrip('/') == urlparse(final_url).path.rstrip('/'):
            continue
        name = Path(unquote(linked.path)).name.lower()
        if name in ('', '..', 'index.html') or name.endswith(('.log', '.tgz')):
            continue
        if filename_prefix and not (
            name.startswith(filename_prefix.lower()) and name.endswith('.zip')
        ):
            continue
        _download_url(
            linked_url,
            download_root,
            base_url,
            depth + 1,
            visited,
            filename_prefix,
            retries,
            timeout,
        )


def _download_directory_listing(url, retries, timeout):
    request_headers = {'User-Agent': 'device-toolkit'}
    total_attempts = retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            request = Request(url, headers=request_headers)
            with urlopen(request, timeout=timeout) as response:
                parser = _DirectoryLinks()
                parser.feed(response.read().decode('utf-8', errors='replace'))
                return parser, response.geturl()
        except OSError as error:
            if attempt == total_attempts:
                raise OSError(
                    f'Failed to fetch {url} after {total_attempts} attempts'
                ) from error
            delay = 5 * attempt
            warning_msg = (
                f'Failed to fetch {url}: {error}; retrying in {delay} seconds. '
                f'(Attempt {attempt}/{total_attempts})'
            )
            LOGGER.warning(warning_msg)
            time.sleep(delay)


def _download_file(url, destination, retries=3, timeout=60):
    """Download one file, resuming an incomplete transfer when Range is supported.

    `retries` is the number of attempts after the initial request. Partial data
    is kept in a sibling `.part` file so another call can resume it.
    """
    if retries < 0:
        raise ValueError('retries must be non-negative')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + '.part')
    if destination.is_file():
        return destination

    total_attempts = retries + 1
    for attempt in range(1, total_attempts + 1):
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        request_headers = {'User-Agent': 'device-toolkit'}
        if offset:
            request_headers['Range'] = f'bytes={offset}-'
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                status = response.getcode()
                if status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range)
                    if not offset or not match or int(match.group(1)) != offset:
                        raise OSError(f'Invalid Content-Range response: {content_range}')
                    expected_total = None if match.group(3) == '*' else int(match.group(3))
                    write_mode = 'ab'
                elif status == 200:
                    offset = 0
                    expected_total = (
                        int(response.headers['Content-Length'])
                        if response.headers.get('Content-Length') else None
                    )
                    write_mode = 'wb'
                else:
                    raise OSError(f'Unexpected HTTP status {status}')

                with partial_path.open(write_mode) as image_file:
                    shutil.copyfileobj(response, image_file, length=1024 * 1024)

                actual_size = partial_path.stat().st_size
                if expected_total is not None and actual_size != expected_total:
                    raise OSError(
                        f'Incomplete download: received {actual_size} of {expected_total} bytes'
                    )
            os.replace(partial_path, destination)
            return destination
        except HTTPError as error:
            match = re.fullmatch(
                r'bytes \*/(\d+)', error.headers.get('Content-Range', '')
            )
            if (
                error.code == 416
                and offset
                and match
                and offset == int(match.group(1))
            ):
                os.replace(partial_path, destination)
                return destination
            if attempt == total_attempts:
                raise OSError(
                    f'Failed to download {url} after {total_attempts} attempts (HTTP error)'
                ) from error
            delay = 5 * attempt
            warning_msg = (
                f'HTTP error fetching {url}: {error}; retrying in {delay} seconds. '
                f'(Attempt {attempt}/{total_attempts})'
            )
            LOGGER.warning(warning_msg)
            time.sleep(delay)
        except OSError as error:
            if attempt == total_attempts:
                raise OSError(
                    f'Failed to download {url} after {total_attempts} attempts (I/O error)'
                ) from error
            delay = 5 * attempt
            warning_msg = (
                f'Download I/O error for {url}: {error}; retrying in {delay} seconds. '
                f'(Attempt {attempt}/{total_attempts})'
            )
            LOGGER.warning(warning_msg)
            time.sleep(delay)


def _download_image(image_url, download_root, archive_prefix=None, retries=3, timeout=60):
    """Download an image file or recursively download matching files from a URL."""
    if retries < 0:
        raise ValueError('retries must be non-negative')
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in ('http', 'https') or not parsed_url.netloc:
        raise ValueError('IMAGE_URL must be an HTTP or HTTPS URL')
    is_file_url = bool(Path(parsed_url.path).suffix)
    if is_file_url:
        base_path = parsed_url.path.rsplit('/', 1)[0] + '/'
        base_url = parsed_url._replace(path=base_path, query='', fragment='').geturl()
    else:
        base_url = image_url if image_url.endswith('/') else image_url + '/'
        image_url = base_url
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    _download_url(
        image_url,
        download_root,
        base_url,
        filename_prefix=archive_prefix,
        retries=retries,
        timeout=timeout,
    )
    return download_root


def _extract_image_archives(download_root):
    extracted = set()
    for _ in range(5):
        archives = [
            archive for archive in download_root.rglob('*.zip')
            if archive not in extracted
        ]
        if not archives:
            break
        for archive in archives:
            with zipfile.ZipFile(archive) as package:
                destination = archive.parent.resolve()
                for member in package.infolist():
                    member_path = (destination / member.filename).resolve()
                    if os.path.commonpath((str(destination), str(member_path))) != str(destination):
                        raise ValueError('Image archive contains an unsafe path')
                package.extractall(destination)
            extracted.add(archive)


def _is_usable_emmc_image(directory):
    return (directory / 'emmc_image_list').is_file() or (directory / 'bootloader.img').is_file()


def _find_emmc_image(download_root):
    candidates = []
    for directory in [download_root] + [
        path for path in download_root.rglob('*') if path.is_dir()
    ]:
        if _is_usable_emmc_image(directory):
            candidates.append(directory)
    if not candidates:
        raise ValueError('Downloaded image does not contain a usable eMMCimg')
    named_candidates = [path for path in candidates if path.name.lower() == 'emmcimg']
    if named_candidates:
        candidates = named_candidates
    candidates.sort(key=lambda path: len(path.parts))
    shallowest = [path for path in candidates if len(path.parts) == len(candidates[0].parts)]
    if len(shallowest) != 1:
        raise ValueError('Downloaded image contains multiple eMMC image directories')
    return shallowest[0]


def _prepare_image(
    image_url=None, image_local=None, archive_prefix=None,
    build_version=None, profile=None, retries=3, timeout=60,
):
    if image_local:
        image_local = Path(image_local).expanduser()
    elif os.environ.get('IMAGE_LOCAL'):
        image_local = Path(os.environ['IMAGE_LOCAL']).expanduser()
    else:
        image_local = _build_image_local(build_version, profile)
    local_emmc_image = image_local / 'eMMCimg'
    if local_emmc_image.is_dir() and _is_usable_emmc_image(local_emmc_image):
        print(f'Using local eMMC image at {local_emmc_image}', flush=True)
        return local_emmc_image
    if not image_url:
        raise ValueError(
            f'IMAGE_URL was not provided and no local eMMC image exists at {local_emmc_image}'
        )

    image_local.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(image_url.encode('utf-8')).hexdigest()
    download_root = image_local / '.downloads' / cache_key
    print(f'Downloading image from {image_url} into {image_local}', flush=True)
    try:
        # Downloads may be quiet for long periods; suppress only the runner heartbeat.
        with suppress_heartbeat():
            _download_image(
                image_url,
                download_root,
                archive_prefix=archive_prefix,
                retries=retries,
                timeout=timeout,
            )
        if archive_prefix:
            matching_archives = [
                path for path in download_root.rglob('*')
                if path.is_file()
                and path.name.lower().startswith(archive_prefix.lower())
                and path.suffix.lower() == '.zip'
            ]
            if len(matching_archives) != 1:
                raise ValueError(
                    f'Expected exactly one {archive_prefix}*.zip archive at {image_url}, found {len(matching_archives)}'
                )
        _extract_image_archives(download_root)
        emmc_image = _find_emmc_image(download_root)

        with tempfile.TemporaryDirectory(prefix='.syna-staged-', dir=image_local) as stage_dir:
            staged_image = Path(stage_dir) / 'eMMCimg'
            shutil.copytree(emmc_image, staged_image)
            backup_image = Path(stage_dir) / 'previous-eMMCimg'
            if local_emmc_image.exists():
                os.replace(local_emmc_image, backup_image)
            try:
                os.replace(staged_image, local_emmc_image)
            except Exception:
                if backup_image.exists():
                    os.replace(backup_image, local_emmc_image)
                raise
        shutil.rmtree(download_root)
    except ValueError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(
            f'Image download or extraction failed: {error}'
        ) from error
    print(f'Image is ready at {local_emmc_image}', flush=True)
    return local_emmc_image


def prepare_tftp_image(
    image_url=None,
    retries=3,
    timeout=60,
    build_version=None,
    profile=None,
    image_base_url=None,
    tftp_root=None,
    update_environment=True,
):
    image_build_version, image_profile = _resolve_image_target(
        image_url, update_environment=update_environment
    )
    build_version = image_build_version or build_version
    profile = image_profile or profile
    if not image_url:
        image_url = _build_image_url(build_version, profile, image_base_url)
    image_local = None
    if tftp_root is not None or not os.environ.get('IMAGE_LOCAL'):
        image_local = _build_image_local(
            build_version, profile, tftp_root=tftp_root
        )
    _prepare_image(
        image_url,
        image_local=image_local,
        archive_prefix='eMMCimg_',
        build_version=build_version,
        profile=profile,
        retries=retries,
        timeout=timeout,
    )
    return _build_tftppath(build_version, profile)


# --- platform-info artifact generation --------------------------------------

def _resolve_platforms(
    config_path,
    sections=None,
    tftppath=None,
    board_versions=None,
    build_version=None,
    profile=None,
):
    section_names = sections or list_config_sections(config_path)

    platforms = []
    for section_name in section_names:
        resolved = _read_config(config_path, section_name)
        if tftppath:
            for key, value in (
                ('BUILD_VERSION', build_version or os.environ.get('BUILD_VERSION')),
                ('PROFILE', profile or os.environ.get('PROFILE')),
            ):
                if value:
                    resolved[key] = value
        section_tftppath = tftppath or _build_tftppath(
            resolved.get('BUILD_VERSION'), resolved.get('PROFILE')
        )
        if board_versions and board_versions.get(section_name):
            resolved['BUILD_VERSION'] = board_versions[section_name]
        resolved['TFTPPATH'] = section_tftppath
        resolved['TFTPIP'] = section_tftppath.split(':', 1)[0]
        resolved['CONFIG_SECTION'] = section_name
        platforms.append(resolved)
    return platforms


def generate_platform_info(
    args,
    output_path=None,
    sections=None,
    tftppath=None,
    board_versions=None,
    build_version=None,
    profile=None,
):
    if output_path is None:
        LOGGER.warning('Skipping platform info generation: no output path provided')
        return
    if args.config is None:
        LOGGER.warning('Skipping platform info generation: no config file provided')
        return

    section_names = sections
    if section_names is None and args.section:
        section_names = [args.section]
    platforms = _resolve_platforms(
        args.config,
        sections=section_names,
        tftppath=tftppath,
        board_versions=board_versions,
        build_version=build_version,
        profile=profile,
    )
    if not platforms:
        LOGGER.warning('Skipping platform info generation: no platforms resolved')
        return

    file_path = Path(output_path)
    with file_path.open('w', encoding='utf-8') as handle:
        json.dump(platforms, handle, indent=2, sort_keys=True)
    LOGGER.info(f'Wrote platform info to {file_path}')
