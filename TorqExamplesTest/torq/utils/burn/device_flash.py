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
from __future__ import annotations

from html.parser import HTMLParser
import configparser
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote, unquote, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile


LOGGER = logging.getLogger(__name__)


# --- config parsing ---------------------------------------------------------

def _read_config(config_path, section=None):
    config = configparser.ConfigParser()
    config.optionxform = str
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError('Configuration file does not exist: {}'.format(config_path))
    config.read(config_path)
    if section is None:
        raise ValueError('Configuration section must be selected')
    if section not in config:
        raise ValueError('Configuration section does not exist: {}'.format(section))
    values = {key.upper(): value for key, value in config.items(section)}
    return values


def list_config_sections(config_path):
    config = configparser.ConfigParser()
    config.optionxform = str
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError('Configuration file does not exist: {}'.format(config_path))
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
        raise ValueError('Missing configuration: {}'.format(', '.join(missing)))
    return {name: values[name] for name in required_environment_keys}


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
        raise ValueError(
            'Missing TFTP configuration: {}'.format(', '.join(missing))
        )
    ip = _get_tftp_ip()
    return f"{ip}:{build_version}/{profile}/eMMCimg"


def _build_image_url(build_version=None, profile=None):
    build_version = build_version or os.environ.get('BUILD_VERSION')
    profile = profile or os.environ.get('PROFILE')
    base_url = os.environ.get('IMAGE_BASE_URL')
    missing = [
        name for name, value in (
            ('BUILD_VERSION', build_version),
            ('PROFILE', profile),
            ('IMAGE_BASE_URL', base_url),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            'Missing image URL configuration: {}'.format(', '.join(missing))
        )
    if len(build_version) != 12 or not build_version.isdigit():
        raise ValueError('BUILD_VERSION must use the YYYYMMDDHHMM format')

    month = build_version[:6]
    build_date = build_version[:8]
    encoded_profile = quote(profile, safe='')
    return (
        '{}/{}/{}/{}/LinuxSDK-Serpens/{}/Eng_Build/'.format(
            base_url.rstrip('/'),
            month,
            build_date,
            build_version,
            encoded_profile,
        )
    )


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
        raise ValueError(
            'Missing image destination configuration: {}'.format(', '.join(missing))
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
    url, download_root, base_url, depth=0, visited=None, filename_prefix=None
):
    if visited is None:
        visited = set()
    url, _ = urldefrag(url)
    if url in visited:
        return
    visited.add(url)
    if depth > 4:
        return

    request = Request(url, headers={'User-Agent': 'device-toolkit'})
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get_content_type()
        final_url = response.geturl()
        if content_type == 'text/html':
            parser = _DirectoryLinks()
            parser.feed(response.read().decode('utf-8', errors='replace'))
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
                )
            return

        relative_path = unquote(urlparse(final_url).path)
        if depth:
            root_path = urlparse(base_url).path.rstrip('/') + '/'
            relative_path = relative_path[len(root_path):]
        else:
            relative_path = Path(relative_path).name
        relative_path = relative_path.lstrip('/')
        destination = (download_root / relative_path).resolve()
        if os.path.commonpath((str(download_root.resolve()), str(destination))) != str(download_root.resolve()):
            raise ValueError('Image URL contains an unsafe download path')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('wb') as image_file:
            shutil.copyfileobj(response, image_file)


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


def _prepare_image(image_url=None, image_local=None, archive_prefix=None):
    if image_local:
        image_local = Path(image_local).expanduser()
    elif os.environ.get('IMAGE_LOCAL'):
        image_local = Path(os.environ['IMAGE_LOCAL']).expanduser()
    else:
        image_local = _build_image_local()
    local_emmc_image = image_local / 'eMMCimg'
    if not image_url:
        if not local_emmc_image.is_dir() or not _is_usable_emmc_image(local_emmc_image):
            raise ValueError(
                'IMAGE_URL was not provided and no local eMMC image exists at {}'.format(
                    local_emmc_image
                )
            )
        print('Using local eMMC image at {}'.format(local_emmc_image), flush=True)
        return local_emmc_image

    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in ('http', 'https') or not parsed_url.netloc:
        raise ValueError('IMAGE_URL must be an HTTP or HTTPS URL')
    image_local.mkdir(parents=True, exist_ok=True)
    path = parsed_url.path
    is_file_url = bool(Path(path).suffix)
    if is_file_url:
        base_path = path.rsplit('/', 1)[0] + '/'
        base_url = parsed_url._replace(path=base_path, query='', fragment='').geturl()
        download_url = image_url
    else:
        base_url = image_url if image_url.endswith('/') else image_url + '/'
        download_url = base_url
    print('Downloading image from {} into {}'.format(image_url, image_local), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix='.syna-uboot-', dir=image_local) as temp_dir:
            download_root = Path(temp_dir)
            _download_url(
                download_url,
                download_root,
                base_url,
                filename_prefix=archive_prefix,
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
                        'Expected exactly one {}*.zip archive at {}, found {}'.format(
                            archive_prefix, image_url, len(matching_archives)
                        )
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
    except ValueError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError('Image download or extraction failed: {}'.format(error)) from error
    print('Image is ready at {}'.format(local_emmc_image), flush=True)
    return local_emmc_image


def prepare_tftp_image(image_url=None):
    if not image_url:
        image_url = _build_image_url()
    _prepare_image(image_url, archive_prefix='eMMCimg_')
    return _build_tftppath()


# --- platform-info artifact generation --------------------------------------

def _resolve_platforms(config_path, sections=None, tftppath=None):
    section_names = sections or list_config_sections(config_path)

    platforms = []
    for section_name in section_names:
        resolved = _read_config(config_path, section_name)
        section_tftppath = tftppath or _build_tftppath(
            resolved.get('BUILD_VERSION'), resolved.get('PROFILE')
        )
        resolved['TFTPPATH'] = section_tftppath
        resolved['TFTPIP'] = section_tftppath.split(':', 1)[0]
        resolved['CONFIG_SECTION'] = section_name
        platforms.append(resolved)
    return platforms


def generate_platform_info(args, output_path=None, sections=None, tftppath=None):
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
    )
    if not platforms:
        LOGGER.warning('Skipping platform info generation: no platforms resolved')
        return

    file_path = Path(output_path)
    with file_path.open('w', encoding='utf-8') as handle:
        json.dump(platforms, handle, indent=2, sort_keys=True)
    LOGGER.info('Wrote platform info to %s', file_path)
