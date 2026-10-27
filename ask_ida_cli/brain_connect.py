"""Explicit Cortex transport for an owner-controlled local IDA runtime.

Contains no numerical runtime, checkpoint reader, private imports or code download.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from uuid import UUID

from .errors import CLIError


class RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise CLIError('Redirect refused; credentials were not forwarded', code='brain_redirect')


def origin(value, *, local=False):
    parsed = urlsplit(value)
    valid = (parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'
             and parsed.port is not None) if local else (
        parsed.scheme == 'https' and parsed.netloc == 'cortexcommons.com')
    if not valid or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise CLIError('Use the Cortex HTTPS origin and a numeric loopback runtime URL', code='brain_origin')
    return value.rstrip('/')


def identity(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError('invalid identifier')
    return value


class BrainConnector:
    def __init__(self, runtime_url, *, server='https://cortexcommons.com', allow_loopback=False):
        self.runtime = origin(runtime_url, local=True)
        self.server = origin(server, local=allow_loopback)
        self.opener = urllib.request.build_opener(RefuseRedirect())
        self.token = None
        self.brain = None
        self.connection = None

    def __repr__(self):
        return f'BrainConnector(connected={self.brain is not None})'

    def _json(self, base, path, payload=None):
        headers = {'Content-Type': 'application/json'}
        if base == self.server and self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        request = urllib.request.Request(base + path, headers=headers,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode())
        try:
            with self.opener.open(request, timeout=100 if base == self.runtime else 15) as response:
                raw = response.read(16385)
                if len(raw) > 16384:
                    raise ValueError('oversize receipt')
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError('object required')
                return value
        except CLIError:
            raise
        except (ValueError, OSError, urllib.error.URLError):
            raise CLIError('Connection unavailable, invalid, or access refused', code='brain_connection') from None

    def connect(self, *, username, password, title, approved=False, justification=''):
        if not approved or not isinstance(justification, str) or not 1 <= len(justification.strip()) <= 500:
            raise CLIError('Connection requires --approve and a justification', code='approval_required')
        info = self._json(self.runtime, '/cortex/metadata')
        expected = {'artifact_sha256', 'artifact_format', 'storage_precision', 'master_format'}
        import re
        if (set(info) != expected or re.fullmatch('[a-f0-9]{64}', str(info['artifact_sha256'])) is None
            or (info['artifact_format'], info['storage_precision'], info['master_format']) not in {
                ('ida.neurocode-package.v1', 'fp32', 'fp32'),
                ('neurocode.region_router.mxfp8.v1', 'mxfp8', 'mxfp8_e4m3_ue8m0_k32')}):
            raise CLIError('Unsupported local runtime metadata', code='brain_metadata')
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120 or any(ord(c) < 32 for c in title):
            raise CLIError('Invalid brain title', code='brain_metadata')
        session = self._json(self.server, '/api/v1/account/login', {'username': username, 'password': password})
        token = session.get('token')
        if not isinstance(token, str) or re.fullmatch('[A-Za-z0-9_-]{43}', token) is None:
            raise CLIError('Invalid sign-in receipt', code='brain_session')
        self.token = token
        receipt = self._json(self.server, '/api/v1/local-brains', dict(info, title=title))
        try:
            self.brain, self.connection = identity(receipt['brain_id']), identity(receipt['connection_id'])
        except (KeyError, ValueError, TypeError, AttributeError):
            raise CLIError('Invalid connection receipt', code='brain_receipt') from None
        return {'brain_id': self.brain, 'execution_location': 'user_device', 'attestation': 'runner_reported'}

    def run(self, *, stop=lambda: False, limit=None):
        completed = 0
        if self.brain is None:
            raise CLIError('Connect first', code='brain_session')
        route = '/api/v1/local-brains/' + self.brain
        while not stop():
            work = self._json(self.server, route + '/poll', {'connection_id': self.connection}).get('request')
            if work is None:
                time.sleep(2)
                continue
            try:
                if (not isinstance(work, dict) or set(work) != {'request_id', 'prompt', 'max_new_bytes'}
                    or not isinstance(work['prompt'], str) or not 1 <= len(work['prompt'].encode()) <= 512
                    or type(work['max_new_bytes']) is not int or not 1 <= work['max_new_bytes'] <= 128):
                    raise ValueError('invalid work')
                identity(work['request_id'])
            except (ValueError, TypeError, AttributeError):
                raise CLIError('Invalid relay request', code='brain_request') from None
            try:
                result = self._json(self.runtime, '/cortex/complete',
                    {'prompt': work['prompt'], 'max_new_bytes': work['max_new_bytes']})
                fields = {'continuation', 'token_bytes', 'generated_bytes', 'seconds', 'state_policy',
                          'inference_engine', 'inference_adapter', 'observed_backend'}
                if set(result) != fields or result['inference_engine'] != 'ida_cli':
                    raise CLIError('Invalid local runtime output', code='brain_output')
            except CLIError:
                result = {'error': 'brain_execution_failed'}
            self._json(self.server, route + '/result', {'connection_id': self.connection,
                'request_id': work['request_id'], 'result': result})
            completed += 1
            if limit is not None and completed >= limit:
                break
        return completed

    def close(self):
        try:
            if self.brain:
                self._json(self.server, '/api/v1/local-brains/' + self.brain + '/disconnect',
                           {'connection_id': self.connection})
        finally:
            try:
                if self.token:
                    request = urllib.request.Request(self.server + '/api/v1/auth/revoke',
                        data=b'', headers={'Authorization': 'Bearer ' + self.token}, method='POST')
                    with self.opener.open(request, timeout=5) as response:
                        response.read(1024)
            except (OSError, CLIError):
                pass
            self.token = self.brain = self.connection = None
