import json
from uuid import uuid4
from unittest.mock import patch

import pytest

from ask_ida_cli.brain_connect import BrainConnector, RefuseRedirect, origin
from ask_ida_cli.errors import CLIError


@pytest.mark.parametrize('value', ['https://cortexcommons.com.evil.test', 'http://cortexcommons.com',
    'https://u:p@cortexcommons.com', 'https://cortexcommons.com/path', 'https://cortexcommons.com/?token=x'])
def test_cloud_destination_is_fixed_and_credentials_cannot_be_redirected(value):
    with pytest.raises(CLIError): origin(value)
    with pytest.raises(CLIError): RefuseRedirect().redirect_request(None)


@pytest.mark.parametrize('value', ['http://localhost:8000', 'http://192.168.1.1:80',
    'http://127.0.0.1:8000/commands', 'https://example.org:80'])
def test_runtime_is_explicit_numeric_loopback_only(value):
    with pytest.raises(CLIError): BrainConnector(value)


def test_approval_is_required_before_any_network():
    client = BrainConnector('http://127.0.0.1:8000')
    with patch.object(client, '_json') as network:
        with pytest.raises(CLIError): client.connect(username='u', password='secret', title='Brain')
        network.assert_not_called()


def test_transport_sends_only_fixed_metadata_and_bounded_work_to_local_runtime():
    client = BrainConnector('http://127.0.0.1:8000')
    brain, connection, request = str(uuid4()), str(uuid4()), str(uuid4())
    info = dict(artifact_sha256='a'*64, artifact_format='ida.neurocode-package.v1',
                storage_precision='fp32', master_format='fp32')
    reply = dict(continuation='a', token_bytes=[97], generated_bytes=1, seconds=0.1,
        state_policy='reset_per_request', inference_engine='ida_cli',
        inference_adapter='ida.neurocode.byte-completion.cpu.v1', observed_backend='torch_cpu')
    calls = []
    def exchange(base, path, payload=None):
        calls.append((base,path,payload))
        if path == '/cortex/metadata': return info
        if path == '/api/v1/account/login': return {'token':'t'*43}
        if path == '/api/v1/local-brains': return dict(brain_id=brain, connection_id=connection)
        if path.endswith('/poll'): return {'request':dict(request_id=request,prompt='x',max_new_bytes=1)}
        if path == '/cortex/complete': return reply
        return {}
    with patch.object(client, '_json', side_effect=exchange):
        receipt=client.connect(username='user',password='hidden',title='Brain',approved=True,justification='Explicit sharing')
        assert receipt['brain_id']==brain
        assert client.run(limit=1)==1
    local_calls=[(path,payload) for base,path,payload in calls if base==client.runtime]
    assert local_calls==[('/cortex/metadata',None),('/cortex/complete',{'prompt':'x','max_new_bytes':1})]
    assert 'hidden' not in repr(client) and 't'*43 not in repr(client)
    assert calls[2][2]==dict(info,title='Brain')


def test_server_cannot_send_commands_or_local_paths():
    client=BrainConnector('http://127.0.0.1:8000')
    client.brain=str(uuid4()); client.connection=str(uuid4())
    with patch.object(client,'_json',return_value={'request':{'command':'execute','path':'secret'}}) as exchange:
        with pytest.raises(CLIError): client.run(limit=1)
        assert exchange.call_count==1


def test_metadata_cannot_upload_extra_files_or_paths():
    client=BrainConnector('http://127.0.0.1:8000')
    with patch.object(client,'_json',return_value={'path':'private.safetensors'}) as exchange:
        with pytest.raises(CLIError):
            client.connect(username='u',password='p',title='Brain',approved=True,justification='Sharing')
        assert exchange.call_count==1
