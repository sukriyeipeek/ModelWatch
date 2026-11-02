"""API contracts, persistence, isolation, and delayed-label accounting."""
import json

from fastapi.testclient import TestClient
import pytest

from modelwatch.api import create_app


def model(identifier='model-v1'):
    return dict(model_id=identifier, name='Fixture model', version='v1', classes=['a', 'b'])


def run(identifier='monitor', model_id='model-v1', role='monitor', source='live', **metadata):
    return dict(run_id=identifier, model_id=model_id, role=role, source=source, metadata=metadata)


def event(identifier='event-1', run_id='monitor', **overrides):
    result = dict(event_id=identifier, run_id=run_id, occurred_at='2026-01-01T00:00:00Z',
                  status='ok', latency_ms=2.0, predicted_label='a', confidence=.8,
                  features={'text_length': 20, 'word_count': 4})
    result.update(overrides)
    return result


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / 'events.sqlite3', api_key=''), raise_server_exceptions=False) as active:
        assert active.post('/api/models', json=model()).status_code == 200
        assert active.post('/api/runs', json=run()).status_code == 200
        yield active


def test_event_retry_is_idempotent_and_conflicting_content_rejected(client):
    assert client.post('/api/events', json=event()).json()['created'] is True
    assert client.post('/api/events', json=event()).json()['created'] is False
    assert client.post('/api/events', json=event(latency_ms=3.0)).status_code == 409
    assert client.get('/api/events', params={'run_id': 'monitor'}).json()['total'] == 1
    assert client.get('/api/overview', params={'run_id': 'monitor'}).json()['metrics']['traffic']['requests'] == 1


def test_delayed_labels_change_quality_without_mutating_prediction(client):
    client.post('/api/events', json=event())
    client.post('/api/events', json=event('event-2', predicted_label='b'))
    before = client.get('/api/overview', params={'run_id': 'monitor'}).json()['metrics']['quality']
    assert before['accuracy'] is None and before['macro_f1'] is None
    feedback = dict(event_id='event-1', label='a', source='human_review', labeled_at='2026-01-02T00:00:00Z')
    assert client.post('/api/labels', json=feedback).json()['created'] is True
    assert client.post('/api/labels', json=feedback).json()['created'] is False
    first = client.get('/api/overview', params={'run_id': 'monitor'}).json()['metrics']['quality']
    assert first['accuracy'] == 1 and first['label_coverage'] == .5
    feedback['event_id'] = 'event-2'
    assert client.post('/api/labels', json=feedback).status_code == 200
    after = client.get('/api/overview', params={'run_id': 'monitor'}).json()['metrics']['quality']
    assert after['accuracy'] == .5 and after['label_coverage'] == 1
    feedback['label'] = 'b'
    assert client.post('/api/labels', json=feedback).status_code == 409
    second = next(e for e in client.get('/api/events', params={'run_id': 'monitor'}).json()['events']
                  if e['event_id'] == 'event-2')
    assert second['predicted_label'] == 'b' and second['ground_truth'] == 'a'


@pytest.mark.parametrize('overrides', [
    {'occurred_at': 'not-a-date'}, {'occurred_at': '2026-01-01T00:00:00'},
    {'predicted_label': 'unknown'}, {'text': 'Raw sensitive customer message'},
    {'confidence': 1.1}, {'latency_ms': -1},
    {'features': {'text_length': True, 'word_count': 1}},
    {'status': 'error'}, {'error_type': 'Timeout'},
])
def test_invalid_event_never_enters_storage(client, overrides):
    assert client.post('/api/events', json=event(**overrides)).status_code == 422
    assert client.get('/api/events', params={'run_id': 'monitor'}).json()['total'] == 0


@pytest.mark.parametrize('field,value', [('latency_ms', float('nan')), ('latency_ms', float('inf')),
                                       ('confidence', float('-inf'))])
def test_nonfinite_json_numbers_rejected_without_server_error(client, field, value):
    response = client.post('/api/events', content=json.dumps(event(**{field: value})),
                           headers={'Content-Type': 'application/json'})
    assert response.status_code == 422, response.text


def test_error_events_count_as_requests_and_latency_but_not_quality(client):
    client.post('/api/events', json=event())
    failed = event('failed', status='error', predicted_label=None, confidence=None,
                   error_type='Timeout', latency_ms=100.0)
    assert client.post('/api/events', json=failed).status_code == 200
    metrics = client.get('/api/overview', params={'run_id': 'monitor'}).json()['metrics']
    assert metrics['traffic'] == dict(requests=2, successes=1, errors=1, error_rate=.5)
    assert metrics['latency']['p50_ms'] == 51
    assert metrics['quality']['label_coverage'] == 0
    assert client.post('/api/labels', json=dict(event_id='failed', label='a', source='human_review',
        labeled_at='2026-01-02T00:00:00Z')).status_code == 422


def test_api_key_protects_reads_and_writes(tmp_path):
    with TestClient(create_app(tmp_path / 'secured.sqlite3', api_key='unit-test-key')) as secured:
        for headers in ({}, {'X-API-Key': 'wrong'}):
            assert secured.get('/api/models', headers=headers).status_code == 401
            assert secured.post('/api/models', json=model(), headers=headers).status_code == 401
        headers = {'X-API-Key': 'unit-test-key'}
        assert secured.post('/api/models', json=model(), headers=headers).status_code == 200
        assert secured.get('/api/models', headers=headers).json()['models'][0]['model_id'] == 'model-v1'


def test_cross_model_reference_and_cross_source_reference_rejected(client):
    assert client.post('/api/models', json=model('other-model')).status_code == 200
    assert client.post('/api/runs', json=run('reference', role='reference')).status_code == 200
    assert client.post('/api/runs', json=run('bad-model', model_id='other-model',
        reference_run_id='reference')).status_code == 422
    assert client.post('/api/runs', json=run('bad-source', source='benchmark_replay',
        reference_run_id='reference')).status_code == 422


@pytest.mark.parametrize('bad_reference', [{}, [], 42])
def test_malformed_reference_identifier_returns_validation_error(client, bad_reference):
    response = client.post('/api/runs', json=run('invalid-reference', reference_run_id=bad_reference))
    assert response.status_code == 422, response.text


def test_run_retry_is_idempotent(client):
    first = client.get('/api/runs', params={'model_id': 'model-v1'}).json()['runs'][0]
    assert client.post('/api/runs', json=run()).json()['created'] is False
    second = client.get('/api/runs', params={'model_id': 'model-v1'}).json()['runs'][0]
    assert first == second
    assert client.post('/api/runs', json=run(role='reference')).status_code == 409


def test_database_survives_app_restart(tmp_path):
    path = tmp_path / 'persistent.sqlite3'
    with TestClient(create_app(path, api_key='')) as first:
        first.post('/api/models', json=model())
        first.post('/api/runs', json=run())
        first.post('/api/events', json=event())
    with TestClient(create_app(path, api_key='')) as second:
        result = second.get('/api/events', params={'run_id': 'monitor'}).json()
        assert result['total'] == 1 and result['events'][0]['event_id'] == 'event-1'


def test_models_runs_and_queries_are_isolated(client):
    client.post('/api/events', json=event())
    client.post('/api/models', json=model('other-model'))
    client.post('/api/runs', json=run('other-run', model_id='other-model'))
    client.post('/api/events', json=event('other-event', run_id='other-run'))
    assert len(client.get('/api/runs', params={'model_id': 'other-model'}).json()['runs']) == 1
    rows = client.get('/api/events', params={'run_id': 'other-run'}).json()
    assert rows['total'] == 1 and rows['events'][0]['event_id'] == 'other-event'
    assert client.get('/api/events', params={'run_id': "monitor' OR 1=1 --"}).status_code == 404
    assert client.get('/api/models').status_code == 200


def test_labels_cannot_precede_events_or_misrepresent_live_data(client):
    client.post('/api/events', json=event())
    feedback = dict(event_id='event-1', label='a', source='human_review', labeled_at='2025-01-01T00:00:00Z')
    assert client.post('/api/labels', json=feedback).status_code == 422
    feedback.update(labeled_at='2026-01-02T00:00:00Z', source='official_dataset')
    assert client.post('/api/labels', json=feedback).status_code == 422
