"""Single-origin dashboard and event ingestion API. Default use is local-only."""
from datetime import datetime, timezone
import hmac
import os
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from modelwatch.schemas import Model, Run, Event, Feedback
from modelwatch.store import Store, Conflict
from modelwatch.metrics import summarize

STATIC = Path(__file__).parent / 'static'

def create_app(db_path=None, api_key=None):
    store = Store(db_path or os.getenv('MODELWATCH_DB', 'runtime/modelwatch.sqlite3'))
    key = api_key if api_key is not None else os.getenv('MODELWATCH_API_KEY')

    def authorize(x_api_key: str | None = Header(default=None)):
        if key and not hmac.compare_digest(x_api_key or '', key):
            raise HTTPException(401, 'Valid X-API-Key required')

    app = FastAPI(title='ModelWatch', version='0.1.0')
    app.state.store = store

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Avoid echoing untrusted request content, including non-JSON NaN/Infinity.
        return JSONResponse(status_code=422, content={'detail': [
            {'loc': list(item['loc']), 'msg': item['msg'], 'type': item['type']}
            for item in error.errors()]})

    def require(table, identifier):
        value = store.get(table, identifier)
        if value is None:
            raise HTTPException(404, f'{table.rstrip("s")} not found')
        return value

    def put(table, identifier, payload, **columns):
        try:
            created = store.put(table, identifier, payload, **columns)
        except Conflict as error:
            raise HTTPException(409, str(error)) from error
        return {'id': identifier, 'created': created}

    @app.get('/health')
    def health():
        return {'status': 'ok', 'version': '0.1.0'}

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(STATIC / 'index.html')

    app.mount('/static', StaticFiles(directory=STATIC), name='static')

    @app.post('/api/models', dependencies=[Depends(authorize)])
    def register_model(model: Model):
        return put('models', model.model_id, model.model_dump(mode='json'))

    @app.get('/api/models', dependencies=[Depends(authorize)])
    def models():
        return {'models': store.models()}

    @app.post('/api/runs', dependencies=[Depends(authorize)])
    def register_run(run: Run):
        require('models', run.model_id)
        reference_id = run.metadata.get('reference_run_id')
        if reference_id:
            ref = require('runs', reference_id)
            if ref['model_id'] != run.model_id or ref['role'] != 'reference' or ref['source'] != run.source:
                raise HTTPException(422, 'Reference must belong to the same model and source and have reference role')
        payload = run.model_dump(mode='json')
        return put('runs', run.run_id, payload, model_id=run.model_id)

    @app.get('/api/runs', dependencies=[Depends(authorize)])
    def runs(model_id: str):
        require('models', model_id)
        return {'runs': store.runs(model_id)}

    @app.post('/api/events', dependencies=[Depends(authorize)])
    def ingest(event: Event):
        run = require('runs', event.run_id)
        model = require('models', run['model_id'])
        if event.predicted_label is not None and event.predicted_label not in model['classes']:
            raise HTTPException(422, 'Prediction not in registered classes')
        payload = event.model_dump(mode='json')
        return put('events', event.event_id, payload, run_id=event.run_id, occurred_at=payload['occurred_at'])

    @app.post('/api/labels', dependencies=[Depends(authorize)])
    def label(feedback: Feedback):
        event = require('events', feedback.event_id)
        run = require('runs', event['run_id'])
        model = require('models', run['model_id'])
        if event['status'] != 'ok' or feedback.label not in model['classes']:
            raise HTTPException(422, 'Labels require a successful event and a registered class')
        if feedback.source == 'official_dataset' and run['source'] != 'benchmark_replay':
            raise HTTPException(422, 'Official-dataset labels belong only to benchmark replay runs')
        if feedback.labeled_at < datetime.fromisoformat(event['occurred_at'].replace('Z', '+00:00')):
            raise HTTPException(422, 'Label arrival cannot precede its event')
        return put('labels', feedback.event_id, feedback.model_dump(mode='json'))

    @app.get('/api/events', dependencies=[Depends(authorize)])
    def events(run_id: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        require('runs', run_id)
        return {'events': store.events(run_id, limit, offset), 'total': store.count(run_id)}

    @app.get('/api/overview', dependencies=[Depends(authorize)])
    def overview(run_id: str):
        run = require('runs', run_id)
        model = require('models', run['model_id'])
        reference = None
        reference_id = run['metadata'].get('reference_run_id')
        if reference_id:
            reference = require('runs', reference_id)
        elif run['role'] == 'monitor':
            candidates = [r for r in store.runs(run['model_id']) if r['role'] == 'reference'
                          and r['source'] == run['source']
                          and r['metadata'].get('replay_id') == run['metadata'].get('replay_id')]
            reference = candidates[0] if candidates else None
        current = store.events(run_id)
        reference_events = store.events(reference['run_id']) if reference else []
        return {'model': model, 'run': run, 'reference_run': reference,
                'metrics': summarize(current, reference_events, model['classes']),
                'recent_events': current[:50], 'provenance': run['metadata']}

    return app
