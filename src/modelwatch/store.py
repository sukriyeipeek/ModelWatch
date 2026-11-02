"""SQLite persistence with per-operation connections and immutable idempotent writes."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

class Conflict(ValueError):
    pass

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), occurred_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_by_run ON events(run_id,occurred_at,id);
                CREATE TABLE IF NOT EXISTS labels (id TEXT PRIMARY KEY REFERENCES events(id), payload TEXT NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, table, identifier):
        assert table in ('models', 'runs', 'events', 'labels')
        with self.connection() as db:
            row = db.execute(f'SELECT payload FROM {table} WHERE id=?', (identifier,)).fetchone()
        return json.loads(row['payload']) if row else None

    def put(self, table, identifier, payload, **columns):
        assert table in ('models', 'runs', 'events', 'labels')
        assert set(columns) <= {'model_id', 'run_id', 'occurred_at'}
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute(f'SELECT payload FROM {table} WHERE id=?', (identifier,)).fetchone()
            if table == 'runs':
                payload = dict(payload, created_at=json.loads(row['payload'])['created_at'] if row else datetime.now(timezone.utc).isoformat())
            packed = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
            if row:
                if row['payload'] != packed:
                    raise Conflict('Identifier already exists with different content')
                return False
            names = ['id', 'payload', *columns]
            marks = ','.join('?' for _ in names)
            db.execute(f'INSERT INTO {table} ({",".join(names)}) VALUES ({marks})',
                       [identifier, packed, *columns.values()])
        return True

    def models(self):
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT payload FROM models ORDER BY id')]

    def runs(self, model_id):
        with self.connection() as db:
            rows = db.execute('SELECT payload FROM runs WHERE model_id=? ORDER BY rowid DESC', (model_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def events(self, run_id, limit=None, offset=0):
        query = '''SELECT e.payload,l.payload AS feedback FROM events e
                   LEFT JOIN labels l ON e.id=l.id WHERE e.run_id=? ORDER BY e.occurred_at DESC,e.id DESC'''
        params = [run_id]
        if limit is not None:
            query += ' LIMIT ? OFFSET ?'
            params += [limit, offset]
        with self.connection() as db:
            rows = db.execute(query, params).fetchall()
        result = []
        for row in rows:
            item = json.loads(row['payload'])
            feedback = json.loads(row['feedback']) if row['feedback'] else None
            item['ground_truth'] = feedback['label'] if feedback else None
            item['label_source'] = feedback['source'] if feedback else None
            result.append(item)
        return result

    def count(self, run_id):
        with self.connection() as db:
            return db.execute('SELECT COUNT(*) FROM events WHERE run_id=?', (run_id,)).fetchone()[0]
