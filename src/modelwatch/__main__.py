"""Local-first server and reproducible dataset replay commands."""
import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog='modelwatch')
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve', help='Start the dashboard and ingestion API')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8010)
    serve.add_argument('--db', default='runtime/modelwatch.sqlite3')
    replay = sub.add_parser('replay', help='Train and measure the official Banking77 benchmark')
    replay.add_argument('--data-dir', type=Path, default=Path('data/banking77'))
    replay.add_argument('--db', default='runtime/modelwatch.sqlite3')
    replay.add_argument('--report', type=Path, default=Path('reports/latest-replay.json'))
    args = parser.parse_args()
    from modelwatch.api import create_app
    if args.command == 'serve':
        import uvicorn
        if args.host not in ('127.0.0.1', 'localhost', '::1') and not os.getenv('MODELWATCH_API_KEY'):
            parser.error('Set MODELWATCH_API_KEY before binding beyond loopback. Use TLS through a reverse proxy.')
        uvicorn.run(create_app(args.db), host=args.host, port=args.port)
    else:
        from fastapi.testclient import TestClient
        from modelwatch.replay import replay as run_replay
        headers = {'X-API-Key': os.environ['MODELWATCH_API_KEY']} if os.getenv('MODELWATCH_API_KEY') else {}
        with TestClient(create_app(args.db), headers=headers) as client:
            report = run_replay(client, args.data_dir)
            response = client.get('/api/overview', params={'run_id': report['monitor_run_id']})
            response.raise_for_status()
            overview = response.json()
            report['metrics'] = overview['metrics']
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'report': str(args.report), 'monitor_run_id': report['monitor_run_id'],
                          'requests': report['metrics']['traffic']['requests'],
                          'accuracy': report['metrics']['quality']['accuracy']}))

if __name__ == '__main__':
    main()
