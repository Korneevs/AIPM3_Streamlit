"""The existing fixed-claim measurement used by the selected M model."""
import base64
import json
from pathlib import Path

import numpy as np
from jsonschema import validate

CONTRACT = json.loads(Path(__file__).with_name('content_precision_contract.json').read_text())


def measure(prepared_video, folder, *, prepared_sha, source_sha, api_key, allow_live):
    from .latest_pipeline import _request, _write_json
    from .runtime_resources import file_sha256
    if file_sha256(prepared_video) != prepared_sha:
        raise ValueError('Content precision media hash mismatch')
    media = 'data:video/mp4;base64,' + base64.b64encode(Path(prepared_video).read_bytes()).decode()

    def request(prompt, path, schema=None, validator=None):
        kwargs = dict(model=CONTRACT['model'], temperature=0,
                      messages=[dict(role='user', content=[dict(type='text', text=prompt),
                                dict(type='image_url', image_url=dict(url=media))])],
                      response_format=(dict(type='json_schema', json_schema=dict(
                          name='content_precision', strict=True, schema=schema))
                          if schema else dict(type='json_object')))
        return _request(kwargs, path, video_sha=prepared_sha, api_key=api_key,
                        allow_live=allow_live, validate=validator)

    def validate_claim(value):
        if not isinstance(value.get('main_claim'), str) or not value['main_claim'].strip():
            raise ValueError('Missing fixed product claim')

    folder = Path(folder) / CONTRACT['protocol_version']
    claim = request(CONTRACT['claim_prompt'], folder / 'claim.json', validator=validate_claim)
    prompt = CONTRACT['prompt'] + '\nfixed_main_claim=' + claim['main_claim']

    def validate_reading(value):
        validate(value, CONTRACT['schema'])
        kinds = [detail['type'] for detail in value['details']]
        expected = (0 if not value['claim'] else 1 if not value['function']
                    else 2 + min(2, len(kinds)))
        if (len(kinds) != len(set(kinds)) or value['score'] != expected
                or value['main_claim'] != claim['main_claim']):
            raise ValueError('Invalid fixed-claim precision reading')

    readings = [request(prompt, folder / f'r{i}.json', CONTRACT['schema'], validate_reading)
                for i in range(1, CONTRACT['repeats'] + 1)]
    result = dict(source_sha=source_sha, prepared_sha=prepared_sha,
                  protocol_version=CONTRACT['protocol_version'], model=CONTRACT['model'],
                  main_claim=claim['main_claim'], readings=readings,
                  mean=float(np.mean([row['score'] for row in readings])))
    _write_json(folder / 'measurement.json', result)
    return result
