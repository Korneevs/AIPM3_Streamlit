"""A malformed batch must be repaired without dropping unsupported image IDs."""
import json

import httpx
import pytest

from aipm3 import uvp_scoring as scoring


def test_retry_specifies_all_ids_without_changing_the_first_request():
    requests = []
    payload = {"questions": [{"id": "q1"}, {"id": "q2"}]}

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        rows = [{"id": "q1"}] if len(requests) == 1 else [{"id": "q1"}, {"id": "q2"}]
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"images": rows})}}]})

    with httpx.Client(base_url="https://example.test/v1/", transport=httpx.MockTransport(handle)) as client:
        result = scoring._call(client, scoring.PANEL[0], scoring.prompts.EVALUATE, payload, [],
                               lambda obj: scoring._rows(obj, "images", ["q1", "q2"]), 500)
    assert [r["id"] for r in result["images"]] == ["q1", "q2"]
    assert len(requests) == 2
    assert len(requests[0]["messages"]) == 2
    assert requests[0]["messages"] == requests[1]["messages"][:2]
    correction = requests[1]["messages"][-1]["content"]
    assert '"q1", "q2"' in correction and "без поддержки" in correction


def test_repair_subset_does_not_ask_for_unflagged_images():
    correction = scoring._format_correction(ValueError("wrong rows"), scoring.prompts.REPAIR,
        {"questions": [{"id": "q1"}, {"id": "q2"}], "repair_ids": ["q1"]})
    assert '"q1"' in correction and '"q2"' not in correction
    assert "ровно 1" in correction


def test_audit_format_retry_does_not_request_image_ratings():
    correction = scoring._format_correction(ValueError("issues missing"), scoring._AUDIT,
                                            {"questions": [{"id": "q1"}]})
    assert "Массив images" not in correction


def test_missing_ids_never_turn_into_a_successful_partial_report():
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"images": [{"id": "q1"}]}'}}]})

    with httpx.Client(base_url="https://example.test/v1/", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(scoring.UVPServiceError):
            scoring._call(client, scoring.PANEL[0], scoring.prompts.EVALUATE,
                          {"questions": [{"id": "q1"}, {"id": "q2"}]}, [],
                          lambda obj: scoring._rows(obj, "images", ["q1", "q2"]), 500)
    assert count == 3
